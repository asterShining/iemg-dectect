"""Digital signal processing for the surface-EMG pipeline.

Two complementary processing paths are kept strictly separate so the
methodological distinction is explicit:

- **Offline** (:func:`process_offline`) uses ``sosfiltfilt`` (zero-phase,
  no group delay) on the entire recording. This is the path used by
  the analysis tab when reviewing a saved EDF file.
- **Real-time** (:class:`RealtimeFilterState`) uses ``sosfilt`` with
  persistent ``zi`` state (causal, stateful). This is the path used by
  the acquisition tab so each newly arrived block is processed
  seamlessly with the running history.

The same Butterworth design (order 2 by default) is used by both paths
to keep the displayed waveforms comparable.

Pipeline
--------
1. **Notch 50 Hz** to suppress mains interference.
2. **Band-pass 20-450 Hz** to remove motion artefacts and high-frequency
   noise outside the surface-EMG informative band.
3. **Rectification** (absolute value).
4. **Low-pass 5 Hz** to extract the envelope.
5. **Sliding RMS** as a power-based amplitude estimate.
6. **Welch PSD** for MNF and MDF computation.

References
----------
.. [1] De Luca CJ. The Use of Surface Electromyography in Biomechanics.
   *J Appl Biomech* (1997).
.. [2] Phinyomark A, Phukpattaranont P, Limsakul C. Feature reduction
   and selection for EMG signal classification. *Expert Syst Appl* (2012).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import (
    iirfilter,
    iirnotch,
    sosfilt,
    sosfilt_zi,
    sosfiltfilt,
    tf2sos,
    welch,
)

from emgteach.i18n import tr

if TYPE_CHECKING:
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]


__all__ = [
    "LiveQualityMonitor",
    "MetricsCsvWriter",
    "OnsetDetector",
    "QualityStatus",
    "RealtimeFilterState",
    "RealtimeMetricAccumulator",
    "assess_channel_quality",
    "compute_iemg",
    "compute_psd_mnf_mdf",
    "compute_realtime_window_metrics",
    "compute_recording_metrics_1s",
    "compute_segments",
    "design_bandpass",
    "design_lowpass",
    "design_notch",
    "detect_acquisition_problems",
    "detect_onsets",
    "metrics_csv_path",
    "mmg_envelope",
    "process_offline",
    "read_metrics_csv",
    "tremor_spectrum",
    "validate_bandpass",
]


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------


def validate_bandpass(f_low: float, f_high: float, fs: float) -> None:
    """Validate that band-pass cut-offs strictly satisfy ``0 < f_low < f_high < fs/2``.

    Raises :class:`ValueError` instead of silently clamping ``f_high`` near
    Nyquist, so the configured filter always matches the executed filter.
    """
    fs_val = float(fs)
    low_val = float(f_low)
    high_val = float(f_high)
    if fs_val <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive (got {fs_val} Hz).")
    nyquist = fs_val / 2.0
    if not (0.0 < low_val < high_val < nyquist):
        raise ValueError(
            f"Invalid band-pass cut-offs ({low_val:g}-{high_val:g} Hz) for "
            f"sampling rate {fs_val:g} Hz (Nyquist {nyquist:g} Hz); "
            f"requires 0 < f_low < f_high < fs/2."
        )


def design_bandpass(
    f_low: float, f_high: float, fs: float, order: int = 2
) -> FloatArray:
    """Butterworth band-pass SOS coefficients for the EMG informative band."""
    validate_bandpass(f_low, f_high, fs)
    return iirfilter(
        order, [f_low, f_high], btype="band", fs=fs, ftype="butter", output="sos"
    )


def design_notch(
    f_notch: float,
    fs: float,
    q: float = 30.0,
    harmonics: bool = True,
    f_high: float = 450.0,
) -> FloatArray:
    """Mains-interference notch as a cascade of narrow IIR notches.

    A single :func:`scipy.signal.iirnotch` (quality factor ``q``) is far
    narrower and deeper than the previous 2nd-order Butterworth band-stop:
    at ``q=30`` the 50 Hz notch is only ~1.7 Hz wide, so it removes the
    mains line while leaving the surrounding EMG spectrum essentially
    untouched.

    Real mains interference is not a pure sinusoid; it also injects odd
    (and, with rectifying loads, even) harmonics at 100, 150, 200 … Hz that
    fall squarely inside the 20-450 Hz EMG band and that the old single-band
    notch left in place. When ``harmonics`` is ``True`` (the default) the
    filter therefore stacks one notch at ``f_notch`` and one at each harmonic
    up to ``f_high`` (bounded by the Nyquist frequency), returning the
    combined SOS cascade.

    Parameters
    ----------
    f_notch : float
        Mains fundamental frequency (Hz), typically 50 (EU) or 60 (US).
    fs : float
        Sampling frequency (Hz).
    q : float, optional
        Quality factor of each notch (higher = narrower). Default 30.
    harmonics : bool, optional
        Also notch the integer harmonics of ``f_notch`` inside the band.
        Default ``True``.
    f_high : float, optional
        Upper bound (Hz) for harmonic notches; harmonics at or above this
        (or above Nyquist) are not added, as the band-pass removes them.
        Default 450.

    Returns
    -------
    ndarray
        Second-order-sections array of shape ``(n_notches, 6)``.
    """
    f_max = min(float(f_high), fs / 2.0)
    freqs = [float(f_notch)]
    if harmonics:
        k = 2
        while f_notch * k < f_max - 1.0:
            freqs.append(f_notch * k)
            k += 1
    sections = [tf2sos(*iirnotch(f0, q, fs=fs)) for f0 in freqs]
    return np.vstack(sections)


def design_lowpass(f_cut: float, fs: float, order: int = 2) -> FloatArray:
    """Butterworth low-pass SOS coefficients (envelope smoothing)."""
    return iirfilter(
        order, f_cut, btype="low", fs=fs, ftype="butter", output="sos"
    )


# ---------------------------------------------------------------------------
# Real-time path (causal, stateful)
# ---------------------------------------------------------------------------


class RealtimeFilterState:
    """Stateful filter chain for the real-time acquisition path.

    Holds notch, band-pass and envelope filter coefficients along with
    their ``zi`` initial-conditions state, so that each call to
    :meth:`process_block` continues seamlessly from the previous block.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    f_low, f_high : float, optional
        Band-pass cut-off frequencies (Hz). Defaults 20 and 450.
    f_notch : float, optional
        Mains notch frequency (Hz). Default 50.
    f_env : float, optional
        Envelope low-pass cut-off (Hz). Default 5.
    order : int, optional
        Butterworth order for band-pass and envelope filters. Default 2.
    notch_enabled : bool, optional
        Whether the mains notch filter is applied in :meth:`process_block`.
        Default ``True`` (legacy behaviour preserved).
    notch_harmonics : bool, optional
        Whether integer harmonics of ``f_notch`` are notched in addition to
        the fundamental. Default ``True`` (legacy behaviour preserved).
        Pass ``False`` explicitly for single-50 Hz notch in LHE7904 sEMG.
    """

    def __init__(
        self,
        fs: float,
        f_low: float = 20.0,
        f_high: float = 450.0,
        f_notch: float = 50.0,
        f_env: float = 5.0,
        order: int = 2,
        *,
        notch_enabled: bool = True,
        notch_harmonics: bool = True,
    ) -> None:
        self.fs = float(fs)
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.f_notch = float(f_notch)
        self.f_env = float(f_env)
        self.order = int(order)
        self.notch_enabled = bool(notch_enabled)
        self.notch_harmonics = bool(notch_harmonics)

        self.sos_band = design_bandpass(f_low, f_high, fs, order)
        self.sos_notch: FloatArray | None = (
            design_notch(
                f_notch,
                fs,
                harmonics=self.notch_harmonics,
                f_high=f_high,
            )
            if self.notch_enabled
            else None
        )
        self.sos_env = design_lowpass(f_env, fs, order)

        # Initial conditions are zeroed so the first samples do not see
        # a spurious step. ``sosfilt_zi`` returns shape (n_sections, 2);
        # multiplied by 0.0 gives a same-shape zero array.
        self.zi_band: FloatArray = sosfilt_zi(self.sos_band) * 0.0
        self.zi_notch: FloatArray = (
            sosfilt_zi(self.sos_notch) * 0.0
            if self.sos_notch is not None
            else np.zeros((0, 2), dtype=np.float64)
        )
        self.zi_env: FloatArray = sosfilt_zi(self.sos_env) * 0.0

    def reset(self) -> None:
        """Reset all causal filter delay states (``zi``) to zero after a gap."""
        self.zi_band = sosfilt_zi(self.sos_band) * 0.0
        self.zi_notch = (
            sosfilt_zi(self.sos_notch) * 0.0
            if self.sos_notch is not None
            else np.zeros((0, 2), dtype=np.float64)
        )
        self.zi_env = sosfilt_zi(self.sos_env) * 0.0

    def process_block(
        self, emg_mv: FloatArray | np.ndarray
    ) -> tuple[FloatArray, FloatArray]:
        """Apply notch -> band-pass -> rectify -> low-pass envelope.

        The ``zi`` states are mutated in place so the next call continues
        seamlessly from the current end of the buffer.

        Parameters
        ----------
        emg_mv : array-like
            Raw EMG block in millivolts.

        Returns
        -------
        emg_filtered : ndarray
            Notch + band-pass filtered EMG (mV).
        emg_envelope : ndarray
            Rectified + low-pass envelope (mV).
        """
        emg = np.asarray(emg_mv, dtype=np.float64)

        if self.notch_enabled and self.sos_notch is not None:
            emg_notched, self.zi_notch = sosfilt(
                self.sos_notch, emg, zi=self.zi_notch
            )
        else:
            emg_notched = emg
        emg_filtered, self.zi_band = sosfilt(
            self.sos_band, emg_notched, zi=self.zi_band
        )
        emg_rect = np.abs(emg_filtered)
        emg_envelope, self.zi_env = sosfilt(self.sos_env, emg_rect, zi=self.zi_env)

        return emg_filtered, emg_envelope


# ---------------------------------------------------------------------------
# Offline path (zero-phase, sosfiltfilt)
# ---------------------------------------------------------------------------


def process_offline(
    emg_raw: FloatArray | np.ndarray,
    fs: float,
    f_low: float = 20.0,
    f_high: float = 450.0,
    f_notch: float = 50.0,
    f_env: float = 5.0,
    rms_window_ms: float = 50.0,
    order: int = 2,
) -> dict[str, FloatArray]:
    """Run the full zero-phase EMG pipeline on a complete recording.

    Order: notch 50 Hz -> band-pass 20-450 Hz -> rectify -> low-pass
    -> sliding RMS. Reflective padding of 500 ms is added before each
    ``sosfiltfilt`` to avoid border artefacts.

    Parameters
    ----------
    emg_raw : array-like
        Raw EMG signal in millivolts.
    fs : float
        Sampling frequency (Hz).
    f_low, f_high : float, optional
        Band-pass cut-off frequencies (default 20, 450 Hz).
    f_notch : float, optional
        Mains notch frequency (default 50 Hz).
    f_env : float, optional
        Envelope low-pass cut-off (default 5 Hz).
    rms_window_ms : float, optional
        Sliding-RMS window in milliseconds (default 50).
    order : int, optional
        Butterworth order (default 2).

    Returns
    -------
    dict
        Keys: ``emg_filtered``, ``emg_rectified``, ``emg_envelope``,
        ``rms_sliding``, ``emg_envelope_normalised``.
    """
    emg_raw = np.asarray(emg_raw, dtype=np.float64)

    sos_notch = design_notch(f_notch, fs, f_high=f_high)
    sos_band = design_bandpass(f_low, f_high, fs, order)
    sos_env = design_lowpass(f_env, fs, order)

    # Reflective padding of 500 ms on each side keeps sosfiltfilt's
    # forward+backward pass clean of border artefacts.
    pad_len = int(fs * 0.5)
    emg_padded = np.concatenate(
        [emg_raw[:pad_len][::-1], emg_raw, emg_raw[-pad_len:][::-1]]
    )

    emg_notched_full = sosfiltfilt(sos_notch, emg_padded)
    emg_filt_full = sosfiltfilt(sos_band, emg_notched_full)
    emg_filtered = emg_filt_full[pad_len:-pad_len]
    emg_rectified = np.abs(emg_filtered)

    emg_rect_padded = np.concatenate(
        [emg_rectified[:pad_len][::-1], emg_rectified, emg_rectified[-pad_len:][::-1]]
    )
    emg_env_full = sosfiltfilt(sos_env, emg_rect_padded)
    emg_envelope = emg_env_full[pad_len:-pad_len]

    rms_window = max(1, int(rms_window_ms / 1000.0 * fs))
    kernel = np.ones(rms_window) / rms_window
    rms_sliding = np.sqrt(np.convolve(emg_filtered**2, kernel, mode="same"))

    max_env = float(np.max(emg_envelope))
    if max_env <= 0:
        max_env = 1.0
    emg_envelope_normalised = emg_envelope / max_env

    return {
        "emg_filtered": emg_filtered,
        "emg_rectified": emg_rectified,
        "emg_envelope": emg_envelope,
        "rms_sliding": rms_sliding,
        "emg_envelope_normalised": emg_envelope_normalised,
    }


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def compute_psd_mnf_mdf(
    emg_filtered: FloatArray | np.ndarray,
    fs: float,
    f_low: float = 20.0,
    f_high: float = 450.0,
    nperseg: int | None = None,
) -> dict[str, Any]:
    """Welch PSD and the two fatigue-related spectral metrics.

    ``nperseg`` defaults to one second of samples; a caller with a shorter
    stretch (one contraction) passes its own length, so the estimate is made
    once over what there is instead of Welch shrinking the window and
    warning about it.

    Parameters
    ----------
    emg_filtered : array-like
        Filtered EMG signal (post notch + band-pass).
    fs : float
        Sampling frequency (Hz).
    f_low, f_high : float, optional
        Frequency band over which MNF and MDF are computed.

    Returns
    -------
    dict
        ``frequencies`` (1-D array, Hz), ``psd`` (1-D array, mV**2/Hz),
        ``mnf`` (mean frequency, Hz), ``mdf`` (median frequency, Hz).
    """
    emg = np.asarray(emg_filtered, dtype=np.float64)
    # Clamped to the signal: Welch shortens an over-long window by itself but
    # keeps the overlap it was given, and an overlap that is no longer smaller
    # than the window is an error, not a spectrum. Anything under a second
    # asked for the default one-second window and raised.
    ventana = int(fs) if nperseg is None else int(nperseg)
    ventana = max(8, min(ventana, emg.size)) if emg.size else 8
    frequencies, psd = welch(emg, fs=fs, nperseg=ventana, noverlap=ventana // 2)

    band_mask = (frequencies >= f_low) & (frequencies <= f_high)
    f_band = frequencies[band_mask]
    psd_band = psd[band_mask]

    # Use trapezoidal integration consistently for both MNF (MPF) and MDF (MF)
    # so the total power denominator and cumulative power curve are identical.
    acumulada = cumulative_trapezoid(psd_band, x=f_band, initial=0.0)
    total_power = float(acumulada[-1]) if acumulada.size else 0.0
    if total_power > 0.0:
        mnf = float(trapezoid(f_band * psd_band, x=f_band) / total_power)
        mdf = float(np.interp(total_power / 2.0, acumulada, f_band))
    else:
        mnf = 0.0
        mdf = 0.0

    return {
        "frequencies": f_band,
        "psd": psd_band,
        "mnf": mnf,
        "mdf": mdf,
    }


def compute_iemg(
    emg_filtered: FloatArray | np.ndarray,
    fs: float,
) -> float:
    """Compute integrated EMG (iEMG) in ``µV·s`` over a filtered EMG window.

    Uses the un-smoothed rectified filtered waveform in millivolts:

    .. math::
        \\mathrm{iEMG} = \\frac{1000}{f_s} \\sum_{n=0}^{N-1} |x[n]|

    Never pass the 5 Hz smoothed envelope to this function.
    """
    fs_val = float(fs)
    if fs_val <= 0.0:
        raise ValueError(f"Sampling rate fs must be positive (got {fs_val} Hz).")
    emg = np.asarray(emg_filtered, dtype=np.float64)
    if emg.size == 0:
        return 0.0
    if not np.all(np.isfinite(emg)):
        return float("nan")
    return float((1000.0 / fs_val) * np.sum(np.abs(emg)))


def compute_realtime_window_metrics(
    emg_filtered: FloatArray | np.ndarray,
    fs: float,
    *,
    f_low: float = 10.0,
    f_high: float = 500.0,
    window_tainted: bool = False,
) -> dict[str, Any]:
    """Compute synchronized 1-second ``iEMG``, ``MPF`` and ``MF`` on one window.

    Returns ``valid=False`` and ``NaN`` values when the window is tainted by
    dropped samples, does not contain a full 1-second sample count, contains
    non-finite values, or is flat/zero-power. Never substitutes ``0`` for an
    invalid window.
    """
    invalid_result = {
        "valid": False,
        "iemg_uv_s": float("nan"),
        "mpf_hz": float("nan"),
        "mf_hz": float("nan"),
        "iemg": float("nan"),
        "mpf": float("nan"),
        "mf": float("nan"),
    }
    fs_val = float(fs)
    expected_n = round(fs_val)
    if window_tainted or expected_n <= 0:
        return invalid_result

    emg = np.asarray(emg_filtered, dtype=np.float64).reshape(-1)
    if emg.size != expected_n or not np.all(np.isfinite(emg)):
        return invalid_result

    # Flat / zero-amplitude window has no meaningful spectral centroid or median.
    if float(np.std(emg)) <= 1e-12 or float(np.max(np.abs(emg))) <= 1e-12:
        return invalid_result

    iemg_uv_s = compute_iemg(emg, fs_val)
    spec = compute_psd_mnf_mdf(
        emg,
        fs=fs_val,
        f_low=f_low,
        f_high=f_high,
        nperseg=expected_n,
    )
    mpf_hz = float(spec["mnf"])
    mf_hz = float(spec["mdf"])
    if (
        not (np.isfinite(iemg_uv_s) and np.isfinite(mpf_hz) and np.isfinite(mf_hz))
        or mpf_hz <= 0.0
        or mf_hz <= 0.0
    ):
        return invalid_result

    return {
        "valid": True,
        "iemg_uv_s": iemg_uv_s,
        "mpf_hz": mpf_hz,
        "mf_hz": mf_hz,
        "iemg": iemg_uv_s,
        "mpf": mpf_hz,
        "mf": mf_hz,
    }


class RealtimeMetricAccumulator:
    """Accumulate contiguous filtered EMG blocks into exact 1-second windows.

    All three metrics (``iEMG``, ``MPF``, ``MF``) are computed on the exact
    same contiguous 1-second window (``int(round(fs))`` samples). If a sample
    gap occurs inside an incoming block, both the partial accumulated window
    and the discontinuous block are discarded, and an explicit invalid metric
    snapshot (``valid=False``, ``NaN``) is emitted.
    """

    def __init__(
        self,
        fs: float,
        n_channels: int,
        kinds: Sequence[str] | None = None,
        *,
        f_low: float = 10.0,
        f_high: float = 500.0,
    ) -> None:
        self.fs = round(float(fs))
        if self.fs <= 0:
            raise ValueError(f"Sampling rate fs must be positive (got {fs}).")
        self.window_samples = self.fs
        self.n_channels = int(n_channels)
        self.kinds = (
            list(kinds) if kinds is not None else ["EMG"] * self.n_channels
        )
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self._buffers: list[FloatArray] = [
            np.empty(0, dtype=np.float64) for _ in range(self.n_channels)
        ]
        self._total_samples: int = 0

    @property
    def buffered_samples(self) -> int:
        """Number of contiguous samples currently buffered toward the next 1 s window."""
        return int(self._buffers[0].size) if self._buffers else 0

    def reset_partial(self) -> None:
        """Discard any partially accumulated window samples."""
        self._buffers = [
            np.empty(0, dtype=np.float64) for _ in range(self.n_channels)
        ]

    def discard_on_gap(self, block_samples: int) -> dict[str, Any]:
        """Handle a block during which ``lost_frames`` increased.

        Discards both the partial window accumulated prior to this block and
        the discontinuous block itself, advances the sample counter by
        ``block_samples``, and returns an invalid metric snapshot
        (``valid=False``, ``NaN``) timestamped at the end of the block.
        """
        self._total_samples += max(0, int(block_samples))
        self.reset_partial()
        time_s = self._total_samples / float(self.fs)
        nans = [float("nan")] * self.n_channels
        return {
            "time_s": float(time_s),
            "valid": [False] * self.n_channels,
            "iemg_uv_s": list(nans),
            "mpf_hz": list(nans),
            "mf_hz": list(nans),
            "iemg": list(nans),
            "mpf": list(nans),
            "mf": list(nans),
        }

    def push_block(
        self, filt_list: Sequence[FloatArray | np.ndarray] | np.ndarray
    ) -> list[dict[str, Any]]:
        """Append one contiguous block of filtered samples per channel.

        Accepts either a sequence of 1-D channel arrays ``[ch0, ch1, ...]`` or
        a 2-D array of shape ``(n_samples, n_channels)``.
        Returns a list of 1-second metric snapshots completed by this block
        (empty when fewer than ``fs`` contiguous samples have accumulated).
        """
        if isinstance(filt_list, np.ndarray):
            if filt_list.ndim == 2:
                channels_in: Sequence[FloatArray | np.ndarray] = [
                    filt_list[:, c] for c in range(filt_list.shape[1])
                ]
            else:
                channels_in = [filt_list]
        else:
            channels_in = filt_list
        if len(channels_in) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {len(channels_in)}."
            )
        arrays = [
            np.asarray(ch, dtype=np.float64).reshape(-1) for ch in channels_in
        ]
        if not arrays or arrays[0].size == 0:
            return []
        block_len = int(arrays[0].size)
        for arr in arrays[1:]:
            if int(arr.size) != block_len:
                raise ValueError("All channels in filt_list must have equal length.")

        for c in range(self.n_channels):
            if self._buffers[c].size == 0:
                self._buffers[c] = arrays[c].copy()
            else:
                self._buffers[c] = np.concatenate([self._buffers[c], arrays[c]])

        self._total_samples += block_len
        completed: list[dict[str, Any]] = []

        while self._buffers[0].size >= self.window_samples:
            leftover = int(self._buffers[0].size - self.window_samples)
            win_end_sample = self._total_samples - leftover
            time_s = win_end_sample / float(self.fs)

            valid_list: list[bool] = []
            iemg_list: list[float] = []
            mpf_list: list[float] = []
            mf_list: list[float] = []

            for c in range(self.n_channels):
                win_c = self._buffers[c][: self.window_samples]
                self._buffers[c] = self._buffers[c][self.window_samples :]
                if self.kinds[c] != "EMG":
                    valid_list.append(False)
                    iemg_list.append(float("nan"))
                    mpf_list.append(float("nan"))
                    mf_list.append(float("nan"))
                else:
                    res = compute_realtime_window_metrics(
                        win_c,
                        self.fs,
                        f_low=self.f_low,
                        f_high=self.f_high,
                    )
                    valid_list.append(bool(res["valid"]))
                    iemg_list.append(float(res["iemg_uv_s"]))
                    mpf_list.append(float(res["mpf_hz"]))
                    mf_list.append(float(res["mf_hz"]))

            completed.append(
                {
                    "time_s": float(time_s),
                    "valid": valid_list,
                    "iemg_uv_s": iemg_list,
                    "mpf_hz": mpf_list,
                    "mf_hz": mf_list,
                    "iemg": list(iemg_list),
                    "mpf": list(mpf_list),
                    "mf": list(mf_list),
                }
            )

        return completed


def metrics_csv_path(edf_path: str | Path) -> Path:
    """Return the companion ``.metrics.csv`` path for a saved EDF recording."""
    return Path(f"{edf_path}.metrics.csv")


class MetricsCsvWriter:
    """Persist 1-second real-time metric snapshots alongside an EDF recording."""

    def __init__(
        self,
        edf_path: str | Path,
        *,
        fs: int = 4000,
        f_low: float = 10.0,
        f_high: float = 500.0,
        f_notch: float = 50.0,
        notch_enabled: bool = True,
        notch_harmonics: bool = False,
        labels: Sequence[str] | None = None,
        channel_labels: Sequence[str] | None = None,
    ) -> None:
        self.path = metrics_csv_path(edf_path)
        resolved_labels = labels if labels is not None else (channel_labels or ["EMG"])
        self._labels = list(resolved_labels)
        self._file: TextIO | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            f = open(self.path, "w", encoding="utf-8", newline="")
            f.write(
                f"# emgteach_metrics_v1 fs={int(fs)} f_low={float(f_low):g} "
                f"f_high={float(f_high):g} f_notch={float(f_notch):g} "
                f"notch_enabled={int(bool(notch_enabled))} "
                f"notch_harmonics={int(bool(notch_harmonics))}\n"
            )
            f.write("time_s,channel,label,valid,iemg_uv_s,mpf_hz,mf_hz\n")
            f.flush()
            self._file = f
        except OSError:
            self._file = None

    def __enter__(self) -> MetricsCsvWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def write_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Write one 1-second multi-channel snapshot and flush to disk."""
        if self._file is None:
            return
        time_s = float(snapshot["time_s"])
        valid_list = snapshot["valid"]
        iemg_list = snapshot.get("iemg_uv_s", snapshot.get("iemg", []))
        mpf_list = snapshot.get("mpf_hz", snapshot.get("mpf", []))
        mf_list = snapshot.get("mf_hz", snapshot.get("mf", []))
        try:
            for c, label in enumerate(self._labels):
                is_valid = bool(valid_list[c]) if c < len(valid_list) else False
                iemg_v = float(iemg_list[c]) if c < len(iemg_list) else float("nan")
                mpf_v = float(mpf_list[c]) if c < len(mpf_list) else float("nan")
                mf_v = float(mf_list[c]) if c < len(mf_list) else float("nan")
                iemg_str = f"{iemg_v:.6f}" if is_valid and np.isfinite(iemg_v) else "NaN"
                mpf_str = f"{mpf_v:.6f}" if is_valid and np.isfinite(mpf_v) else "NaN"
                mf_str = f"{mf_v:.6f}" if is_valid and np.isfinite(mf_v) else "NaN"
                safe_label = str(label).replace(",", " ")
                self._file.write(
                    f"{time_s:.3f},{c},{safe_label},{1 if is_valid else 0},"
                    f"{iemg_str},{mpf_str},{mf_str}\n"
                )
            self._file.flush()
            os.fsync(self._file.fileno())
        except OSError:
            pass

    def close(self, *, remove: bool = False) -> None:
        """Close the CSV file and optionally remove it if the recording was aborted."""
        f, self._file = self._file, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        if remove:
            try:
                self.path.unlink()
            except OSError:
                pass


def read_metrics_csv(
    edf_path: str | Path, n_channels: int | None = None
) -> dict[str, Any] | None:
    """Load persisted 1-second metric series from ``<edf_path>.metrics.csv``."""
    path = metrics_csv_path(edf_path)
    if not path.is_file():
        return None
    meta: dict[str, str] = {}
    by_channel: dict[int, dict[str, list[ Any]]] = {}
    try:
        with open(path, encoding="utf-8", newline="") as f:
            data_lines: list[str] = []
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    for token in stripped[1:].split():
                        if "=" in token:
                            k, v = token.split("=", 1)
                            meta[k] = v
                else:
                    data_lines.append(line)
            if not data_lines:
                return None
            reader = csv.DictReader(data_lines)
            for row in reader:
                c = int(row["channel"])
                bucket = by_channel.setdefault(
                    c,
                    {
                        "time_s": [],
                        "valid": [],
                        "iemg_uv_s": [],
                        "mpf_hz": [],
                        "mf_hz": [],
                    },
                )
                bucket["time_s"].append(float(row["time_s"]))
                is_valid = row.get("valid", "0").strip() in ("1", "True", "true")
                bucket["valid"].append(is_valid)
                if is_valid:
                    bucket["iemg_uv_s"].append(float(row["iemg_uv_s"]))
                    bucket["mpf_hz"].append(float(row["mpf_hz"]))
                    bucket["mf_hz"].append(float(row["mf_hz"]))
                else:
                    bucket["iemg_uv_s"].append(float("nan"))
                    bucket["mpf_hz"].append(float("nan"))
                    bucket["mf_hz"].append(float("nan"))
    except (OSError, KeyError, ValueError):
        return None

    if not by_channel:
        return None

    total_ch = (
        int(n_channels)
        if n_channels is not None
        else (max(by_channel.keys()) + 1)
    )
    time_list: list[FloatArray] = []
    valid_list: list[npt.NDArray[np.bool_]] = []
    iemg_list: list[FloatArray] = []
    mpf_list: list[FloatArray] = []
    mf_list: list[FloatArray] = []
    for c in range(total_ch):
        b = by_channel.get(c)
        if b is None:
            time_list.append(np.empty(0, dtype=np.float64))
            valid_list.append(np.empty(0, dtype=np.bool_))
            iemg_list.append(np.empty(0, dtype=np.float64))
            mpf_list.append(np.empty(0, dtype=np.float64))
            mf_list.append(np.empty(0, dtype=np.float64))
        else:
            time_list.append(np.asarray(b["time_s"], dtype=np.float64))
            valid_list.append(np.asarray(b["valid"], dtype=np.bool_))
            iemg_list.append(np.asarray(b["iemg_uv_s"], dtype=np.float64))
            mpf_list.append(np.asarray(b["mpf_hz"], dtype=np.float64))
            mf_list.append(np.asarray(b["mf_hz"], dtype=np.float64))

    return {
        "n_channels": total_ch,
        "meta": meta,
        "metadata": meta,
        "time_s": time_list,
        "valid": valid_list,
        "iemg_uv_s": iemg_list,
        "mpf_hz": mpf_list,
        "mf_hz": mf_list,
        "iemg": iemg_list,
        "mpf": mpf_list,
        "mf": mf_list,
    }


def compute_recording_metrics_1s(
    raw_channels: Sequence[FloatArray | np.ndarray] | np.ndarray,
    fs: float,
    *,
    f_low: float = 20.0,
    f_high: float = 450.0,
    f_notch: float = 50.0,
    f_env: float = 5.0,
    notch_enabled: bool = True,
    notch_harmonics: bool = True,
) -> dict[str, Any]:
    """Compute 1-second non-overlapping metric series for post-recording review
    when no companion ``.metrics.csv`` file is present."""
    if isinstance(raw_channels, np.ndarray):
        if raw_channels.ndim == 1:
            channels_seq: Sequence[FloatArray | np.ndarray] = [raw_channels]
        elif raw_channels.ndim == 2:
            channels_seq = [
                raw_channels[:, c] for c in range(raw_channels.shape[1])
            ]
        else:
            channels_seq = []
    else:
        channels_seq = raw_channels
    n_ch = len(channels_seq)
    if n_ch == 0:
        return {
            "time_s": [],
            "valid": [],
            "iemg_uv_s": [],
            "mpf_hz": [],
            "mf_hz": [],
            "iemg": [],
            "mpf": [],
            "mf": [],
        }
    fs_int = round(float(fs))
    nyq = fs_int / 2.0
    eff_high = f_high if f_high < nyq else min(450.0, nyq * 0.9)
    eff_low = f_low if 0.0 < f_low < eff_high else min(20.0, eff_high * 0.5)
    fstates = [
        RealtimeFilterState(
            fs=fs_int,
            f_low=eff_low,
            f_high=eff_high,
            f_notch=f_notch,
            f_env=f_env,
            notch_enabled=notch_enabled,
            notch_harmonics=notch_harmonics,
        )
        for _ in range(n_ch)
    ]
    filt_channels = [
        fstates[c].process_block(np.asarray(channels_seq[c], dtype=np.float64))[0]
        for c in range(n_ch)
    ]
    acc = RealtimeMetricAccumulator(
        fs=fs_int,
        n_channels=n_ch,
        f_low=eff_low,
        f_high=eff_high,
    )
    snapshots = acc.push_block(filt_channels)
    time_list = [
        np.asarray([s["time_s"] for s in snapshots], dtype=np.float64)
        for _ in range(n_ch)
    ]
    valid_list = [
        np.asarray([s["valid"][c] for s in snapshots], dtype=np.bool_)
        for c in range(n_ch)
    ]
    iemg_list = [
        np.asarray([s["iemg_uv_s"][c] for s in snapshots], dtype=np.float64)
        for c in range(n_ch)
    ]
    mpf_list = [
        np.asarray([s["mpf_hz"][c] for s in snapshots], dtype=np.float64)
        for c in range(n_ch)
    ]
    mf_list = [
        np.asarray([s["mf_hz"][c] for s in snapshots], dtype=np.float64)
        for c in range(n_ch)
    ]
    return {
        "time_s": time_list,
        "valid": valid_list,
        "iemg_uv_s": iemg_list,
        "mpf_hz": mpf_list,
        "mf_hz": mf_list,
        "iemg": iemg_list,
        "mpf": mpf_list,
        "mf": mf_list,
    }


def compute_segments(
    emg_filtered: FloatArray | np.ndarray,
    fs: float,
    seg_len_s: float = 1.0,
    overlap: float = 0.5,
    f_low: float = 20.0,
    f_high: float = 450.0,
) -> dict[str, FloatArray]:
    """Sliding-window RMS and MDF, suitable for fatigue trend analysis.

    Each segment's MDF is :func:`compute_psd_mnf_mdf`'s, over the same
    ``f_low``-``f_high`` band as the summary's and each contraction's, from
    one Welch window the length of the segment. It used to be the median of
    the whole spectrum, 0 Hz to half the sampling rate, taken as the first
    bin at which a running sum of the bins reached half the power, so the
    three MDFs the application reports were three different measures.

    Parameters
    ----------
    emg_filtered : array-like
        Filtered EMG signal.
    fs : float
        Sampling frequency (Hz).
    seg_len_s : float, optional
        Segment length in seconds (default 1.0).
    overlap : float, optional
        Fraction of overlap between consecutive segments (default 0.5).
    f_low, f_high : float, optional
        Band over which each segment's MDF is computed (default 20-450 Hz).

    Returns
    -------
    dict
        ``t_seg`` (segment start times, seconds), ``rms_seg`` (mV),
        ``mdf_seg`` (Hz).
    """
    emg = np.asarray(emg_filtered, dtype=np.float64)
    points = int(seg_len_s * fs)
    step = max(1, int(points * (1.0 - overlap)))

    rms_seg: list[float] = []
    mdf_seg: list[float] = []

    for start in range(0, len(emg) - points + 1, step):
        segment = emg[start : start + points]
        rms_seg.append(float(np.sqrt(np.mean(segment**2))))

        mdf_seg.append(float(compute_psd_mnf_mdf(
            segment, fs, f_low=f_low, f_high=f_high, nperseg=points,
        )["mdf"]))

    t_seg = np.arange(len(rms_seg)) * (step / fs)

    return {
        "t_seg": t_seg,
        "rms_seg": np.asarray(rms_seg, dtype=np.float64),
        "mdf_seg": np.asarray(mdf_seg, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Acquisition quality diagnostics
# ---------------------------------------------------------------------------


def detect_acquisition_problems(
    emg_raw: FloatArray | np.ndarray, fs: float
) -> dict[str, Any]:
    """Flag two common acquisition failure modes used in the teaching lab.

    Detected problems:

    1. **Saturation** -- the signal hits a flat plateau at the ADC's
       extremes for at least 10 ms (see De Luca 1997). Usually means
       the gain is too high or the electrode lost contact momentarily.
    2. **Flat baseline** -- the standard deviation of the first 2 s of
       the recording is less than 1% of the global standard deviation.
       A genuine surface EMG always shows +/-5-20 microvolt baseline
       noise even at rest, so a perfectly flat baseline strongly
       suggests a disconnected electrode or a misconfigured gain.

    Parameters
    ----------
    emg_raw : array-like
        Raw EMG signal in any units (mV or ADC counts both work).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict
        ``saturation_pct`` (float, %), ``flat_baseline`` (bool),
        ``warnings`` (list of human-readable warning strings).
    """
    emg = np.asarray(emg_raw, dtype=np.float64)
    warnings: list[str] = []

    # 1) Saturation: contiguous runs of >= 10 ms at ADC extremes
    v_max = float(np.max(emg))
    v_min = float(np.min(emg))
    threshold_max = v_max - 0.01 * (v_max - v_min)
    threshold_min = v_min + 0.01 * (v_max - v_min)
    extreme = (emg >= threshold_max) | (emg <= threshold_min)

    min_run = max(1, int(0.010 * fs))
    saturation_count = 0
    run = 0
    for at_extreme in extreme:
        if at_extreme:
            run += 1
        else:
            if run >= min_run:
                saturation_count += run
            run = 0
    if run >= min_run:
        saturation_count += run

    saturation_pct = 100.0 * saturation_count / len(emg) if len(emg) > 0 else 0.0

    if saturation_pct > 1.0:
        warnings.append(
            tr(
                "Possible saturation: {pct:.1f}% of samples sit at ADC extremes "
                "for runs ≥ 10 ms. Check electrode contact and gain."
            ).format(pct=saturation_pct)
        )

    # 2) Flat baseline in the first 2 s (before any contraction)
    n_initial = min(int(2.0 * fs), len(emg))
    std_initial = float(np.std(emg[:n_initial])) if n_initial > 0 else 0.0
    std_global = float(np.std(emg))

    flat_baseline = std_global > 0 and std_initial < 0.01 * std_global
    if flat_baseline:
        warnings.append(
            tr(
                "Suspiciously flat baseline at the start of the recording. "
                "May indicate a disconnected electrode or misconfigured gain."
            )
        )

    return {
        "saturation_pct": saturation_pct,
        "flat_baseline": flat_baseline,
        "warnings": warnings,
    }


def mmg_envelope(
    acc: FloatArray | np.ndarray,
    fs: float,
    f_low: float = 5.0,
    f_high: float = 50.0,
    f_env: float = 5.0,
) -> FloatArray:
    """Mechanomyography (MMG) envelope from an accelerometer on the muscle belly.

    Band-passes the acceleration to the muscle-vibration band (default
    5-50 Hz, which removes the DC/gravity term and gross-movement drift below
    ~5 Hz), rectifies it and low-passes to an envelope — the mechanical
    counterpart of the EMG envelope. Zero-phase (``sosfiltfilt``), matching the
    offline EMG path so the two envelopes are directly comparable in time.

    Parameters
    ----------
    acc : array-like
        Accelerometer signal (any units; the mean is removed).
    fs : float
        Sampling frequency (Hz).
    f_low, f_high : float, optional
        Muscle-vibration band-pass cut-offs (Hz). Defaults 5 and 50.
    f_env : float, optional
        Envelope low-pass cut-off (Hz). Default 5.
    """
    a = np.asarray(acc, dtype=np.float64)
    a = a - float(np.mean(a)) if a.size else a
    nyq = fs / 2.0
    fh = min(float(f_high), 0.9 * nyq)
    filtered = sosfiltfilt(design_bandpass(f_low, fh, fs), a)
    rectified = np.abs(filtered)
    return sosfiltfilt(design_lowpass(f_env, fs), rectified)


def movement_envelope(
    acc: FloatArray | np.ndarray,
    fs: float,
    hp_cutoff: float = 0.5,
    f_env: float = 3.0,
) -> FloatArray:
    """Kinematic movement envelope from an accelerometer on a moving segment.

    For a limb-mounted accelerometer the teaching signal is gross movement, not
    muscle vibration: high-pass the acceleration to drop gravity and slow drift,
    integrate it once to a velocity (arbitrary units — the accelerometer is
    uncalibrated), high-pass the velocity again to remove the residual
    integration drift, then rectify and low-pass to a smooth envelope. Overlaid
    on the EMG envelope it shows that the movement follows the contraction.
    Zero-phase (``sosfiltfilt``), matching the EMG/MMG envelope paths so the
    traces line up in time.

    Parameters
    ----------
    acc : array-like
        Accelerometer signal (any units; the mean is removed).
    fs : float
        Sampling frequency (Hz).
    hp_cutoff : float, optional
        High-pass cut-off (Hz) applied to both the acceleration and the
        integrated velocity to drop gravity and integration drift. Default 0.5.
    f_env : float, optional
        Envelope low-pass cut-off (Hz). Default 3.
    """
    a = np.asarray(acc, dtype=np.float64)
    if a.size == 0:
        return a
    nyq = fs / 2.0
    hp = min(float(hp_cutoff), 0.9 * nyq)
    sos_hp = iirfilter(2, hp, btype="high", fs=fs, ftype="butter", output="sos")
    acc_hp = sosfiltfilt(sos_hp, a - float(np.mean(a)))
    vel = cumulative_trapezoid(acc_hp, dx=1.0 / fs, initial=0.0)
    vel = sosfiltfilt(sos_hp, vel)  # remove the residual integration drift
    rectified = np.abs(vel)
    fe = min(float(f_env), 0.9 * nyq)
    return sosfiltfilt(design_lowpass(fe, fs), rectified)


def tremor_spectrum(
    acc: FloatArray | np.ndarray,
    fs: float,
    f_min: float = 2.0,
    f_max: float = 20.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Welch PSD of an accelerometer axis and its tremor peak frequency.

    Returns ``(frequencies, psd, peak_hz)`` where ``peak_hz`` is the frequency
    of maximum power inside ``[f_min, f_max]`` — the tremor band (physiological
    tremor sits around 8-12 Hz). ``peak_hz`` is ``0.0`` when the band is empty.

    Parameters
    ----------
    acc : array-like
        Accelerometer signal (the mean is removed before the transform).
    fs : float
        Sampling frequency (Hz).
    f_min, f_max : float, optional
        Tremor search band (Hz). Defaults 2 and 20.
    """
    a = np.asarray(acc, dtype=np.float64)
    a = a - float(np.mean(a)) if a.size else a
    nper = min(len(a), int(4 * fs)) if a.size else 64
    freqs, psd = welch(a, fs=fs, nperseg=max(64, nper))
    band = (freqs >= f_min) & (freqs <= f_max)
    peak = float(freqs[band][int(np.argmax(psd[band]))]) if np.any(band) else 0.0
    return freqs, psd, peak


# Full-scale amplitude of a BITalino EMG channel in mV at the electrodes:
# (VCC / 2) * 1000 / G, with VCC = 3.3 V and a front-end gain G = 1009 (see
# emgteach.devices.bitalino). Used only as a fallback when a recording does
# not declare its own physical range.
DEFAULT_PHYSICAL_MAX_MV = 1.635


def assess_channel_quality(
    signal: FloatArray | np.ndarray,
    fs: float,
    physical_max: float = DEFAULT_PHYSICAL_MAX_MV,
) -> str:
    """One-word quality verdict for a whole EMG channel, for a load-time check.

    Returns one of:

    * ``"flat"`` — essentially no signal (a disconnected electrode or a
      channel that was declared but never wired): very low amplitude over the
      whole recording.
    * ``"saturated"`` — the electrode lost contact / the gain was too high, so
      the trace spends a large fraction of the time pinned at the ADC rails.
    * ``"weak"`` — there is a signal but its amplitude is low; usable but worth
      a heads-up.
    * ``"ok"`` — a normal surface-EMG channel.

    Complements :func:`detect_acquisition_problems` (whose saturation run
    detector it reuses) with a *whole-channel* flat/weak amplitude check, so
    the GUI can warn per channel the moment a file is opened.
    """
    s = np.asarray(signal, dtype=np.float64)
    if s.size == 0:
        return "flat"
    rms = float(np.std(s))
    p2p = float(np.ptp(s))
    # Saturation: a big share of samples sit near the ±full-scale rails.
    rail = float(np.mean(np.abs(s) >= 0.94 * abs(physical_max))) if physical_max else 0.0
    if rail > 0.05:
        return "saturated"
    # Flat: no real activity anywhere in the recording.
    if rms < 0.012 and p2p < 0.10:
        return "flat"
    if rms < 0.03:
        return "weak"
    return "ok"


@dataclass(frozen=True)
class QualityStatus:
    """Outcome of one :meth:`LiveQualityMonitor.update` call.

    Attributes
    ----------
    code : str
        ``"ok"``, ``"saturation"`` or ``"flat"``.
    message : str
        Short, already-translated text for a status label.
    saturation_frac : float
        Fraction (0-1) of the block sitting at the hardware rails.
    """

    code: str
    message: str
    saturation_frac: float = 0.0


class LiveQualityMonitor:
    """Per-block signal-quality check for the live acquisition preview.

    Unlike :func:`detect_acquisition_problems`, which inspects a whole saved
    recording against its own data range, this monitor judges each incoming
    block against the **device's true physical rails** (``physical_min`` /
    ``physical_max``), so it can warn *while recording* that:

    * the amplifier is **saturating** (a fraction of the block greater than
      ``sat_frac`` sits within 1% of either rail — gain too high or a railing
      electrode), or
    * the electrode looks **disconnected** (the block's standard deviation is
      below ``flat_std``, i.e. a suspiciously flat line — a real biopotential
      always carries some baseline noise).

    Otherwise the signal is reported as OK. The check is stateless per block,
    so it tracks the signal as it streams without accumulating history.

    Parameters
    ----------
    rail_min, rail_max : float
        Hardware full-scale in physical units (mV), e.g. the device's
        ``physical_min`` / ``physical_max``.
    sat_frac : float, optional
        Fraction of a block at the rails above which saturation is flagged
        (default 0.02, i.e. 2%).
    flat_std : float, optional
        Standard-deviation threshold below which the block is deemed flat.
        Defaults to 0.05% of the full-scale span.
    """

    def __init__(
        self,
        rail_min: float,
        rail_max: float,
        sat_frac: float = 0.02,
        flat_std: float | None = None,
    ) -> None:
        self.rail_min = float(rail_min)
        self.rail_max = float(rail_max)
        span = self.rail_max - self.rail_min
        if span <= 0:
            raise ValueError("rail_max must be greater than rail_min.")
        self._hi = self.rail_max - 0.01 * span
        self._lo = self.rail_min + 0.01 * span
        self._sat_frac = float(sat_frac)
        self._flat_std = 5e-4 * span if flat_std is None else float(flat_std)

    def update(self, block: FloatArray | np.ndarray) -> QualityStatus:
        """Assess one raw block and return its :class:`QualityStatus`."""
        x = np.asarray(block, dtype=np.float64).ravel()
        if x.size == 0:
            return QualityStatus("ok", tr("Signal OK"))

        sat_frac = float(np.mean((x >= self._hi) | (x <= self._lo)))
        if sat_frac > self._sat_frac:
            return QualityStatus(
                "saturation",
                tr("Saturation: {pct:.0f}% at rails — lower gain").format(
                    pct=sat_frac * 100.0
                ),
                sat_frac,
            )
        if float(np.std(x)) < self._flat_std:
            return QualityStatus(
                "flat",
                tr("Flat signal — check electrode contact"),
                sat_frac,
            )
        return QualityStatus("ok", tr("Signal OK"), sat_frac)


# ---------------------------------------------------------------------------
# Contraction-onset detection
# ---------------------------------------------------------------------------


class OnsetDetector:
    """Stateful, real-time detector of contraction onsets on the envelope.

    The detector implements the classical single-threshold surface-EMG
    onset rule: it estimates a resting baseline from the first
    ``baseline_s`` seconds of signal, sets the threshold to
    ``mean + k * std`` of that baseline, and flags an **onset** on each
    rising crossing of the threshold. A refractory period suppresses
    repeated detections within the same contraction.

    It is fed the same envelope blocks the acquisition worker produces,
    so detection runs live; :meth:`process` returns the onset times found
    in each block. The sample counter is global, so onset times are
    expressed in seconds from the start of the stream and line up with
    the EDF time base.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    k : float, optional
        Threshold sensitivity in baseline standard deviations (default 3).
        Lower ``k`` detects weaker contractions.
    baseline_s : float, optional
        Length of the initial resting window used to estimate the
        baseline mean and standard deviation (default 1.0 s). The subject
        must be at rest during this window.
    refractory_s : float, optional
        Minimum time between consecutive onsets (default 0.5 s).
    min_duration_s : float, optional
        The envelope must stay above the threshold for at least this long
        before an onset is declared, which debounces brief noise spikes
        (default 0.05 s).
    """

    def __init__(
        self,
        fs: float,
        k: float = 3.0,
        baseline_s: float = 1.0,
        refractory_s: float = 0.5,
        min_duration_s: float = 0.05,
    ) -> None:
        self.fs = float(fs)
        self.k = float(k)
        self._baseline_n = max(1, int(baseline_s * fs))
        self._refractory_n = max(1, int(refractory_s * fs))
        self._min_duration_n = max(1, int(min_duration_s * fs))
        self._n = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0
        self._threshold: float | None = None
        self._above_run = 0
        self._last_onset_n = -(10**12)

    @property
    def threshold(self) -> float | None:
        """The calibrated threshold, or ``None`` while still in baseline."""
        return self._threshold

    def rearm(self) -> None:
        """Measure the resting level again, from the next samples on.

        The threshold is set once, from the first ``baseline_s`` of the
        stream, and in a guided session that second is the start of the
        warm-up — contractions, not rest. Re-armed at the start of the
        recording phase, after a countdown that is rest by construction,
        the threshold describes the rest the task rises from. The sample
        counter is kept, so onset times stay in the file's time base.
        """
        self._threshold = None
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0
        self._above_run = 0

    def process(self, envelope_block: FloatArray | np.ndarray) -> list[float]:
        """Feed one envelope block; return onset times (s) found in it.

        Parameters
        ----------
        envelope_block : array-like
            A block of envelope samples, contiguous with previous blocks.

        Returns
        -------
        list of float
            Onset times in seconds from the start of the stream.
        """
        block = np.asarray(envelope_block, dtype=np.float64).ravel()
        onsets: list[float] = []
        for x in block:
            if self._threshold is None:
                # Baseline calibration phase.
                self._sum += x
                self._sumsq += x * x
                self._count += 1
                if self._count >= self._baseline_n:
                    mean = self._sum / self._count
                    var = max(self._sumsq / self._count - mean * mean, 0.0)
                    self._threshold = mean + self.k * (var**0.5)
            elif x > self._threshold:
                self._above_run += 1
                # Declare an onset once the signal has stayed above the
                # threshold for the minimum duration (debounces noise spikes),
                # timed at the crossing that started the run.
                if self._above_run == self._min_duration_n:
                    onset_n = self._n - (self._min_duration_n - 1)
                    if (onset_n - self._last_onset_n) >= self._refractory_n:
                        onsets.append(onset_n / self.fs)
                        self._last_onset_n = onset_n
            else:
                self._above_run = 0
            self._n += 1
        return onsets


def detect_onsets(
    envelope: FloatArray | np.ndarray,
    fs: float,
    k: float = 3.0,
    baseline_s: float = 1.0,
    refractory_s: float = 0.5,
    min_duration_s: float = 0.05,
) -> list[float]:
    """Detect contraction onsets in a complete envelope (offline helper).

    Convenience wrapper that runs :class:`OnsetDetector` over the whole
    signal in one call. Because the detector is stateful, feeding the
    signal in blocks via :meth:`OnsetDetector.process` yields identical
    results.

    Parameters
    ----------
    envelope : array-like
        Envelope (rectified, low-pass filtered EMG).
    fs : float
        Sampling frequency (Hz).
    k, baseline_s, refractory_s, min_duration_s : float, optional
        Forwarded to :class:`OnsetDetector`.

    Returns
    -------
    list of float
        Onset times in seconds.
    """
    detector = OnsetDetector(
        fs,
        k=k,
        baseline_s=baseline_s,
        refractory_s=refractory_s,
        min_duration_s=min_duration_s,
    )
    return detector.process(envelope)
