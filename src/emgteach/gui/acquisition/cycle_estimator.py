"""Conservative sine-frequency estimate for the live plot's Auto window.

This estimates a display scale, not an EMG metric or a calibrated frequency.
Broadband and ambiguous input intentionally has no frequency estimate.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def estimate_sine_frequency(samples: npt.ArrayLike, fs: float) -> float | None:
    """Return the dominant sine frequency, or ``None`` without enough evidence.

    Inspect at most the latest two seconds, after at least half a second is
    available. A Hann-windowed FFT peak and its two neighbours must contain
    at least 60% of non-DC power, and the observation must span three cycles.
    Invalid or non-finite inputs, flat traces and numerical rounding noise
    have no estimate. A log-power parabolic interpolation refines the peak.
    """
    if isinstance(fs, bool):
        return None
    try:
        rate = float(fs)
        values = np.asarray(samples, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(rate) or rate <= 0 or values.ndim != 1:
        return None
    if len(values) < max(4, int(np.ceil(0.5 * rate))):
        return None
    max_samples = int(2 * rate)
    if max_samples < 4:
        return None
    values = values[-max_samples:]
    if not np.isfinite(values).all():
        return None

    # Normalizing before the FFT avoids overflow and underflow without an
    # arbitrary voltage threshold. Reject variation at the input's roundoff.
    input_scale = float(np.max(np.abs(values)))
    if input_scale <= np.finfo(np.float64).tiny:
        return None
    scaled = values / input_scale
    centered = scaled - np.mean(scaled)
    amplitude = float(np.max(np.abs(centered)))
    if amplitude <= 8 * np.finfo(np.float64).eps:
        return None
    centered /= amplitude
    spectrum = np.fft.rfft(centered * np.hanning(len(centered)))
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= 0:
        return None
    peak = int(np.argmax(power))
    if peak == 0:
        return None
    neighbourhood = power[max(1, peak - 1):peak + 2]
    if float(np.sum(neighbourhood)) < 0.6 * total:
        return None

    offset = 0.0
    if 1 < peak < len(power) - 1 and np.all(power[peak - 1:peak + 2] > 0):
        left, middle, right = np.log(power[peak - 1:peak + 2])
        denominator = left - 2 * middle + right
        if denominator < 0:
            offset = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    cycles = peak + offset
    if cycles < 3.0:
        return None
    frequency = cycles * rate / len(centered)
    return float(frequency) if np.isfinite(frequency) and frequency > 0 else None
