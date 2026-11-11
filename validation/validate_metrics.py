"""Numerical validation of emgteach's core sEMG metrics.

Each metric produced by the analytic core is checked against (a) an analytic
ground truth (signals whose theoretical MNF/MDF, fatigue slope and amplitude
percentiles are known in closed form) and (b) an independent reference
implementation built directly on scipy/numpy, so the check does not reuse
emgteach's own integration choices.

Run:  python validation/validate_metrics.py
Exit code 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

SEED = 12345
FS = 1000.0
F_LO, F_HI = 20.0, 450.0

# --- import the Qt-free analytic core without running the package __init__
# (which pulls in the PySide6 workers/devices layers). We register a stub
# package so the submodule imports resolve, then import only the core modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if os.path.isdir(os.path.join(_SRC, "emgteach")):
    sys.path.insert(0, _SRC)
    _pkg = types.ModuleType("emgteach")
    _pkg.__path__ = [os.path.join(_SRC, "emgteach")]
    sys.modules.setdefault("emgteach", _pkg)
from emgteach.apda import compute_apdf  # noqa: E402
from emgteach.dsp import (  # noqa: E402
    RealtimeFilterState,
    RealtimeMetricAccumulator,
    compute_iemg,
    compute_psd_mnf_mdf,
    compute_realtime_window_metrics,
    compute_segments,
    read_metrics_csv,
)
from emgteach.fatigue import fit_mdf_vs_time  # noqa: E402
from emgteach.mvc import compute_mvc, normalise_to_mvc  # noqa: E402


def shaped_noise(shape_fn, n, fs, seed):
    """White Gaussian noise shaped in frequency so its PSD follows shape_fn(f)."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    gain = np.sqrt(np.clip(shape_fn(f), 0.0, None))
    return np.fft.irfft(spectrum * gain, n=n)


def theoretical_mnf_mdf(shape_fn, f_lo=F_LO, f_hi=F_HI):
    """Exact MNF/MDF of a target PSD shape over [f_lo, f_hi] by dense integration."""
    f = np.linspace(f_lo, f_hi, 200001)
    s = np.clip(shape_fn(f), 0.0, None)
    total = trapezoid(s, f)
    mnf = trapezoid(f * s, f) / total
    cum = np.cumsum((s[:-1] + s[1:]) / 2 * np.diff(f))
    mdf = float(f[1:][np.searchsorted(cum, total / 2.0)])
    return float(mnf), mdf


def ref_mnf_mdf(sig, fs=FS, f_lo=F_LO, f_hi=F_HI):
    """Independent reference: Welch + trapezoid MNF and interpolated MDF."""
    f, p = welch(sig, fs=fs, nperseg=int(fs), noverlap=int(fs) // 2)
    mask = (f >= f_lo) & (f <= f_hi)
    f, p = f[mask], p[mask]
    total = trapezoid(p, f)
    mnf = trapezoid(f * p, f) / total
    cum = np.concatenate([[0.0], np.cumsum((p[:-1] + p[1:]) / 2 * np.diff(f))])
    mdf = float(np.interp(total / 2.0, cum, f))
    return float(mnf), mdf


_rows = []


def check(metric, case, got, ref, tol, unit="", rel=False):
    err = abs(got - ref)
    ok = (err / abs(ref) <= tol) if rel else (err <= tol)
    _rows.append((metric, case, got, ref, err, tol, "rel" if rel else "abs", unit, ok))


def run():
    # 1) MNF / MDF vs analytic ground truth and independent reference
    shapes = {
        "flat band [30,440]": lambda f: ((f >= 30) & (f <= 440)).astype(float),
        "rising PSD ~f": lambda f: np.where((f >= 30) & (f <= 440), f, 0.0),
        "falling PSD ~1/f": lambda f: np.where(
            (f >= 30) & (f <= 440), 1.0 / np.maximum(f, 1.0), 0.0
        ),
    }
    n = int(180 * FS)
    for name, shp in shapes.items():
        sig = shaped_noise(shp, n, FS, SEED)
        out = compute_psd_mnf_mdf(sig, FS, F_LO, F_HI)
        mnf_th, mdf_th = theoretical_mnf_mdf(shp)
        mnf_ref, mdf_ref = ref_mnf_mdf(sig)
        check("MNF", f"{name} vs analytic", out["mnf"], mnf_th, 0.02, "Hz", rel=True)
        check("MDF", f"{name} vs analytic", out["mdf"], mdf_th, 0.02, "Hz", rel=True)
        check("MNF", f"{name} vs indep ref", out["mnf"], mnf_ref, 0.01, "Hz", rel=True)
        check("MDF", f"{name} vs indep ref", out["mdf"], mdf_ref, 0.02, "Hz", rel=True)

    for f0 in (80.0, 250.0):
        t = np.arange(int(60 * FS)) / FS
        out = compute_psd_mnf_mdf(np.sin(2 * np.pi * f0 * t), FS, F_LO, F_HI)
        check("MNF", f"pure tone {f0:.0f} Hz", out["mnf"], f0, 2.0, "Hz")
        check("MDF", f"pure tone {f0:.0f} Hz", out["mdf"], f0, 2.0, "Hz")

    # 2) Fatigue slope recovery (end-to-end): impose a linear MDF ramp
    dur, f_start, f_end = 60.0, 150.0, 90.0
    t = np.arange(int(dur * FS)) / FS
    inst_f = f_start + (f_end - f_start) * (t / dur)
    rng = np.random.default_rng(SEED)
    sig = np.sin(2 * np.pi * np.cumsum(inst_f) / FS) + 0.02 * rng.standard_normal(t.size)
    seg = compute_segments(sig, FS, seg_len_s=1.0, overlap=0.5)
    fit = fit_mdf_vs_time(seg["t_seg"], seg["mdf_seg"], degree=2)
    check("Fatigue slope", "linear MDF ramp", fit["slope"],
          (f_end - f_start) / dur, 0.10, "Hz/s", rel=True)

    # 3) MVC + APDF vs a known amplitude distribution (uniform on [0, A])
    amp = 40.0
    env = np.linspace(0.0, amp, 500001)
    mvc = compute_mvc(env, percentile=95.0)
    check("MVC (P95)", "uniform[0,A]", mvc, 0.95 * amp, 1e-3, "mV", rel=True)
    ap = compute_apdf(normalise_to_mvc(env, mvc))
    got = {"static P10": ap.static.value, "median P50": ap.median.value,
           "peak P90": ap.peak.value}
    for lvl, k in (("static P10", 10), ("median P50", 50), ("peak P90", 90)):
        check(f"APDF {lvl}", "uniform[0,A]", got[lvl], k / 100.0 * amp / mvc * 100.0,
              0.01, "%MVC", rel=True)

    # 4) LHE7904 4000 SPS real-time 1 s metrics (iEMG, MPF, MF) & 60 s stability
    fs_lhe = 4000.0
    t_1s = np.arange(int(fs_lhe), dtype=np.float64) / fs_lhe
    # 4a) User bench signal: 100 Hz, 1.0 mV, 0 offset
    bench_100 = 1.0 * np.sin(2.0 * np.pi * 100.0 * t_1s)
    m_100 = compute_realtime_window_metrics(bench_100, fs_lhe, f_low=10.0, f_high=500.0)
    ref_iemg_1mv = (2.0 / np.pi) * 1000.0
    check("iEMG (4kHz)", "100 Hz 1 mV bench sine", m_100["iemg_uv_s"], ref_iemg_1mv, 0.005, "uV*s", rel=True)
    check("MPF (4kHz)", "100 Hz 1 mV bench sine", m_100["mpf_hz"], 100.0, 0.5, "Hz")
    check("MF (4kHz)", "100 Hz 1 mV bench sine", m_100["mf_hz"], 100.0, 0.5, "Hz")

    # 4b) 200 Hz, 2.0 mV sine
    sig_200 = 2.0 * np.sin(2.0 * np.pi * 200.0 * t_1s)
    m_200 = compute_realtime_window_metrics(sig_200, fs_lhe, f_low=10.0, f_high=500.0)
    check("iEMG (4kHz)", "200 Hz 2 mV sine", m_200["iemg_uv_s"], 2.0 * ref_iemg_1mv, 0.01, "uV*s", rel=True)
    check("MPF (4kHz)", "200 Hz 2 mV sine", m_200["mpf_hz"], 200.0, 0.5, "Hz")
    check("MF (4kHz)", "200 Hz 2 mV sine", m_200["mf_hz"], 200.0, 0.5, "Hz")

    # 4c) Dual-tone (100+200 Hz) MPF and Three-tone (100+150+200 Hz) MPF/MF
    sig_2t = np.sin(2.0 * np.pi * 100.0 * t_1s) + np.sin(2.0 * np.pi * 200.0 * t_1s)
    m_2t = compute_realtime_window_metrics(sig_2t, fs_lhe, f_low=10.0, f_high=500.0)
    check("MPF (4kHz)", "dual-tone 100+200 Hz", m_2t["mpf_hz"], 150.0, 1.0, "Hz")
    sig_3t = (
        np.sin(2.0 * np.pi * 100.0 * t_1s)
        + np.sin(2.0 * np.pi * 150.0 * t_1s)
        + np.sin(2.0 * np.pi * 200.0 * t_1s)
    )
    m_3t = compute_realtime_window_metrics(sig_3t, fs_lhe, f_low=10.0, f_high=500.0)
    check("MPF (4kHz)", "3-tone 100+150+200 Hz", m_3t["mpf_hz"], 150.0, 1.0, "Hz")
    check("MF (4kHz)", "3-tone 100+150+200 Hz", m_3t["mf_hz"], 150.0, 1.0, "Hz")

    # 4d) Single-50 Hz notch effect on 50 Hz + 100 Hz composite signal (2nd second steady state)
    t_2s = np.arange(int(2 * fs_lhe), dtype=np.float64) / fs_lhe
    comp_50_100 = np.sin(2.0 * np.pi * 50.0 * t_2s) + np.sin(2.0 * np.pi * 100.0 * t_2s)
    filt_lhe = RealtimeFilterState(
        fs_lhe, f_low=10.0, f_high=500.0, f_notch=50.0, notch_enabled=True, notch_harmonics=False
    )
    comp_filtered, _ = filt_lhe.process_block(comp_50_100)
    m_notch = compute_realtime_window_metrics(
        comp_filtered[int(fs_lhe) :], fs_lhe, f_low=10.0, f_high=500.0
    )
    check("MPF (4kHz)", "50+100 Hz after 50 Hz notch", m_notch["mpf_hz"], 100.0, 1.0, "Hz")
    check("MF (4kHz)", "50+100 Hz after 50 Hz notch", m_notch["mf_hz"], 100.0, 1.0, "Hz")

    # 4e) 60-second chunked streaming stability (600 x 400 samples at 4000 SPS)
    filt_60s = RealtimeFilterState(
        fs_lhe, f_low=10.0, f_high=500.0, f_notch=50.0, notch_enabled=True, notch_harmonics=False
    )
    acc_60s = RealtimeMetricAccumulator(fs=fs_lhe, n_channels=1, f_low=10.0, f_high=500.0)
    t_60s = np.arange(int(60 * fs_lhe), dtype=np.float64) / fs_lhe
    sig_60s = 1.0 * np.sin(2.0 * np.pi * 100.0 * t_60s)
    snaps_60s = []
    for k in range(600):
        blk = sig_60s[k * 400 : (k + 1) * 400]
        f_blk, _ = filt_60s.process_block(blk)
        snaps_60s.extend(acc_60s.push_block([f_blk]))
    check("60s stream", "window count (600x400)", float(len(snaps_60s)), 60.0, 0.0, "windows")
    check(
        "iEMG (60s)",
        "steady-state window #60",
        snaps_60s[-1]["iemg_uv_s"][0],
        compute_iemg(bench_100, fs_lhe),
        0.02,
        "uV*s",
        rel=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate emgteach sEMG metrics.")
    parser.add_argument(
        "--edf",
        type=str,
        default=None,
        help="Optional recorded EDF path to inspect alongside its .metrics.csv sidecar.",
    )
    parser.add_argument(
        "--expect-freq",
        type=float,
        default=100.0,
        help="Expected sine frequency (Hz) when --edf is provided (default: 100.0).",
    )
    parser.add_argument(
        "--expect-amp",
        type=float,
        default=1.0,
        help="Expected sine amplitude (mV) when --edf is provided (default: 1.0).",
    )
    args = parser.parse_args()

    run()
    if args.edf:
        csv_info = read_metrics_csv(args.edf)
        if csv_info is not None and len(csv_info["time_s"][0]) >= 2:
            valid_mask = csv_info["valid"][0]
            # Skip first 1s filter warm-up window if possible
            idx = -1
            if np.any(valid_mask):
                valid_indices = np.flatnonzero(valid_mask)
                idx = int(valid_indices[-1])
            ref_iemg = args.expect_amp * (2.0 / np.pi) * 1000.0
            check(
                "EDF .metrics",
                "recorded iEMG vs bench ref",
                float(csv_info["iemg_uv_s"][0][idx]),
                ref_iemg,
                0.05,
                "uV*s",
                rel=True,
            )
            check(
                "EDF .metrics",
                "recorded MPF vs bench ref",
                float(csv_info["mpf_hz"][0][idx]),
                args.expect_freq,
                2.0,
                "Hz",
            )
            check(
                "EDF .metrics",
                "recorded MF vs bench ref",
                float(csv_info["mf_hz"][0][idx]),
                args.expect_freq,
                2.0,
                "Hz",
            )

    print(f"emgteach metric validation  (seed={SEED}, fs={FS:.0f} Hz / LHE7904 4000 Hz)\n")
    hdr = (f"{'metric':<16}{'case':<32}{'emgteach':>11}{'reference':>11}"
           f"{'err':>9}{'tol':>7}   result")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    for metric, case, got, ref, err, tol, kind, _unit, ok in _rows:
        n_pass += ok
        tols = f"{tol * 100:.1f}%" if kind == "rel" else f"{tol:g}"
        print(f"{metric:<16}{case:<32}{got:>11.3f}{ref:>11.3f}"
              f"{err:>9.3f}{tols:>7}   {'PASS' if ok else 'FAIL'}")
    print("-" * len(hdr))
    print(f"{n_pass}/{len(_rows)} checks passed")
    return 0 if n_pass == len(_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
