"""LHE7904 selection and UART-to-EDF integration without hardware."""
from __future__ import annotations

import struct

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from emgteach.devices import BACKEND_LHE7904_UART, create_device
from emgteach.gui.tabs.acquisition import AcquisitionTab
from emgteach.gui.widgets.logger import LoggerWidget
from emgteach.io import read_edf_pyedflib
from emgteach.workers import AcquisitionWorker


@pytest.mark.gui
def test_lhe7904_selection_uses_device_rate_and_one_muscle(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    tab = AcquisitionTab(LoggerWidget(), settings)
    try:
        tab._chk_acc.setChecked(True)
        tab._combo_n_channels.setCurrentIndex(1)
        tab._conexion._combo_device_type.setCurrentIndex(2)
        assert tab._fs == 4000
        assert tab._trazado._fs == 4000
        assert tab._n_channels == 1
        assert not tab._combo_n_channels.isEnabled()
        assert not tab._chk_acc.isChecked()
        assert not tab._chk_acc.isEnabled()
        assert not tab._trazado._sidebar.isHidden()
        tab._conexion._combo_device_type.setCurrentIndex(0)
        assert tab._fs == 1000
        assert tab._trazado._fs == 1000
        assert tab._combo_n_channels.isEnabled()
    finally:
        tab.close()


@pytest.mark.gui
def test_uart_samples_round_trip_through_worker_to_4000hz_edf(qapp, tmp_path, monkeypatch):
    fs = 4000
    expected = np.sin(2 * np.pi * 80 * np.arange(fs + 2) / fs) * 0.5
    payload = bytearray(b"".join(
        struct.pack("<fff", float(value), 123.0, i / fs) + b"\x00\x00\x80\x7f"
        for i, value in enumerate(expected)
    ))

    class Serial:
        def __init__(self, **kwargs):
            self.is_open = True
            self.timeout = kwargs.get("timeout")

        def read(self, size):
            data = bytes(payload[:size])
            del payload[:size]
            return data

        def close(self):
            self.is_open = False

    monkeypatch.setattr("serial.Serial", Serial)
    device = create_device(BACKEND_LHE7904_UART, port="mock")
    worker = AcquisitionWorker(device=device, save_dir=str(tmp_path), n_per_read=400,
                               auto_detect=False)
    original_read = device.read
    count = 0

    def read_and_stop(n):
        nonlocal count
        block = original_read(n)
        count += len(block)
        if count >= fs:
            worker.stop()
        return block

    monkeypatch.setattr(device, "read", read_and_stop)
    errors = []
    paths = []
    worker.error.connect(errors.append)
    worker.finished_ok.connect(paths.append)
    worker.run()
    assert not errors
    assert paths and paths[0]
    result = read_edf_pyedflib(paths[0])
    assert result["sfreq"] == fs
    assert result["dimension"] == "mV"
    assert len(result["emg_raw"]) == fs
    # EDF's 16-bit quantization over +/-2400mV limits precision to ~0.074mV.
    np.testing.assert_allclose(result["emg_raw"], expected[:fs], atol=0.074)


@pytest.mark.gui
def test_100hz_1mv_bench_signal_saves_metrics_csv_and_renders_4plot_review(
    qapp, tmp_path, monkeypatch
):
    """Mandatory Review #2, #4, #5: 100 Hz 1 mV 0 offset bench signal through worker -> EDF + .metrics.csv + 4-plot review."""
    from emgteach.dsp import read_metrics_csv

    fs = 4000
    n_samples = 2 * fs  # 2 seconds
    t = np.arange(n_samples + 400, dtype=np.float64) / float(fs)
    expected = 1.0 * np.sin(2.0 * np.pi * 100.0 * t)
    payload = bytearray(
        b"".join(
            struct.pack("<fff", float(val), float(val), i / fs) + b"\x00\x00\x80\x7f"
            for i, val in enumerate(expected)
        )
    )

    class Serial:
        def __init__(self, **kwargs):
            self.is_open = True
            self.timeout = kwargs.get("timeout")

        def read(self, size):
            data = bytes(payload[:size])
            del payload[:size]
            return data

        def close(self):
            self.is_open = False

    monkeypatch.setattr("serial.Serial", Serial)
    device = create_device(BACKEND_LHE7904_UART, port="mock")
    worker = AcquisitionWorker(
        device=device,
        save_dir=str(tmp_path),
        n_per_read=400,
        f_low=10.0,
        f_high=500.0,
        f_notch=50.0,
        notch_enabled=True,
        notch_harmonics=False,
        auto_detect=False,
    )
    original_read = device.read
    count = 0

    def read_and_stop(n):
        nonlocal count
        block = original_read(n)
        count += len(block)
        if count >= n_samples:
            worker.stop()
        return block

    monkeypatch.setattr(device, "read", read_and_stop)
    emitted_metrics = []
    paths = []
    worker.data_ready.connect(
        lambda d: emitted_metrics.extend(d.get("metrics_1s_batch") or [])
    )
    worker.finished_ok.connect(paths.append)
    worker.run()

    assert len(paths) == 1
    edf_path = paths[0]
    assert len(emitted_metrics) == 2
    # Second 1-s window is in steady state for 100 Hz 1 mV sine wave
    assert emitted_metrics[1]["valid"] == [True]
    assert emitted_metrics[1]["iemg"][0] == pytest.approx(636.62, rel=0.03)
    assert emitted_metrics[1]["mpf"][0] == pytest.approx(100.0, abs=1.0)
    assert emitted_metrics[1]["mf"][0] == pytest.approx(100.0, abs=1.0)

    # Verify sidecar .metrics.csv
    csv_data = read_metrics_csv(edf_path)
    assert csv_data is not None
    assert len(csv_data["time_s"][0]) == 2
    assert csv_data["mpf"][0][1] == pytest.approx(100.0, abs=1.0)

    # Verify 4-plot post-recording review loads both EDF and .metrics.csv
    settings = QSettings(str(tmp_path / "review.ini"), QSettings.Format.IniFormat)
    tab = AcquisitionTab(LoggerWidget(), settings)
    try:
        tab._trazado.show_recording(edf_path, lambda s: s)
        x_iemg, y_iemg = tab._trazado._curves_iemg[0].getData()
        _x_mpf, y_mpf = tab._trazado._curves_mpf[0].getData()
        _x_mf, y_mf = tab._trazado._curves_mf[0].getData()
        assert len(x_iemg) == 2
        assert y_iemg[1] == pytest.approx(636.62, rel=0.03)
        assert y_mpf[1] == pytest.approx(100.0, abs=1.0)
        assert y_mf[1] == pytest.approx(100.0, abs=1.0)
    finally:
        tab.close()


@pytest.mark.gui
def test_worker_lost_frames_discards_partial_and_discontinuous_block(
    qapp, tmp_path, monkeypatch
):
    """Mandatory Review #4: When lost_frames increases during a read block, worker resets filter
    and discards both the partial window and the discontinuous block."""
    fs = 4000
    # 16 blocks of 400 samples:
    # Blocks 0..4 (2000 samples): clean
    # Block 5 (400 samples): lost_frames increases during device.read()
    # Blocks 6..15 (4000 samples): 10 clean blocks -> 1 valid 1s window after the gap
    total_samples = 17 * 400
    t = np.arange(total_samples, dtype=np.float64) / float(fs)
    sig = 1.0 * np.sin(2.0 * np.pi * 100.0 * t)
    payload = bytearray(
        b"".join(
            struct.pack("<fff", float(val), float(val), i / fs) + b"\x00\x00\x80\x7f"
            for i, val in enumerate(sig)
        )
    )

    class Serial:
        def __init__(self, **kwargs):
            self.is_open = True
            self.timeout = kwargs.get("timeout")

        def read(self, size):
            data = bytes(payload[:size])
            del payload[:size]
            return data

        def close(self):
            self.is_open = False

    monkeypatch.setattr("serial.Serial", Serial)
    device = create_device(BACKEND_LHE7904_UART, port="mock")
    device.lost_frames = 0
    worker = AcquisitionWorker(
        device=device,
        save_dir=str(tmp_path),
        n_per_read=400,
        f_low=10.0,
        f_high=500.0,
        f_notch=50.0,
        notch_enabled=True,
        notch_harmonics=False,
        auto_detect=False,
    )
    original_read = device.read
    blocks_read = 0

    def read_16_blocks(n):
        nonlocal blocks_read
        block = original_read(n)
        if blocks_read == 5:
            device.lost_frames += 12  # Simulate 12 frames dropped inside block 5
        blocks_read += 1
        if blocks_read >= 16:
            worker.stop()
        return block

    monkeypatch.setattr(device, "read", read_16_blocks)
    emitted_metrics = []
    worker.data_ready.connect(
        lambda d: emitted_metrics.extend(d.get("metrics_1s_batch") or [])
    )
    worker.run()

    # Exactly 2 metric emissions:
    # 1st at block 5 (t = 2400/4000 = 0.6 s): invalid (NaN) due to gap
    # 2nd at block 15 (t = 6400/4000 = 1.6 s): valid 1s window from blocks 6..15
    assert len(emitted_metrics) == 2
    assert emitted_metrics[0]["valid"] == [False]
    assert np.isnan(emitted_metrics[0]["iemg"][0])
    assert emitted_metrics[0]["time_s"] == pytest.approx(0.6)
    assert emitted_metrics[1]["valid"] == [True]
    assert emitted_metrics[1]["time_s"] == pytest.approx(1.6)
    assert emitted_metrics[1]["mpf"][0] == pytest.approx(100.0, abs=1.0)

