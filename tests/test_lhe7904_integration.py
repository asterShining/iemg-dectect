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
