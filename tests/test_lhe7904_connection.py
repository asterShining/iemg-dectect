"""LHE7904 connection configuration without serial hardware."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from emgteach.devices import BACKEND_ARDUINO, BACKEND_LHE7904_UART
from emgteach.gui.acquisition import connection

pytestmark = pytest.mark.gui


@pytest.fixture
def box(qapp, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "connection.ini"), QSettings.Format.IniFormat)
    settings.setValue("adquisicion/port", "COM5")
    calls = []

    class Device:
        @staticmethod
        def list_ports():
            return ["COM5", "/dev/ttyUSB0"]

    def device_class(backend):
        calls.append(backend)
        return Device

    monkeypatch.setattr(connection, "device_class", device_class)
    monkeypatch.setattr(connection, "read_station_address", lambda: None)
    widget = connection.ConnectionBox(settings, lambda: None)
    yield widget, settings, calls
    widget.stop()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_lhe7904_has_its_own_serial_configuration(box):
    widget, _settings, calls = box
    assert widget._combo_device_type.count() == 3
    assert widget._widget_lhe7904.isHidden()
    widget._combo_device_type.setCurrentIndex(2)
    assert not widget._widget_arduino.isHidden()
    assert not widget._widget_lhe7904.isHidden()
    assert widget._widget_mac.isHidden()
    assert widget._combo_port.isEditable()
    assert calls[-1] == BACKEND_LHE7904_UART
    assert widget.lhe7904_options() == {
        "fs": 4000, "include_filtered": True, "pga_gain": 1,
    }
    assert [widget._combo_lhe_fs.itemData(i) for i in range(3)] == [1000, 2000, 4000]
    widget._combo_device_type.setCurrentIndex(1)
    assert widget._widget_lhe7904.isHidden()
    assert not widget._combo_port.isEditable()
    assert calls[-1] == BACKEND_ARDUINO


def test_options_signal_and_settings_survive_reconstruction(box, qapp):
    widget, settings, _calls = box
    changed = []
    widget.configuration_changed.connect(lambda: changed.append(widget.lhe7904_options()))
    widget._combo_device_type.setCurrentIndex(2)
    widget._combo_lhe_fs.setCurrentIndex(1)
    widget._chk_lhe_filtered.setChecked(False)
    widget._combo_lhe_gain.setCurrentIndex(7)
    widget._combo_port.setEditText("/dev/serial/by-id/lhe7904")
    widget.connect_button.setChecked(True)
    assert widget.connect_device() == "/dev/serial/by-id/lhe7904"
    assert len(changed) == 3
    assert settings.value("adquisicion/port") == "COM5"
    assert settings.value("adquisicion/lhe7904/port") == "/dev/serial/by-id/lhe7904"
    restored = connection.ConnectionBox(settings, lambda: None)
    try:
        assert restored.device_index() == 2
        assert restored.serial_port() == "/dev/serial/by-id/lhe7904"
        assert restored.lhe7904_options() == {
            "fs": 2000, "include_filtered": False, "pga_gain": 24,
        }
    finally:
        restored.stop()
        restored.deleteLater()
        qapp.processEvents()


def test_connection_locks_lhe_options_until_disconnect(box):
    widget, _settings, _calls = box
    widget._combo_device_type.setCurrentIndex(2)
    widget._combo_port.setEditText("COM9")
    widget.connect_button.setChecked(True)
    assert widget.connect_device() == "COM9"
    assert not widget._widget_lhe7904.isEnabled()
    assert not widget._combo_port.isEnabled()
    widget.disconnect_device()
    assert widget._widget_lhe7904.isEnabled()
    assert widget._combo_port.isEnabled()


def test_lhe7904_requires_a_port_and_bit_only_practicals_still_switch(box):
    widget, _settings, _calls = box
    widget._combo_device_type.setCurrentIndex(2)
    widget._combo_port.setEditText("")
    warnings = []
    widget.warned.connect(warnings.append)
    widget.connect_button.setChecked(True)
    assert widget.connect_device() is None
    assert len(warnings) == 1
    assert not widget.connect_button.isChecked()
    widget.set_practical(False)
    assert widget.device_index() == 0
    assert widget._widget_lhe7904.isHidden()
