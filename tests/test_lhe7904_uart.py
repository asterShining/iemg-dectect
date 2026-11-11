"""Exercise the actual LHE7904 decoder without attached hardware."""

from __future__ import annotations

import struct
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emgteach.devices.lhe7904_uart import LHE7904UartDevice

TAIL = b"\x00\x00\x80\x7f"


def frame(raw: float, timestamp: float, filtered: bool = True) -> bytes:
    values = (raw, raw / 2, timestamp) if filtered else (raw, timestamp)
    return struct.pack("<" + "f" * len(values), *values) + TAIL


class FakeSerial:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = bytearray()
        self.chunk_size = 4096
        self.is_open = True

    def read(self, size):
        count = min(size, self.chunk_size)
        data = bytes(self.data[:count])
        del self.data[:count]
        return data

    def close(self):
        self.is_open = False


@pytest.fixture
def serial_factory(monkeypatch):
    factory = Mock(side_effect=FakeSerial)
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=factory))
    monkeypatch.setattr(LHE7904UartDevice, "_TIMEOUT_READ_S", 0.02)
    return factory


@pytest.mark.parametrize("filtered", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_frames_arbitrary_chunks_and_persistent_buffer(serial_factory, filtered, chunk_size):
    device = LHE7904UartDevice("COM4", include_filtered=filtered, pga_gain=24)
    device.open()
    ser = device._connection.serial
    ser.chunk_size = chunk_size
    ser.data.extend(
        b"noise" + b"".join(frame(x, i / 4000, filtered) for i, x in enumerate([1, -2, 3, 4]))
    )
    first = device.read(2)
    second = device.read(2)
    assert first.dtype == np.float64
    assert first.shape == (2, 1)
    np.testing.assert_array_equal(np.vstack([first, second])[:, 0], [1, -2, 3, 4])
    assert device.n_channels == 1
    assert device.channel_units() == ["mV"]
    assert device.channel_physical_ranges() == [(-100, 100)]
    assert device.channel_digital_ranges() == [(-32768, 32767)]
    device.close()


def test_open_passive_and_serial_configuration(serial_factory):
    device = LHE7904UartDevice("COM9")
    device.open()
    kwargs = device._connection.serial.kwargs
    assert kwargs == dict(
        port="COM9",
        baudrate=1_000_000,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0.1,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    # FakeSerial has no write method: open/close must never transmit commands.
    assert device.fs == 4000
    assert device.name == "LHE7904 UART (COM9)"
    assert device.physical_max == 2400
    with pytest.raises(RuntimeError, match="already open"):
        device.open()
    device.close()
    device.close()


@pytest.mark.parametrize("timestamp", [0.01, 0.0])
def test_clear_gap_or_reset_rejected(serial_factory, timestamp):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(frame(1, 0.001) + frame(2, 0.00125) + frame(3, timestamp))
    with pytest.raises(RuntimeError, match="timestamp discontinuity"):
        device.read(3)
    device.close()


def test_long_uptime_float_rounding_not_reported_as_gap(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(b"".join(frame(1, 5000 + i / 4000) for i in range(100)))
    assert device.read(100).shape == (100, 1)
    device.close()


@pytest.mark.parametrize("bad", [frame(float("nan"), 0), frame(1, float("inf")), frame(1, -1)])
def test_invalid_initial_frames_discarded(serial_factory, bad):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(bad + frame(2, 1) + frame(3, 1.00025))
    np.testing.assert_array_equal(device.read(2)[:, 0], [2, 3])
    device.close()


def test_recovers_framing_without_silently_compressing_lost_sample(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(frame(1, 0) + frame(2, 0.00025))
    device.read(2)
    ser.data.extend(b"broken frame" + frame(4, 0.00075) + frame(5, 0.001))
    with pytest.raises(RuntimeError, match="timestamp discontinuity"):
        device.read(2)
    device.close()


def test_resynchronizes_inserted_noise_between_intact_samples(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(frame(1, 0) + frame(2, 0.00025))
    device.read(2)
    ser.data.extend(b"junk" + frame(3, 0.0005) + frame(4, 0.00075))
    np.testing.assert_array_equal(device.read(2)[:, 0], [3, 4])
    device.close()


def test_timeout_with_noise_keeps_buffer_bounded(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(b"junk" * 10_000)
    with pytest.raises(RuntimeError, match="Timeout"):
        device.read(1)
    assert len(device._connection.buffer) < 32
    device.close()


def test_wrong_layout_times_out(serial_factory):
    device = LHE7904UartDevice("COM4", include_filtered=False)
    device.open()
    device._connection.serial.data.extend(frame(1, 0) + frame(2, 0.00025))
    with pytest.raises(RuntimeError, match="Timeout"):
        device.read(2)
    device.close()


def test_disconnect_reported(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    device._connection.serial.read = Mock(side_effect=OSError("unplugged"))
    with pytest.raises(RuntimeError, match="connection lost"):
        device.read(1)
    device.close()


def test_force_close_unblocks_read_and_reopen_resets_time(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    entered, released = threading.Event(), threading.Event()
    errors = []

    def blocking_read(size):
        entered.set()
        assert released.wait(1)
        return b""

    def read():
        try:
            device.read(1)
        except RuntimeError as exc:
            errors.append(str(exc))

    ser.read = blocking_read
    ser.cancel_read = released.set
    thread = threading.Thread(target=read)
    thread.start()
    assert entered.wait(1)
    device.force_close()
    thread.join(1)
    assert not thread.is_alive()
    assert errors and "closed" in errors[0]
    assert not ser.is_open
    device.open()
    device._connection.serial.data.extend(frame(7, 0) + frame(8, 0.00025))
    np.testing.assert_array_equal(device.read(2)[:, 0], [7, 8])
    device.close()


@pytest.mark.parametrize(
    "kwargs", [{"fs": 8000}, {"fs": 0}, {"pga_gain": 5}, {"include_filtered": "yes"}]
)
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        LHE7904UartDevice("COM4", **kwargs)


def test_read_argument_and_closed_state(serial_factory):
    device = LHE7904UartDevice("COM4")
    with pytest.raises(RuntimeError, match="not open"):
        device.read(1)
    device.open()
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError):
            device.read(value)
    assert device.read(0).shape == (0, 1)
    device.close()


def test_open_mid_frame_retains_first_complete_sample(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.data.extend(frame(99, 0)[7:] + frame(2, 0.00025) + frame(3, 0.0005))
    np.testing.assert_array_equal(device.read(2)[:, 0], [2, 3])
    device.close()


def test_mismatched_sample_rate_rejected(serial_factory):
    device = LHE7904UartDevice("COM4", fs=1000)
    device.open()
    device._connection.serial.data.extend(frame(1, 0) + frame(2, 0.00025))
    with pytest.raises(RuntimeError, match="mismatched sampling rate"):
        device.read(2)
    device.close()


def test_force_close_tolerates_unplugged_handle(serial_factory):
    device = LHE7904UartDevice("COM4")
    device.open()
    ser = device._connection.serial
    ser.cancel_read = Mock(side_effect=OSError("unplugged"))
    ser.close = Mock(side_effect=OSError("invalid handle"))
    device.force_close()
    device.force_close()
    ser.close.assert_called_once()
