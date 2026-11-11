"""LHE7904 UART21 JustFloat acquisition (1,000,000 baud, 8N1).

Each sample is ``raw_mV, [filtered_mV,] time_s`` as little-endian float32,
followed by ``00 00 80 7f``. The optional filtered value is another view of
ADC channel 1, not a second EMG channel. Only raw mV enters the application's
processing chain. The firmware streams continuously: no commands are sent.

There is no CRC, sequence number, sample-rate or input-mode metadata. Settings
must match the firmware. Timestamps detect clear discontinuities, but float32
precision decreases with uptime and cannot prove the absence of lost samples.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from emgteach.devices.base import AcquisitionDevice


@dataclass
class _Connection:
    serial: Any
    buffer: bytearray = field(default_factory=bytearray)
    synchronized: bool = False
    timestamp: float | None = None


class LHE7904UartDevice(AcquisitionDevice):
    """Read one EMG channel, already in input-referred millivolts.

    ``fs``, ``include_filtered`` and ``pga_gain`` describe the firmware; they
    do not configure the board. PGA affects the physical range only, never
    the received values. A new connection resets framing and timestamp state.
    """

    _BAUD = 1_000_000
    _TIMEOUT_READ_S = 3.0
    _POLL_S = 0.1
    _CHUNK_BYTES = 4096
    _TAIL = b"\x00\x00\x80\x7f"
    _GAINS = (1, 2, 3, 4, 6, 8, 12, 24)
    _RATES = (500, 1000, 2000, 4000)

    def __init__(
        self,
        port: str,
        fs: int = 4000,
        *,
        include_filtered: bool = True,
        pga_gain: int = 1,
    ) -> None:
        # At 1 Mbaud even the 12-byte frame cannot carry 8000 SPS (8N1).
        if fs not in self._RATES:
            raise ValueError("LHE7904 UART sampling rate must be 500, 1000, 2000 or 4000 Hz.")
        if pga_gain not in self._GAINS:
            raise ValueError(f"LHE7904 PGA gain must be one of {self._GAINS}.")
        if not isinstance(include_filtered, bool):
            raise ValueError("include_filtered must be a boolean matching the firmware.")
        self._port = port
        self._fs = int(fs)
        self._gain = pga_gain
        self._decoder = struct.Struct("<fff" if include_filtered else "<ff")
        self._frame_bytes = self._decoder.size + len(self._TAIL)
        self._connection: _Connection | None = None
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()

    @property
    def fs(self) -> float:
        return float(self._fs)

    @property
    def name(self) -> str:
        return f"LHE7904 UART ({self._port})"

    @classmethod
    def full_scale_mv(cls, sensor_gain: float | None = None) -> float:
        gain = 1.0 if sensor_gain is None else float(sensor_gain)
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Sensor gain must be finite and positive.")
        return 2400.0 / gain

    @property
    def physical_min(self) -> float:
        return -self.full_scale_mv(self._gain)

    @property
    def physical_max(self) -> float:
        return self.full_scale_mv(self._gain)

    def channel_digital_ranges(self) -> list[tuple[int, int]]:
        # EDF stores int16: use the entire range rather than a 10-bit
        # Arduino-compatible default, especially with the wide PGA=1 span.
        return [(-32768, 32767)]

    @staticmethod
    def list_ports() -> list[str]:
        from serial.tools import list_ports

        return [p.device for p in list_ports.comports()]

    def open(self) -> None:
        import serial

        with self._lock:
            if self._connection is not None:
                raise RuntimeError("The LHE7904 serial port is already open.")
            ser = serial.Serial(
                port=self._port,
                baudrate=self._BAUD,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=self._POLL_S,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            self._connection = _Connection(ser)

    def _decode(self, buffer: bytearray, offset: int = 0) -> tuple[float, ...] | None:
        end = offset + self._frame_bytes
        if buffer[end - 4 : end] != self._TAIL:
            return None
        values = self._decoder.unpack_from(buffer, offset)
        if not all(math.isfinite(v) for v in values) or values[-1] < 0:
            return None
        return values

    def _next_sample(self, connection: _Connection) -> float | None:
        buf = connection.buffer
        while len(buf) >= self._frame_bytes:
            values = self._decode(buf)
            if values is None:
                del buf[0]
                connection.synchronized = False
                continue
            if not connection.synchronized:
                # Two consecutive finite frames establish the configured layout,
                # rather than accepting a tail encountered in an arbitrary payload.
                if len(buf) < 2 * self._frame_bytes:
                    return None
                if self._decode(buf, self._frame_bytes) is None:
                    del buf[0]
                    continue
                connection.synchronized = True
            timestamp = values[-1]
            previous = connection.timestamp
            if previous is not None:
                period = 1.0 / self._fs
                # Both endpoints were rounded to float32 seconds. At long uptimes
                # adjacent times may be equal; do not misreport that as a reset.
                rounding = max(
                    float(np.spacing(np.float32(previous))),
                    float(np.spacing(np.float32(timestamp))),
                )
                tolerance = period * 0.35 + rounding
                if abs(timestamp - previous - period) > tolerance:
                    raise RuntimeError(
                        "LHE7904 sample timestamp discontinuity: "
                        f"{previous:.9g} -> {timestamp:.9g} s; "
                        "recording stopped (sample loss, reset or mismatched sampling rate)."
                    )
            connection.timestamp = timestamp
            del buf[: self._frame_bytes]
            return values[0]
        return None

    def read(self, n_samples: int) -> npt.NDArray[np.float64]:
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 0
        ):
            raise ValueError("n_samples must be a non-negative integer.")
        with self._read_lock:
            with self._lock:
                connection = self._connection
            if connection is None:
                raise RuntimeError("The LHE7904 device is not open.")
            result = np.empty((n_samples, 1), dtype=np.float64)
            deadline = time.monotonic() + self._TIMEOUT_READ_S + n_samples / self._fs
            index = 0
            while index < n_samples:
                with self._lock:
                    if self._connection is not connection:
                        raise RuntimeError("The LHE7904 device was closed while reading.")
                value = self._next_sample(connection)
                if value is not None:
                    result[index, 0] = value
                    index += 1
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Timeout while reading LHE7904; check connection and frame format."
                    )
                try:
                    chunk = connection.serial.read(
                        min(
                            self._CHUNK_BYTES,
                            max(2 * self._frame_bytes, (n_samples - index) * self._frame_bytes),
                        )
                    )
                except (OSError, ValueError) as exc:
                    raise RuntimeError("LHE7904 serial connection lost or closed.") from exc
                connection.buffer.extend(chunk)
                # Each read is bounded, and the parser discards noise until at
                # most two partial frames remain before requesting more bytes.
            return result

    def close(self) -> None:
        self.force_close()

    def force_close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            cancel = getattr(connection.serial, "cancel_read", None)
            if cancel is not None:
                cancel()
        except OSError:
            pass  # Unplugging may already have invalidated the OS handle.
        finally:
            try:
                connection.serial.close()
            except OSError:
                pass
