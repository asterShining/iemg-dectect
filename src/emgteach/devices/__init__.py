"""Hardware backends for biopotential acquisition.

Interchangeable backends are provided:

- :class:`ArduinoDevice` — Arduino RedBoard Plus + MyoWare 2.0 over USB serial
- :class:`BitalinoDevice` — BITalino *(revolution)* over Bluetooth (optional;
  requires the ``bitalino`` extra)

- :class:`LHE7904UartDevice` — LHE7904 UART JustFloat at 1 Mbaud

All implement the :class:`AcquisitionDevice` interface so the rest of
the package (workers, GUI tabs) is hardware-agnostic.
"""

from __future__ import annotations

from emgteach.devices.arduino import ArduinoDevice
from emgteach.devices.base import AcquisitionDevice
from emgteach.devices.bitalino import BitalinoDevice
from emgteach.devices.factory import (
    BACKEND_ARDUINO,
    BACKEND_BITALINO,
    BACKEND_LHE7904_UART,
    available_backends,
    create_device,
    device_class,
    register_device,
)
from emgteach.devices.lhe7904_uart import LHE7904UartDevice

__all__ = [
    "BACKEND_ARDUINO",
    "BACKEND_BITALINO",
    "BACKEND_LHE7904_UART",
    "AcquisitionDevice",
    "ArduinoDevice",
    "BitalinoDevice",
    "LHE7904UartDevice",
    "available_backends",
    "create_device",
    "device_class",
    "register_device",
]
