"""Concurrent EDF inspection must not invalidate another reader's samples."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from emgteach.io import (
    BufferedEdfWriter,
    ChannelInfo,
    RecordingMetadata,
    list_edf_emg_channels,
    read_edf_metadata,
    read_edf_pyedflib,
)


def test_parallel_header_and_signal_reads_preserve_samples(tmp_path, capfd):
    path = tmp_path / "parallel.edf"
    expected = np.sin(2 * np.pi * 80 * np.arange(84000) / 4000)
    with BufferedEdfWriter(
        path, [ChannelInfo("EMG", sample_frequency=4000)],
        RecordingMetadata(protocol="sine"),
    ) as writer:
        writer.add_samples(expected)
    reference = read_edf_pyedflib(path)["emg_raw"]

    def read(index):
        if index % 3 == 0:
            return read_edf_pyedflib(path)["emg_raw"]
        if index % 3 == 1:
            return read_edf_metadata(path)
        return list_edf_emg_channels(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, range(90)))
    for result in results[::3]:
        np.testing.assert_array_equal(result, reference)
    assert "less than" not in capfd.readouterr().out
