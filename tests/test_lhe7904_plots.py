"""The device sampling clock controls live plot duration and scale."""
from __future__ import annotations

import numpy as np
import pytest

from emgteach.gui.acquisition.plots import MAX_POINTS, LivePlots
from emgteach.profiles import EMG_PROFILE

pytestmark = pytest.mark.gui


@pytest.fixture
def plots(qapp):
    widget = LivePlots(
        fs=1000,
        colours=[(65, 105, 225)],
        colour_hex=["#4169E1"],
        profile=lambda: EMG_PROFILE,
        n_channels=lambda: 1,
        acc_enabled=lambda: False,
        labels=lambda: ["EMG"],
    )
    yield widget
    widget.stop_live()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_4000_hz_keeps_thirty_seconds_and_the_absolute_clock(plots):
    plots.set_sample_frequency(4000)
    assert MAX_POINTS == 30_000
    assert plots._buf_raw[0].maxlen == 120_000
    assert plots._buf_env[0].maxlen == 120_000
    assert plots._buf_acc.maxlen == 120_000
    assert plots._n_visible == 20_000
    samples = np.zeros(120_000)
    plots.append([samples], [samples])
    plots.redraw()
    x, _ = plots._curves_raw[0].getData()
    assert x[0] == pytest.approx(25.0)
    assert x[-1] == pytest.approx(30 - 1 / 4000)
    plots._on_combo_zoom_changed(0)
    assert plots._n_visible == 120_000
    x, _ = plots._curves_raw[0].getData()
    assert x[0] == pytest.approx(0)


def test_switch_back_preserves_window_seconds_and_clears_old_clock(plots):
    plots.set_sample_frequency(4000)
    plots._n_visible = 40_000  # ten seconds
    plots.append([np.ones(4000)], [np.ones(4000)])
    plots.add_marker(0.5, "old clock")
    plots.redraw()
    plots.set_sample_frequency(1000)
    assert plots._n_visible == 10_000
    assert len(plots._buf_raw[0]) == 30_000
    assert plots._buf_raw[0].maxlen == 30_000
    assert plots._total_samples == 0
    assert plots._marker_events == []
    x, _ = plots._curves_raw[0].getData()
    assert x is None or len(x) == 0
    assert plots._plot_raw.getViewBox().viewRange()[0] == pytest.approx([0, 10])


def test_same_frequency_does_not_discard_live_data(plots):
    plots.append([np.ones(1000)], [np.ones(1000)])
    plots.set_sample_frequency(1000)
    assert plots._total_samples == 1000


def test_lhe7904_full_scale_tracks_configured_pga(plots):
    plots.follow_device(2, recording=False)
    assert plots._y_ranges_init[0] == (-2400, 2400)
    plots.follow_device(2, recording=False, sensor_gain=24)
    assert plots._y_ranges_init[0] == (-100, 100)


@pytest.mark.parametrize("fs", [0, -1, 1000.5])
def test_invalid_sample_rate_is_rejected_without_changing_clock(plots, fs):
    with pytest.raises(ValueError, match="positive integer"):
        plots.set_sample_frequency(fs)
    assert plots._fs == 1000
