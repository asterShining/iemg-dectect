"""Time navigation and amplitude fitting use the curves actually on screen."""
from __future__ import annotations

import numpy as np
import pytest

from emgteach.gui.acquisition.plots import LivePlots
from emgteach.profiles import EMG_PROFILE

pytestmark = pytest.mark.gui


@pytest.fixture
def plots(qapp):
    widget = LivePlots(fs=4000, colours=[(65, 105, 225), (214, 39, 40)],
                       colour_hex=["#4169E1", "#D62728"],
                       profile=lambda: EMG_PROFILE, n_channels=lambda: 1,
                       acc_enabled=lambda: True, labels=lambda: ["EMG"])
    yield widget
    widget.stop_live()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def ranges(plots):
    return [p.getViewBox().viewRange()[0] for p in plots.plot_widgets()]


def review(plots):
    plots._revisando = True
    plots._review_duration = 20
    x = np.arange(80_000) / 4000
    plots._curves_raw[0].setData(x, np.sin(x) * 0.2)
    plots._curves_env[0].setData(x, np.ones(len(x)) * 0.1)
    plots._curve_acc.setData(x, 2 + np.sin(x) * 0.1)
    plots._set_shared_x(0, 20)


def test_review_buttons_zoom_about_centre_and_mouse_axis_stays_shared(plots):
    review(plots)
    plots._on_tiempo_reducir()
    for interval in ranges(plots):
        assert interval == pytest.approx([5, 15])
    plots._plot_env.setXRange(18, 28, padding=0)
    for interval in ranges(plots):
        assert interval == pytest.approx([10, 20])
    plots._btn_window_50ms.click()
    for interval in ranges(plots):
        assert interval == pytest.approx([14.975, 15.025])
    assert "50" in plots._lbl_ventana_info.text()


def test_live_shortcuts_and_mouse_zoom_follow_latest_sample(plots):
    data = np.sin(np.arange(40_000) / 10) * 0.05
    plots.append([data, data], [abs(data)])
    plots.redraw()
    plots._btn_window_100ms.click()
    assert plots._n_visible == 400
    for interval in ranges(plots):
        assert interval == pytest.approx([9.89975, 9.99975])
    plots._plot_acc.setXRange(1, 1.05, padding=0)
    plots.append([data[:400], data[:400]], [abs(data[:400])])
    plots.redraw()
    for interval in ranges(plots):
        assert interval == pytest.approx([10.04975, 10.09975])
    plots.set_time_window(0.001)
    assert ranges(plots)[0][1] - ranges(plots)[0][0] == pytest.approx(0.01)


def test_explicit_fit_uses_only_finite_visible_data_and_persists(plots):
    x = np.array([0, 1, 2, 3], dtype=float)
    plots._curves_raw[0].setData(x, [200, -0.2, 0.1, np.nan])
    plots._curves_env[0].setData(x, [100, 0.05, 0.1, np.nan])
    plots._curve_acc.setData(x, [100, 2, 3, np.nan])
    plots._set_shared_x(1, 3)
    plots.fit_amplitude()
    assert plots._plot_raw.getViewBox().viewRange()[1] == pytest.approx([-0.23, 0.23])
    assert plots._plot_env.getViewBox().viewRange()[1] == pytest.approx([0, 0.115])
    assert plots._plot_acc.getViewBox().viewRange()[1] == pytest.approx([1.925, 3.075])
    plots._escala_envolvente_en_vivo(500)
    assert plots._plot_env.getViewBox().viewRange()[1] == pytest.approx([0, 0.115])


def test_lhe_first_block_fits_once_then_preserves_manual_amplitude(plots):
    plots.follow_device(2, recording=False)
    plots.set_scale_controls_visible(False)
    assert not plots._btn_fit_amplitude.isHidden()
    assert not plots._btn_reset_escala.isHidden()
    plots.redraw()
    assert plots._cycle_auto_active
    plots.append([np.array([0.1, -0.2])], [np.array([0.01, 0.02])])
    plots.redraw()
    assert plots._plot_raw.getViewBox().viewRange()[1] == pytest.approx([-0.23, 0.23])
    plots._plot_raw.setYRange(-1, 1, padding=0)
    plots._plot_raw.getViewBox().sigRangeChangedManually.emit([False, True])
    plots.append([np.array([10.0])], [np.array([0.1])])
    plots.redraw()
    assert plots._plot_raw.getViewBox().viewRange()[1] == pytest.approx([-1, 1])


def test_reset_review_and_leave_review_do_not_fit_stale_curves(plots):
    review(plots)
    plots.set_time_window(0.05)
    plots.reset_all_scales()
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 20])
    plots.clear()
    plots.leave_review()
    assert not plots._revisando
    assert plots._review_duration == 0
    assert plots._n_visible == 20_000
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 5])


def test_fit_no_data_keeps_scale(plots):
    before = [p.getViewBox().viewRange()[1] for p in plots.plot_widgets()]
    plots.fit_amplitude()
    for plot, expected in zip(plots.plot_widgets(), before, strict=True):
        assert plot.getViewBox().viewRange()[1] == pytest.approx(expected)


def test_constant_traces_have_nonzero_fit_ranges(plots):
    x = np.arange(10) / 4000
    plots._curves_raw[0].setData(x, np.zeros(10))
    plots._curves_env[0].setData(x, np.zeros(10))
    plots._curve_acc.setData(x, np.full(10, 2.0))
    plots._set_shared_x(0, 0.01)
    plots.fit_amplitude()
    for plot in plots.plot_widgets():
        low, high = plot.getViewBox().viewRange()[1]
        assert np.isfinite(low) and np.isfinite(high) and high > low


def test_stacked_fit_keeps_lane_baselines_and_updates_gain(plots):
    plots._channel_count = lambda: 2
    plots._labels = lambda: ["A", "B"]
    plots.restack()
    x = np.arange(10) / 4000
    before = []
    for c in range(2):
        base = plots._lane_baseline(0, c)
        before.append(base)
        plots._curves_raw[c].setData(x, base + np.linspace(-0.1, 0.1, 10))
    plots._set_shared_x(0, 0.01)
    plots.fit_amplitude()
    assert plots._y_gain[0] > 1
    for c in range(2):
        _, y = plots._curves_raw[c].getData()
        assert (y.min() + y.max()) / 2 == pytest.approx(before[c])
        assert max(abs(y - before[c])) == pytest.approx(plots._lane_half(0) / 1.15)


def test_shown_widget_keeps_processing_events_after_axis_sync(plots, qapp):
    plots.show()
    data = np.sin(np.arange(4000) / 10)
    plots.append([data, data], [abs(data)])
    for _ in range(20):
        plots.redraw()
        qapp.processEvents()
    plots._plot_env.setXRange(0.2, 0.3, padding=0)
    for _ in range(20):
        qapp.processEvents()
    for interval in ranges(plots):
        assert interval == pytest.approx([0.2, 0.3])


def test_100hz_5mvpp_signal_has_five_unclipped_cycles_in_50ms(plots):
    from scipy.signal import find_peaks

    t = np.arange(4000) / 4000
    signal = 2.5 * np.sin(2 * np.pi * 100 * t)
    plots.append([signal], [abs(signal)])
    plots.set_time_window(0.05)
    plots.fit_amplitude()
    x, y = plots._curves_raw[0].getData()
    assert len(y) == 200
    peaks, _ = find_peaks(y)
    assert len(peaks) == 5
    assert y.max() - y.min() == pytest.approx(5)
    low, high = plots._plot_raw.getViewBox().viewRange()[1]
    assert low < y.min() and high > y.max()
    assert x[-1] - x[0] == pytest.approx(0.05 - 1 / 4000)


def test_short_review_clamps_to_entire_recording_below_minimum_window(plots):
    plots._revisando = True
    plots._review_duration = 0.005
    plots._set_shared_x(0, 0.005)
    plots.set_time_window(0.05)
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 0.005])
    plots._plot_acc.setXRange(-3, 1, padding=0)
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 0.005])


def test_empty_time_window_remains_requested_after_processing_events(plots, qapp):
    plots.show()
    plots.redraw()
    plots._on_tiempo_ampliar()
    for _ in range(10):
        qapp.processEvents()
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 10])
    assert "10.0" in plots._lbl_ventana_info.text()


def test_initial_shown_widget_starts_at_five_seconds_without_a_redraw(plots, qapp):
    plots.show()
    for _ in range(10):
        qapp.processEvents()
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 5])
    assert plots._view_seconds == pytest.approx(5)
    plots._on_tiempo_ampliar()
    qapp.processEvents()
    for interval in ranges(plots):
        assert interval == pytest.approx([0, 10])
