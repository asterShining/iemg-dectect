"""Auto observes raw sine cycles without changing acquired values."""
from __future__ import annotations

import numpy as np
import pytest

from emgteach.gui.acquisition.plots import LivePlots
from emgteach.profiles import EMG_PROFILE

pytestmark = pytest.mark.gui


@pytest.fixture
def view(qapp):
    plot = LivePlots(fs=4000, colours=[(65, 105, 225), (214, 39, 40)],
                     colour_hex=['#4169E1', '#D62728'], profile=lambda: EMG_PROFILE,
                     n_channels=lambda: 1, acc_enabled=lambda: False,
                     labels=lambda: ['EMG'])
    plot.follow_device(2, recording=False)
    yield plot
    plot.stop_live()
    plot.close()
    plot.deleteLater()
    qapp.processEvents()


def feed(view, seconds, frequency=100, amplitude=2.5):
    for _ in range(round(seconds * 4)):
        t = (view._total_samples + np.arange(1000)) / 4000
        raw = amplitude * np.sin(2 * np.pi * frequency * t)
        view.append([raw], [np.abs(raw)])
        view.redraw()


@pytest.mark.parametrize('frequency', [10, 50, 100, 200, 1000])
def test_auto_shows_five_cycles_or_minimum_window(view, frequency):
    assert view._btn_cycle_auto.isChecked()
    feed(view, 3, frequency)
    expected = max(0.01, 5 / frequency)
    assert view._view_seconds == pytest.approx(expected, rel=0.03)
    assert view._auto_frequency == pytest.approx(frequency, rel=0.01)
    raw_range = view._plot_raw.getViewBox().viewRange()[0]
    assert raw_range[1] == pytest.approx(view._total_samples / 4000 - 1 / 4000)
    assert view._plot_env.getViewBox().viewRange()[0] == pytest.approx(raw_range)
    lo, hi = view._plot_raw.getViewBox().viewRange()[1]
    assert -lo == pytest.approx(hi)
    assert hi >= 2.5


def test_frequency_and_amplitude_changes_follow_without_clipping(view):
    feed(view, 3, 100)
    assert view._view_seconds == pytest.approx(.05, rel=.03)
    feed(view, 3, 50, amplitude=5)
    assert view._view_seconds == pytest.approx(.1, rel=.03)
    assert view._plot_raw.getViewBox().viewRange()[1][1] >= 5
    feed(view, .25, 50, amplitude=.5)
    half = view._plot_raw.getViewBox().viewRange()[1][1]
    assert .575 < half < 5.75
    feed(view, 3, 50, amplitude=.5)
    assert view._plot_raw.getViewBox().viewRange()[1][1] < half


def test_nonperiodic_or_flat_signal_falls_back_after_two_seconds(view):
    feed(view, 3)
    rng = np.random.default_rng(23)
    for _ in range(20):
        raw = rng.normal(0, .1, 1000)
        view.append([raw], [abs(raw)])
        view.redraw()
    assert view._cycle_auto_active
    assert view._view_seconds == pytest.approx(5)
    assert view._auto_frequency is None
    feed(view, 3, amplitude=0)
    low, high = view._plot_raw.getViewBox().viewRange()[1]
    assert np.isfinite(low) and high > low


@pytest.mark.parametrize('action', ['window', 'fit', 'mouse', 'vertical'])
def test_manual_actions_exit_and_auto_can_restart(view, action):
    feed(view, 3)
    if action == 'window':
        view._btn_window_100ms.click()
    elif action == 'fit':
        view.fit_amplitude()
    elif action == 'mouse':
        view._plot_raw.getViewBox().sigRangeChangedManually.emit([True, True])
    else:
        view._y_zoom(0, True)
    assert not view._cycle_auto_active
    assert not view._cycle_auto_preference
    view._btn_cycle_auto.click()
    feed(view, 1)
    assert view._cycle_auto_active
    assert view._view_seconds == pytest.approx(.05, rel=.03)


def test_stopping_preserves_preference_and_restarting_rearms_detection(view):
    feed(view, 3)
    view.stop_live()
    assert not view._cycle_auto_active
    assert view._cycle_auto_preference
    view.clear()
    view.start_live()
    feed(view, 1)
    assert view._cycle_auto_active
    assert view._view_seconds == pytest.approx(.05, rel=.03)


def test_multichannel_disables_auto_without_changing_lane_mode(view):
    view._channel_count = lambda: 2
    view.set_channels(2)
    assert not view._cycle_auto_active
    assert not view._btn_cycle_auto.isEnabled()


def test_programmatic_new_session_reset_preserves_auto_preference(view):
    feed(view, 3)
    view.clear()
    view.reset_all_scales(preserve_auto=True)
    assert view._cycle_auto_preference
    assert view._cycle_auto_active
    feed(view, 1)
    assert view._view_seconds == pytest.approx(.05, rel=.03)


def test_manual_reset_exits_auto(view):
    feed(view, 3)
    view._btn_reset_escala.click()
    assert not view._cycle_auto_active
    assert not view._cycle_auto_preference
    assert view._view_seconds == pytest.approx(5)


def test_unstable_frequency_candidates_also_fall_back(view, monkeypatch):
    from itertools import cycle

    feed(view, 3)
    candidates = cycle([50.0, 200.0])
    monkeypatch.setattr(
        "emgteach.gui.acquisition.plots.estimate_sine_frequency",
        lambda *_: next(candidates),
    )
    feed(view, 3)
    assert view._auto_frequency is None
    assert view._view_seconds == pytest.approx(5)


def test_review_auto_uses_original_samples_and_fits_five_cycles(view):
    view._revisando = True
    view._review_duration = 15
    # Deliberately different from the live sampling rate.
    x = np.arange(15000) / 1000
    raw = 2.5 * np.sin(2 * np.pi * 100 * x)
    view._curves_raw[0].setData(x, raw)
    view._curves_env[0].setData(x, np.full(len(x), .2))
    view._plot_raw.setDownsampling(ds=20, auto=False, mode='peak')
    view._set_shared_x(0, 15)
    view._set_cycle_auto(True)
    assert view._btn_cycle_auto.isChecked()
    assert view._view_seconds == pytest.approx(.05, rel=.02)
    assert view._auto_frequency == pytest.approx(100, rel=.01)
    left, right = view._plot_raw.getViewBox().viewRange()[0]
    assert (left + right) / 2 == pytest.approx(7.5)
    assert view._plot_env.getViewBox().viewRange()[0] == pytest.approx([left, right])
    low, high = view._plot_raw.getViewBox().viewRange()[1]
    assert low < -2.3 and high > 2.3
    view._btn_reset_escala.click()
    assert view._view_seconds == pytest.approx(15)
    assert not view._cycle_auto_active
    view._btn_cycle_auto.click()
    assert view._view_seconds == pytest.approx(.05, rel=.02)


def test_review_auto_nonperiodic_preserves_overall_view(view):
    view._revisando = True
    view._review_duration = 4
    x = np.arange(16000) / 4000
    view._curves_raw[0].setData(x, np.random.default_rng(3).normal(size=len(x)))
    view._set_shared_x(1, 2)
    view._set_cycle_auto(True)
    assert view._view_seconds == pytest.approx(4)
    assert view._auto_frequency is None
    assert view._cycle_auto_preference
