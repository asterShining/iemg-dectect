"""The live Auto window follows a clear sine and rejects ambiguous input."""
from __future__ import annotations

import numpy as np
import pytest

from emgteach.gui.acquisition.cycle_estimator import estimate_sine_frequency


def sine(frequency, *, fs=4000, seconds=2, amplitude=2.5, offset=0):
    t = np.arange(round(fs * seconds)) / fs
    return offset + amplitude * np.sin(2 * np.pi * frequency * t + 0.3)


@pytest.mark.parametrize("frequency", [10, 50, 100, 200, 1000])
def test_recognizes_requested_frequencies(frequency):
    assert estimate_sine_frequency(sine(frequency), 4000) == pytest.approx(frequency, abs=0.01)


def test_interpolation_recognizes_frequency_between_fft_bins():
    assert estimate_sine_frequency(sine(100.13), 4000) == pytest.approx(100.13, abs=0.02)


def test_offset_and_moderate_noise_keep_the_sine_estimate():
    samples = sine(100, offset=25)
    samples += np.random.default_rng(13).normal(0, 0.2, len(samples))
    assert estimate_sine_frequency(samples, 4000) == pytest.approx(100, abs=0.02)


def test_uses_the_supplied_sampling_rate():
    assert estimate_sine_frequency(sine(80, fs=1000), 1000) == pytest.approx(80, abs=0.01)


def test_uses_only_the_latest_two_seconds():
    samples = np.concatenate([sine(50), sine(100)])
    assert estimate_sine_frequency(samples, 4000) == pytest.approx(100, abs=0.01)


def test_half_second_is_enough_for_a_high_frequency():
    assert estimate_sine_frequency(sine(100, seconds=0.5), 4000) == pytest.approx(100, abs=0.02)


@pytest.mark.parametrize("samples", [[], [0] * 8000, [3] * 8000, [[0, 1], [2, 3]]])
def test_empty_flat_and_wrong_dimension_have_no_estimate(samples):
    assert estimate_sine_frequency(samples, 4000) is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_sample_has_no_estimate(invalid):
    samples = sine(100)
    samples[100] = invalid
    assert estimate_sine_frequency(samples, 4000) is None


@pytest.mark.parametrize("fs", [0, -1, 0.1, np.nan, np.inf, 1e308, True, "invalid"])
def test_invalid_sample_rate_has_no_estimate(fs):
    assert estimate_sine_frequency(sine(100), fs) is None


def test_short_observation_has_no_estimate():
    assert estimate_sine_frequency(sine(100, seconds=0.49), 4000) is None


def test_less_than_three_cycles_has_no_estimate():
    assert estimate_sine_frequency(sine(1), 4000) is None


def test_broadband_noise_has_no_estimate():
    noise = np.random.default_rng(13).normal(size=8000)
    assert estimate_sine_frequency(noise, 4000) is None


def test_two_equal_separated_tones_have_no_single_estimate():
    assert estimate_sine_frequency(sine(50) + sine(100), 4000) is None


def test_small_finite_sine_is_not_removed_by_an_arbitrary_voltage_floor():
    assert estimate_sine_frequency(sine(100, amplitude=1e-100), 4000) == pytest.approx(100)


def test_variation_at_a_large_offsets_rounding_level_is_rejected():
    assert estimate_sine_frequency(sine(100, amplitude=1e-15, offset=10), 4000) is None
