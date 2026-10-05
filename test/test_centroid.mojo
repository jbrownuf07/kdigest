"""Tests for the Centroid weighted-mean update."""

from std.testing import assert_almost_equal, assert_equal, TestSuite

from stream_fire.centroid import Centroid


def test_singleton() raises:
    var c = Centroid.singleton(3.5)
    assert_almost_equal(c.mean, 3.5)
    assert_almost_equal(c.weight, 1.0)


def test_add_two_unit_samples_averages() raises:
    var c = Centroid.singleton(5.0)
    c.add_sample(7.0, 1.0)
    assert_almost_equal(c.mean, 6.0)
    assert_almost_equal(c.weight, 2.0)


def test_weighted_mean_matches_closed_form() raises:
    # Fold several weighted samples; compare against the direct weighted mean.
    var values = [1.0, 10.0, 100.0, 2.0]
    var weights = [3.0, 1.0, 0.5, 4.0]

    var c = Centroid(mean=values[0], weight=weights[0])
    for i in range(1, len(values)):
        c.add_sample(values[i], weights[i])

    var num = 0.0
    var den = 0.0
    for i in range(len(values)):
        num += values[i] * weights[i]
        den += weights[i]

    assert_almost_equal(c.mean, num / den, atol=1e-12)
    assert_almost_equal(c.weight, den, atol=1e-12)


def test_merge_equivalent_to_add_sample() raises:
    var a = Centroid(mean=4.0, weight=2.0)
    var b = Centroid(mean=10.0, weight=3.0)
    a.merge(b)
    # (4*2 + 10*3) / 5 = 38/5 = 7.6
    assert_almost_equal(a.mean, 7.6, atol=1e-12)
    assert_almost_equal(a.weight, 5.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
