"""Tests for the scale functions K0..K3.

Validates the three invariants from PLAN.md Phase 1:
  1. Round-trip identity: q(k(q)) == q
  2. Monotonicity: k is strictly increasing in q
  3. Tail steepness: dk/dq is larger near the tails than in the middle
     (this is the property that drives extreme-quantile accuracy).
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from stream_fire.scale import ScaleFunction, ScaleK0, ScaleK1, ScaleK2, ScaleK3


comptime COMPRESSION = 200.0
comptime N = 100_000.0


def check_round_trip[S: ScaleFunction](scale: S) raises:
    var nrm = scale.normalizer(COMPRESSION, N)
    var qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    for i in range(len(qs)):
        var q = qs[i]
        var round = scale.q(scale.k(q, nrm), nrm)
        assert_almost_equal(round, q, atol=1e-6)


def check_monotonic[S: ScaleFunction](scale: S) raises:
    var nrm = scale.normalizer(COMPRESSION, N)
    var prev = scale.k(1e-6, nrm)
    var steps = 200
    for i in range(1, steps):
        var q = Float64(i) / Float64(steps)
        var cur = scale.k(q, nrm)
        assert_true(cur > prev, "k must strictly increase in q")
        prev = cur


def check_tail_steepness[S: ScaleFunction](scale: S) raises:
    # Compare dk/dq near the lower tail, middle, and upper tail using a small
    # symmetric finite difference. For tail-controlling scale functions the
    # slope at the extremes must exceed the slope at the median.
    var nrm = scale.normalizer(COMPRESSION, N)
    var h = 1e-4

    var low = (scale.k(0.01 + h, nrm) - scale.k(0.01 - h, nrm)) / (2.0 * h)
    var mid = (scale.k(0.5 + h, nrm) - scale.k(0.5 - h, nrm)) / (2.0 * h)
    var high = (scale.k(0.99 + h, nrm) - scale.k(0.99 - h, nrm)) / (2.0 * h)
    assert_true(low > mid, "lower-tail slope must exceed median slope")
    assert_true(high > mid, "upper-tail slope must exceed median slope")


# --- Round-trip (all four) ---


def test_round_trip_k0() raises:
    check_round_trip(ScaleK0())


def test_round_trip_k1() raises:
    check_round_trip(ScaleK1())


def test_round_trip_k2() raises:
    check_round_trip(ScaleK2())


def test_round_trip_k3() raises:
    check_round_trip(ScaleK3())


# --- Monotonicity (all four) ---


def test_monotonic_k0() raises:
    check_monotonic(ScaleK0())


def test_monotonic_k1() raises:
    check_monotonic(ScaleK1())


def test_monotonic_k2() raises:
    check_monotonic(ScaleK2())


def test_monotonic_k3() raises:
    check_monotonic(ScaleK3())


# --- Tail steepness (K1..K3; K0 is linear by design, so it is excluded) ---


def test_tail_steepness_k1() raises:
    check_tail_steepness(ScaleK1())


def test_tail_steepness_k2() raises:
    check_tail_steepness(ScaleK2())


def test_tail_steepness_k3() raises:
    check_tail_steepness(ScaleK3())


def test_k0_is_linear() raises:
    # K0's slope is constant: tails and middle have equal steepness.
    var s = ScaleK0()
    var nrm = s.normalizer(COMPRESSION, N)
    var h = 1e-4
    var low = (s.k(0.01 + h, nrm) - s.k(0.01 - h, nrm)) / (2.0 * h)
    var mid = (s.k(0.5 + h, nrm) - s.k(0.5 - h, nrm)) / (2.0 * h)
    assert_almost_equal(low, mid, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
