"""Tests for the scale functions K0..K3.

Validates the three invariants from PLAN.md Phase 1:
  1. Round-trip identity: q(k(q)) == q
  2. Monotonicity: k is strictly increasing in q
  3. Tail steepness: dk/dq is larger near the tails than in the middle
     (this is the property that drives extreme-quantile accuracy).
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from kdigest.scale import (
    ScaleFunction,
    ScaleK0,
    ScaleK1,
    ScaleK2,
    ScaleK3,
    ScaleK0Impl,
    ScaleK1Impl,
    ScaleK2Impl,
    ScaleK3Impl,
)


comptime COMPRESSION = 200.0
comptime N = 100_000.0


def check_round_trip[S: ScaleFunction](scale: S, atol: Float64) raises:
    comptime dt = S.DT
    var nrm = scale.normalizer(Scalar[dt](COMPRESSION), Scalar[dt](N))
    var qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    for i in range(len(qs)):
        var q = Scalar[dt](qs[i])
        var round = scale.q(scale.k(q, nrm), nrm)
        assert_almost_equal(Float64(round), qs[i], atol=atol)


def check_monotonic[S: ScaleFunction](scale: S) raises:
    comptime dt = S.DT
    var nrm = scale.normalizer(Scalar[dt](COMPRESSION), Scalar[dt](N))
    var prev = scale.k(Scalar[dt](1e-6), nrm)
    var steps = 200
    for i in range(1, steps):
        var q = Scalar[dt](Float64(i) / Float64(steps))
        var cur = scale.k(q, nrm)
        assert_true(cur > prev, "k must strictly increase in q")
        prev = cur


def check_tail_steepness[S: ScaleFunction](scale: S) raises:
    # Compare dk/dq near the lower tail, middle, and upper tail using a small
    # symmetric finite difference. For tail-controlling scale functions the
    # slope at the extremes must exceed the slope at the median.
    comptime dt = S.DT
    var nrm = scale.normalizer(Scalar[dt](COMPRESSION), Scalar[dt](N))
    var h = Scalar[dt](1e-4)

    var low = (scale.k(0.01 + h, nrm) - scale.k(0.01 - h, nrm)) / (2.0 * h)
    var mid = (scale.k(0.5 + h, nrm) - scale.k(0.5 - h, nrm)) / (2.0 * h)
    var high = (scale.k(0.99 + h, nrm) - scale.k(0.99 - h, nrm)) / (2.0 * h)
    assert_true(low > mid, "lower-tail slope must exceed median slope")
    assert_true(high > mid, "upper-tail slope must exceed median slope")


# --- Round-trip (all four) ---


def test_round_trip_k0() raises:
    check_round_trip(ScaleK0(), atol=1e-6)


def test_round_trip_k1() raises:
    check_round_trip(ScaleK1(), atol=1e-6)


def test_round_trip_k2() raises:
    check_round_trip(ScaleK2(), atol=1e-6)


def test_round_trip_k3() raises:
    check_round_trip(ScaleK3(), atol=1e-6)


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


# --- Float32 parity (Stage 5a) -----------------------------------------------
# The DType-generic scale functions must satisfy the same identities at
# float32, within a looser tolerance set by fp32's ~1e-7 machine epsilon.

comptime F32_ATOL = 1e-4


def test_round_trip_k0_f32() raises:
    check_round_trip(ScaleK0Impl[DType.float32](), atol=F32_ATOL)


def test_round_trip_k1_f32() raises:
    check_round_trip(ScaleK1Impl[DType.float32](), atol=F32_ATOL)


def test_round_trip_k2_f32() raises:
    check_round_trip(ScaleK2Impl[DType.float32](), atol=F32_ATOL)


def test_round_trip_k3_f32() raises:
    check_round_trip(ScaleK3Impl[DType.float32](), atol=F32_ATOL)


def test_monotonic_k1_f32() raises:
    check_monotonic(ScaleK1Impl[DType.float32]())


def test_monotonic_k2_f32() raises:
    check_monotonic(ScaleK2Impl[DType.float32]())


def test_monotonic_k3_f32() raises:
    check_monotonic(ScaleK3Impl[DType.float32]())


def test_tail_steepness_k1_f32() raises:
    check_tail_steepness(ScaleK1Impl[DType.float32]())


def test_tail_steepness_k2_f32() raises:
    check_tail_steepness(ScaleK2Impl[DType.float32]())


def test_f32_matches_f64_k1() raises:
    # The float32 forward map agrees with float64 within fp32 tolerance across
    # the quantile range (a direct cross-precision parity check).
    var s64 = ScaleK1()
    var s32 = ScaleK1Impl[DType.float32]()
    var n64 = s64.normalizer(COMPRESSION, N)
    var n32 = s32.normalizer(Float32(COMPRESSION), Float32(N))
    var qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    for i in range(len(qs)):
        var k64 = s64.k(qs[i], n64)
        var k32 = s32.k(Float32(qs[i]), n32)
        assert_almost_equal(Float64(k32), k64, atol=F32_ATOL)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
