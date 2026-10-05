"""Edge cases and serialization (Phase 4).

Covers the Phase 4 hardening deliverables from PLAN.md:
  - Edge cases: empty digest, single point, all-identical values, NaN/inf
    rejection, q out of [0, 1].
  - Serialization: to_bytes / from_bytes round-trip, including cross-"process"
    merge of a restored digest.
"""

from std.random import random_float64, seed
from std.testing import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    TestSuite,
)

from kdigest.decay import Decay
from kdigest.scale import ScaleK1, ScaleK2
from kdigest.tdigest import TDigest


comptime COMPRESSION: Float64 = 100.0


def _nan() -> Float64:
    var zero = 0.0
    return zero / zero


def _inf() -> Float64:
    var zero = 0.0
    return 1.0 / zero


# --- Empty digest ------------------------------------------------------------


def test_empty_quantile_raises() raises:
    var d = TDigest[ScaleK1]()
    with assert_raises():
        _ = d.quantile(0.5)


def test_empty_cdf_raises() raises:
    var d = TDigest[ScaleK1]()
    with assert_raises():
        _ = d.cdf(0.0)


def test_empty_total_weight_zero() raises:
    var d = TDigest[ScaleK1]()
    assert_almost_equal(d.total_weight(), 0.0)
    assert_equal(len(d), 0)


# --- Single point ------------------------------------------------------------


def test_single_point_all_quantiles() raises:
    var d = TDigest[ScaleK1]()
    d.add(7.0)
    assert_almost_equal(d.quantile(0.0), 7.0)
    assert_almost_equal(d.quantile(0.5), 7.0)
    assert_almost_equal(d.quantile(1.0), 7.0)
    # cdf of a single point: 0 below, 1 above, 0.5 at.
    assert_almost_equal(d.cdf(6.0), 0.0)
    assert_almost_equal(d.cdf(8.0), 1.0)
    assert_almost_equal(d.cdf(7.0), 0.5)


# --- All-identical values ----------------------------------------------------


def test_all_identical_values() raises:
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(10_000):
        d.add(3.0)
    assert_almost_equal(d.quantile(0.0), 3.0)
    assert_almost_equal(d.quantile(0.5), 3.0)
    assert_almost_equal(d.quantile(0.999), 3.0)
    assert_almost_equal(d.quantile(1.0), 3.0)
    assert_almost_equal(d.min(), 3.0)
    assert_almost_equal(d.max(), 3.0)
    # Everything collapses to one centroid.
    assert_equal(len(d), 1)


# --- NaN / inf rejection -----------------------------------------------------


def test_rejects_nan_value() raises:
    var d = TDigest[ScaleK1]()
    with assert_raises():
        d.add(_nan())


def test_rejects_inf_value() raises:
    var d = TDigest[ScaleK1]()
    with assert_raises():
        d.add(_inf())
    with assert_raises():
        d.add(-_inf())


def test_rejects_nonpositive_weight() raises:
    var d = TDigest[ScaleK1]()
    with assert_raises():
        d.add(1.0, weight=0.0)
    with assert_raises():
        d.add(1.0, weight=-2.0)


def test_rejected_sample_leaves_digest_unchanged() raises:
    var d = TDigest[ScaleK1]()
    d.add(5.0)
    var before = d.total_weight()
    with assert_raises():
        d.add(_nan())
    assert_almost_equal(d.total_weight(), before)


# --- q out of range ----------------------------------------------------------


def test_quantile_rejects_out_of_range() raises:
    var d = TDigest[ScaleK1]()
    d.add(1.0)
    with assert_raises():
        _ = d.quantile(-0.001)
    with assert_raises():
        _ = d.quantile(1.001)


# --- Serialization -----------------------------------------------------------


def test_serialization_round_trip() raises:
    seed(1)
    var d = TDigest[ScaleK2](
        compression=COMPRESSION, decay=Decay.half_life(5000.0)
    )
    for _ in range(20_000):
        d.add(random_float64())

    var bytes = d.to_bytes()
    var restored = TDigest[ScaleK2].from_bytes(bytes, scale=ScaleK2())

    assert_equal(len(restored), len(d))
    assert_almost_equal(restored.total_weight(), d.total_weight(), atol=1e-9)
    assert_almost_equal(restored.min(), d.min(), atol=1e-12)
    assert_almost_equal(restored.max(), d.max(), atol=1e-12)

    var qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    for i in range(len(qs)):
        assert_almost_equal(
            restored.quantile(qs[i]), d.quantile(qs[i]), atol=1e-12
        )


def test_serialization_version_guard() raises:
    var bad = List[UInt8]()
    bad.append(UInt8(99))  # wrong version byte
    with assert_raises():
        _ = TDigest[ScaleK1].from_bytes(bad)


def test_serialization_truncated_guard() raises:
    seed(2)
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(1000):
        d.add(random_float64())
    var bytes = d.to_bytes()
    # Chop off the tail so a centroid read runs past the end.
    var truncated = List[UInt8]()
    for i in range(len(bytes) - 16):
        truncated.append(bytes[i])
    with assert_raises():
        _ = TDigest[ScaleK1].from_bytes(truncated)


def test_merge_restored_digest_across_processes() raises:
    # Simulate map-reduce: build two digests, serialize both (as if produced by
    # separate workers), restore, and merge. Result should match a single
    # digest over all the data.
    seed(3)
    var count = 20_000
    var all_data = List[Float64]()
    for _ in range(count):
        all_data.append(random_float64())

    var da = TDigest[ScaleK1](compression=COMPRESSION)
    var db = TDigest[ScaleK1](compression=COMPRESSION)
    for i in range(count):
        if i % 2 == 0:
            da.add(all_data[i])
        else:
            db.add(all_data[i])

    # Round-trip both through bytes before merging.
    var ra = TDigest[ScaleK1].from_bytes(da.to_bytes())
    var rb = TDigest[ScaleK1].from_bytes(db.to_bytes())
    ra.merge_into(rb)

    var combined = TDigest[ScaleK1](compression=COMPRESSION)
    for i in range(count):
        combined.add(all_data[i])

    var qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    for i in range(len(qs)):
        assert_almost_equal(
            ra.quantile(qs[i]), combined.quantile(qs[i]), atol=0.03
        )
    assert_almost_equal(ra.total_weight(), Float64(count), atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
