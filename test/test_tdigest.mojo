"""Tests for the core MergingDigest (Phase 2).

Validates the Phase 2 "Done when" criteria from PLAN.md:
  - Accuracy vs. a brute-force exact quantile over N samples from several
    distributions (normal, uniform, exponential, Pareto).
  - Tail error (p0.1 / p99.9) within tight bounds; median within a looser one.
  - Tail quantile error beats a coarse histogram by orders of magnitude.
  - Merge equivalence: digest(A ∪ B) ≈ digest(A).merge_into(digest(B)).
  - Compression bound: centroid count stays O(δ).

Accuracy is measured in *quantile space*: for an estimate v = quantile(q), the
true empirical fraction of samples <= v should be close to q. This |rank error|
is the metric the reference uses and is scale-free across distributions.
"""

from std.math import log, sqrt, cos, pi
from std.random import random_float64, seed
from std.testing import (
    assert_true,
    assert_almost_equal,
    assert_raises,
    TestSuite,
)

from kdigest.tdigest import TDigest
from kdigest.scale import ScaleK1, ScaleK2


comptime N = 50_000
comptime COMPRESSION: Float64 = 100.0


# --- Sample generators -------------------------------------------------------


def _uniform_samples(count: Int) -> List[Float64]:
    var xs = List[Float64]()
    for _ in range(count):
        xs.append(random_float64())
    return xs^


def _normal_samples(count: Int) -> List[Float64]:
    # Box-Muller transform of two uniforms.
    var xs = List[Float64]()
    for _ in range(count):
        var u1 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var u2 = random_float64()
        xs.append(sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2))
    return xs^


def _exponential_samples(count: Int) -> List[Float64]:
    # Inverse-CDF sampling: x = -ln(1 - u), rate 1.
    var xs = List[Float64]()
    for _ in range(count):
        var u = random_float64()
        if u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        xs.append(-log(1.0 - u))
    return xs^


def _pareto_samples(count: Int) -> List[Float64]:
    # Heavy-tailed: x = (1 - u)^(-1/alpha), alpha = 1.5, x_m = 1.
    comptime alpha = 1.5
    var xs = List[Float64]()
    for _ in range(count):
        var u = random_float64()
        if u > 1.0 - 1e-12:
            u = 1.0 - 1e-12
        xs.append((1.0 - u) ** (-1.0 / alpha))
    return xs^


# --- Exact (brute force) reference ------------------------------------------


def _sorted(data: List[Float64]) -> List[Float64]:
    var xs = data.copy()
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key
    return xs^


def _empirical_cdf(sorted_data: List[Float64], value: Float64) -> Float64:
    """Fraction of samples <= value, via binary search on sorted data."""
    var lo = 0
    var hi = len(sorted_data)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_data[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return Float64(lo) / Float64(len(sorted_data))


# --- Accuracy driver ---------------------------------------------------------


def _check_accuracy(
    name: String,
    data: List[Float64],
    tail_tol: Float64,
    median_tol: Float64,
) raises:
    var sorted_data = _sorted(data)

    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for i in range(len(data)):
        d.add(data[i])

    # Rank error at a quantile: |empirical_cdf(quantile(q)) - q|.
    var qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    var tols = [tail_tol, tail_tol, median_tol, tail_tol, tail_tol]
    for i in range(len(qs)):
        var q = qs[i]
        var v = d.quantile(q)
        var got = _empirical_cdf(sorted_data, v)
        var err = abs(got - q)
        assert_true(
            err <= tols[i],
            String("rank error too large for ", name, " q=", q, " err=", err),
        )


# --- Accuracy tests ----------------------------------------------------------


def test_accuracy_uniform() raises:
    seed(1)
    _check_accuracy(
        "uniform", _uniform_samples(N), tail_tol=0.002, median_tol=0.01
    )


def test_accuracy_normal() raises:
    seed(2)
    _check_accuracy(
        "normal", _normal_samples(N), tail_tol=0.002, median_tol=0.01
    )


def test_accuracy_exponential() raises:
    seed(3)
    _check_accuracy(
        "exponential", _exponential_samples(N), tail_tol=0.002, median_tol=0.01
    )


def test_accuracy_pareto() raises:
    seed(4)
    _check_accuracy(
        "pareto", _pareto_samples(N), tail_tol=0.003, median_tol=0.01
    )


# --- Beats a coarse histogram at the tail ------------------------------------


def test_beats_histogram_at_tail() raises:
    # A fixed-width histogram over [0, 1] with B buckets resolves the 99.9th
    # percentile only to +/- 1/B in value. The digest, concentrating centroids
    # at the tail, pins it far more tightly. Compare value error at p99.9.
    seed(5)
    var data = _uniform_samples(N)
    var sorted_data = _sorted(data)
    var true_p999 = sorted_data[Int(0.999 * Float64(N))]

    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for i in range(len(data)):
        d.add(data[i])
    var digest_err = abs(d.quantile(0.999) - true_p999)

    # 100-bucket histogram: best-case value resolution at the tail is one bucket.
    comptime buckets = 100
    var hist_err = 1.0 / Float64(buckets)

    assert_true(
        digest_err * 10.0 < hist_err,
        String(
            "digest tail error ", digest_err, " not << histogram ", hist_err
        ),
    )


# --- Merge equivalence -------------------------------------------------------


def test_merge_equivalence() raises:
    seed(6)
    var a = _normal_samples(N // 2)
    var b = _normal_samples(N // 2)

    # Combined digest over A ∪ B.
    var combined = TDigest[ScaleK2](compression=COMPRESSION)
    for i in range(len(a)):
        combined.add(a[i])
    for i in range(len(b)):
        combined.add(b[i])

    # Two separate digests merged.
    var da = TDigest[ScaleK2](compression=COMPRESSION)
    for i in range(len(a)):
        da.add(a[i])
    var db = TDigest[ScaleK2](compression=COMPRESSION)
    for i in range(len(b)):
        db.add(b[i])
    da.merge_into(db)

    var qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    for i in range(len(qs)):
        var q = qs[i]
        # Both estimate the same underlying distribution; agree closely.
        assert_almost_equal(combined.quantile(q), da.quantile(q), atol=0.05)

    assert_almost_equal(da.total_weight(), Float64(N), atol=1e-9)


# --- Compression bound -------------------------------------------------------


def test_compression_bound() raises:
    seed(7)
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(N):
        d.add(random_float64())
    d.merge()
    # Centroid count must stay O(delta); the reference bound is a small multiple.
    var centroid_budget = Int(6.0 * Float64(COMPRESSION))
    assert_true(
        len(d) <= centroid_budget,
        String("centroid count ", len(d), " exceeds O(delta) bound"),
    )
    assert_true(len(d) > 10, "expected many centroids for large N")


# --- Edge cases & interpolation ---------------------------------------------


def test_single_point() raises:
    var d = TDigest[ScaleK1]()
    d.add(42.0)
    assert_almost_equal(d.quantile(0.0), 42.0)
    assert_almost_equal(d.quantile(0.5), 42.0)
    assert_almost_equal(d.quantile(1.0), 42.0)


def test_quantile_extremes_hit_min_max() raises:
    seed(8)
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(1000):
        d.add(random_float64())
    assert_almost_equal(d.quantile(0.0), d.min(), atol=1e-12)
    assert_almost_equal(d.quantile(1.0), d.max(), atol=1e-12)


def test_quantile_rejects_out_of_range() raises:
    var d = TDigest[ScaleK1]()
    d.add(1.0)
    with assert_raises():
        _ = d.quantile(-0.1)
    with assert_raises():
        _ = d.quantile(1.1)


def test_cdf_monotone_and_bounds() raises:
    seed(9)
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(5000):
        d.add(random_float64())
    assert_almost_equal(d.cdf(d.min() - 1.0), 0.0)
    assert_almost_equal(d.cdf(d.max() + 1.0), 1.0)
    # cdf is non-decreasing across the support.
    var prev = d.cdf(0.0)
    var steps = 50
    for i in range(1, steps):
        var x = Float64(i) / Float64(steps)
        var cur = d.cdf(x)
        assert_true(cur >= prev - 1e-9, "cdf must be non-decreasing")
        prev = cur


def test_cdf_quantile_round_trip() raises:
    seed(10)
    var d = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(N):
        d.add(random_float64())
    # cdf(quantile(q)) ≈ q away from the extremes.
    var qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for i in range(len(qs)):
        var q = qs[i]
        assert_almost_equal(d.cdf(d.quantile(q)), q, atol=0.02)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
