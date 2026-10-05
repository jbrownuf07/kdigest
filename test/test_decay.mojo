"""Tests for exponential decay (Phase 3).

Validates the Phase 3 "Done when" criteria from PLAN.md:
  - Drift test: a decayed digest tracks the recent distribution after the
    stream switches regimes; an undecayed digest stays near the blend.
  - Half-life correctness: after `h` ticks with no new mass in a region, that
    region's contribution to the effective total has halved.
  - Memory bound: centroid count stays bounded under indefinite streaming with
    pruning enabled.
Plus supporting checks: Decay constructors, and that λ = 0 matches Phase 2.
"""

from std.math import log, sqrt, cos, pi, exp
from std.random import random_float64, seed
from std.testing import (
    assert_true,
    assert_almost_equal,
    assert_raises,
    TestSuite,
)

from kdigest.decay import Decay
from kdigest.scale import ScaleK1
from kdigest.tdigest import TDigest


comptime COMPRESSION: Float64 = 100.0


def _normal(mean: Float64) -> Float64:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    var u2 = random_float64()
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


# --- Decay policy constructors ----------------------------------------------


def test_decay_none_is_inactive() raises:
    var d = Decay.none()
    assert_true(not d.is_active())
    assert_almost_equal(d.rate_lambda, 0.0)


def test_decay_rate() raises:
    var d = Decay.rate(0.5)
    assert_true(d.is_active())
    assert_almost_equal(d.rate_lambda, 0.5)


def test_decay_half_life_lambda() raises:
    # λ = ln 2 / h.
    var d = Decay.half_life(10.0)
    assert_almost_equal(d.rate_lambda, log(2.0) / 10.0, atol=1e-12)


def test_decay_half_life_rejects_nonpositive() raises:
    with assert_raises():
        _ = Decay.half_life(0.0)
    with assert_raises():
        _ = Decay.half_life(-1.0)


# --- Drift test --------------------------------------------------------------


def test_drift_tracks_recent_distribution() raises:
    # Stream N(0,1), then switch to N(10,1). A short half-life forgets the old
    # regime; no decay stays near the historical blend.
    comptime phase = 20000

    seed(1)
    var decayed = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(2000.0)
    )
    for _ in range(phase):
        decayed.add(_normal(0.0))
    assert_true(abs(decayed.quantile(0.5)) < 0.5, "phase-1 median near 0")
    for _ in range(phase):
        decayed.add(_normal(10.0))
    # Should have converged toward the recent mean of 10.
    assert_true(
        abs(decayed.quantile(0.5) - 10.0) < 1.0,
        String("decayed median should track 10, got ", decayed.quantile(0.5)),
    )

    seed(1)
    var plain = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(phase):
        plain.add(_normal(0.0))
    for _ in range(phase):
        plain.add(_normal(10.0))
    # No decay: median sits in the blend, far from the recent mean.
    assert_true(
        abs(plain.quantile(0.5) - 10.0) > 2.0,
        String(
            "undecayed median should stay in the blend, got ",
            plain.quantile(0.5),
        ),
    )


# --- Half-life correctness ---------------------------------------------------


def test_half_life_total_weight_halves() raises:
    # With steady unit input at rate 1 (one tick per add) and half-life h, the
    # effective total converges to the geometric-series steady state
    #     Σ_{k>=0} e^{-λk} = 1 / (1 - e^{-λ}),   λ = ln 2 / h.
    # Run many half-lives so the cold start has washed out, then check we are
    # within a few percent of that closed form. This directly validates the
    # per-tick decay factor (and hence the half-life).
    comptime h_ticks = 1000
    var d = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(Float64(h_ticks))
    )
    # ~12 half-lives: e^{-12 ln2} ≈ 2e-4, well converged.
    for _ in range(12 * h_ticks):
        d.add(_normal(0.0))
    var total = d.total_weight()

    var lam = log(2.0) / Float64(h_ticks)
    var steady = 1.0 / (1.0 - exp(-lam))
    assert_true(
        abs(total - steady) / steady < 0.05,
        String(
            "effective total ",
            total,
            " should approach steady state ",
            steady,
        ),
    )


def test_half_life_old_region_contribution_halves() raises:
    # Direct half-life check via timestamps: a seed observed at t=0 should have
    # exactly half the effective weight after one half-life of elapsed time,
    # regardless of what else arrives. We isolate the seed's contribution by
    # differencing two identical digests — one with the seed, one without —
    # both advanced to the same final time with the same later samples.
    comptime h_ticks = 2000
    var with_seed = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(Float64(h_ticks))
    )
    var without_seed = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(Float64(h_ticks))
    )

    # Seed one far-away unit at t=0 in only the first digest.
    with_seed.add_at(1000.0, timestamp=0.0)

    # Both digests receive the same fresh sample at t = h (one half-life later).
    with_seed.add_at(0.0, timestamp=Float64(h_ticks))
    without_seed.add_at(0.0, timestamp=Float64(h_ticks))

    # The only difference is the seed, now decayed by e^{-λ·h} = 1/2.
    var seed_effective = with_seed.total_weight() - without_seed.total_weight()
    assert_almost_equal(seed_effective, 0.5, atol=1e-6)


# --- Memory bound ------------------------------------------------------------


def test_memory_bounded_under_indefinite_stream() raises:
    seed(3)
    var d = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(5000.0)
    )
    # Long run; pruning + renormalization must keep centroid count bounded.
    for _ in range(300_000):
        d.add(_normal(0.0))
    d.merge()
    assert_true(
        len(d) <= Int(6.0 * COMPRESSION),
        String("centroid count ", len(d), " exceeds O(delta) bound"),
    )


# --- λ = 0 equals Phase 2 ----------------------------------------------------


def test_zero_decay_matches_undecayed() raises:
    seed(4)
    var decayed = TDigest[ScaleK1](compression=COMPRESSION, decay=Decay.none())
    seed(4)
    var plain = TDigest[ScaleK1](compression=COMPRESSION)
    for _ in range(20000):
        var x = random_float64()
        decayed.add(x)
        plain.add(x)
    var qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    for i in range(len(qs)):
        assert_almost_equal(
            decayed.quantile(qs[i]), plain.quantile(qs[i]), atol=1e-12
        )


# --- Timestamped decay -------------------------------------------------------


def test_add_at_time_based_decay() raises:
    # Two points one half-life apart in time; the older should weigh half the
    # newer. Query the effective total after both.
    comptime h = 100.0
    var d = TDigest[ScaleK1](compression=COMPRESSION, decay=Decay.half_life(h))
    d.add_at(0.0, timestamp=0.0)
    d.add_at(0.0, timestamp=h)  # one half-life later
    # Effective total = older·(1/2) + newer·1 = 1.5.
    assert_almost_equal(d.total_weight(), 1.5, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
