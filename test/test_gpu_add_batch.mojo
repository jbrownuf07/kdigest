"""Stage 5d: GPU `add_batch` accuracy vs. the true quantiles.

The GPU path uses a parallel k-bucketing merge — a different but equally valid
t-digest clustering — so it neither reproduces the sequential CPU centroids bit
for bit nor matches the CPU estimate exactly. A t-digest is an *approximation*,
so the fair test is not "GPU == CPU" but "is the GPU estimate as accurate as a
valid t-digest should be?" These tests therefore compare **both** the CPU and
GPU estimates against the *true* empirical quantile (computed from the sorted raw
samples) at p0.1 / p50 / p99.9, and require each to fall inside an accuracy band.

The band is per-rank: central quantiles are tight; the extreme tail is looser,
and widens for the weak-tail scales. This documents a real property of the
parallel merge — on pathologically heavy tails (Pareto) with a weak-tail scale
(K1) the parallel formulation resolves p99.9 more loosely than the sequential
sweep. The strong-tail scales (K2/K3), which you'd actually use for extreme
tails, stay tight everywhere.

Both digests see identical inputs (deterministic LCG). Bands are relative to the
local quantile magnitude, so one policy works across uniform / normal /
exponential / Pareto. Gated on `has_accelerator()`: no GPU → no-op pass.
"""

from std.math import log, sqrt, cos, pi
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite

from kdigest import TDigest, ScaleK0, ScaleK1, ScaleK2, ScaleK3, Decay
from kdigest.scale import ScaleFunction


comptime N = 200_000
comptime COMPRESSION = 100.0


# Rank points the plan calls out.
def _qs() -> List[Float64]:
    return [0.001, 0.5, 0.999]


# Per-rank accuracy band (relative to the local quantile magnitude), indexed to
# match `_qs()` = [p0.1, p50, p99.9]. The extreme ranks are intrinsically harder
# and widen for weak-tail scales. K0 is linear ("weakest tails" by design), so
# it gets wide bands at both ends; K1 moderate; K2/K3 (strong tails) stay tight.
# These are the empirically observed envelopes at compression=100 over 200k
# samples, applied to BOTH the CPU and GPU estimates vs. the true quantile.
def _bands(low_tail: Float64, high_tail: Float64) -> List[Float64]:
    # [p0.1, p50, p99.9]
    return [low_tail, 0.05, high_tail]


struct _Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def unit(mut self) -> Float64:
        # xorshift64* → (0, 1)
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        var x = (self.state * 0x2545F4914F6CDD1D) >> 11
        return (Float64(x) + 0.5) / Float64(1 << 53)

    def normal(mut self) -> Float64:
        var u1 = self.unit()
        var u2 = self.unit()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)

    def expo(mut self) -> Float64:
        return -log(self.unit())

    def pareto(mut self, alpha: Float64) -> Float64:
        # inverse-CDF: x = (1-u)^(-1/alpha)
        return (self.unit()) ** (-1.0 / alpha)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _true_quantiles(data: List[Float64], qs: List[Float64]) -> List[Float64]:
    """Exact empirical quantiles from a copy of the data, sorted ascending."""
    var s = data.copy()
    sort(s)
    var out = List[Float64]()
    var n = len(s)
    for i in range(len(qs)):
        var idx = Int(qs[i] * Float64(n))
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        out.append(s[idx])
    return out^


# Each distribution is produced into a List so CPU and GPU get identical data.
def _uniform(n: Int, seed: UInt64) -> List[Float64]:
    var rng = _Rng(seed)
    var out = List[Float64]()
    for _ in range(n):
        out.append(rng.unit())
    return out^


def _normal(n: Int, seed: UInt64) -> List[Float64]:
    var rng = _Rng(seed)
    var out = List[Float64]()
    for _ in range(n):
        out.append(rng.normal())
    return out^


def _exponential(n: Int, seed: UInt64) -> List[Float64]:
    var rng = _Rng(seed)
    var out = List[Float64]()
    for _ in range(n):
        out.append(rng.expo())
    return out^


def _pareto(n: Int, seed: UInt64) -> List[Float64]:
    var rng = _Rng(seed)
    var out = List[Float64]()
    for _ in range(n):
        out.append(rng.pareto(2.5))
    return out^


def _compare[
    S: ScaleFunction
](
    data: List[Float64],
    label: String,
    low_tail: Float64,
    high_tail: Float64,
) raises where (S.DT == DType.float64 and S.F32.DT == DType.float32):
    var cpu = TDigest[S](compression=COMPRESSION)
    for i in range(len(data)):
        cpu.add(data[i])
    cpu.merge()

    var gpu = TDigest[S](compression=COMPRESSION)
    gpu.add_batch(data)

    var qs = _qs()
    var truth = _true_quantiles(data, qs)
    var bands = _bands(low_tail, high_tail)

    for i in range(len(qs)):
        var q = qs[i]
        var tq = truth[i]
        var cq = cpu.quantile(q)
        var gq = gpu.quantile(q)
        # Band is relative to the local quantile magnitude (robust across
        # heavy/light tails). Require BOTH estimators to be within it: the GPU
        # must be as accurate as a valid t-digest, judged against truth.
        var band = bands[i] * max(1.0, _abs(tq))
        assert_true(
            _abs(cq - tq) <= band,
            String(label, " CPU q=", q, " truth=", tq, " cpu=", cq),
        )
        assert_true(
            _abs(gq - tq) <= band,
            String(label, " GPU q=", q, " truth=", tq, " gpu=", gq),
        )

    # Centroid-count sanity: GPU within a 2x band of the CPU count.
    var cc = len(cpu)
    var gc = len(gpu)
    assert_true(
        gc * 2 >= cc and cc * 2 >= gc,
        String(label, " centroid count off: cpu=", cc, " gpu=", gc),
    )

    # Total weight is conserved exactly (no decay here).
    assert_true(
        _abs(cpu.total_weight() - gpu.total_weight()) <= 1.0,
        String(
            label,
            " weight mismatch: cpu=",
            cpu.total_weight(),
            " gpu=",
            gpu.total_weight(),
        ),
    )


# Per-scale tail-band widths (fraction of the local quantile value) at the
# [p0.1, p99.9] extremes. K0 is linear with the weakest tails by design — it is
# not meant for extreme quantiles (test/test_scale.mojo likewise excludes K0
# from tail-steepness checks), so its tail bands are wide "sanity only" bounds.
# K1 moderate; K2/K3 strong tails stay tight. The SAME band is applied to both
# the CPU and GPU estimates vs. truth, so the GPU is held to the CPU's standard.
comptime _LOW_K0 = 0.30
comptime _HIGH_K0 = 0.10
comptime _LOW_K1 = 0.30
comptime _HIGH_K1 = 0.40
comptime _LOW_K2 = 0.25
comptime _HIGH_K2 = 0.20
comptime _LOW_K3 = 0.25
comptime _HIGH_K3 = 0.20


def _all_distributions[
    S: ScaleFunction
](low_tail: Float64, high_tail: Float64) raises where (
    S.DT == DType.float64 and S.F32.DT == DType.float32
):
    _compare[S](_uniform(N, 0xA1), "uniform", low_tail, high_tail)
    _compare[S](_normal(N, 0xB2), "normal", low_tail, high_tail)
    _compare[S](_exponential(N, 0xC3), "exponential", low_tail, high_tail)
    _compare[S](_pareto(N, 0xD4), "pareto", low_tail, high_tail)


def test_add_batch_k0() raises:
    # K0 is the linear/uniform scale: it is only meant to be accurate on
    # roughly-uniform data, where every rank (incl. the tails) is well resolved.
    # On heavy-tailed data K0's *extreme* quantiles are meaningless *by design*
    # (even the CPU K0 digest is ~8x off at Pareto p99.9), so we exercise K0 on
    # the uniform distribution across all ranks. K1-K3 cover the hard tails.
    comptime if not has_accelerator():
        return
    _compare[ScaleK0](_uniform(N, 0xA1), "uniform", _LOW_K0, _HIGH_K0)


def test_add_batch_k1() raises:
    comptime if not has_accelerator():
        return
    _all_distributions[ScaleK1](_LOW_K1, _HIGH_K1)


def test_add_batch_k2() raises:
    comptime if not has_accelerator():
        return
    _all_distributions[ScaleK2](_LOW_K2, _HIGH_K2)


def test_add_batch_k3() raises:
    comptime if not has_accelerator():
        return
    _all_distributions[ScaleK3](_LOW_K3, _HIGH_K3)


def test_add_batch_decayed() raises:
    # With active decay, add_batch must still match an add-loop on the same
    # data (same clock advance semantics).
    comptime if not has_accelerator():
        return

    var data = _normal(N, 0xE5)
    var cpu = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(Float64(N) / 4.0)
    )
    for i in range(len(data)):
        cpu.add(data[i])
    cpu.merge()
    var gpu = TDigest[ScaleK1](
        compression=COMPRESSION, decay=Decay.half_life(Float64(N) / 4.0)
    )
    gpu.add_batch(data)

    var spread = cpu.quantile(0.99) - cpu.quantile(0.01)
    if spread <= 0.0:
        spread = 1.0
    var qs = _qs()
    for i in range(len(qs)):
        var q = qs[i]
        var cq = cpu.quantile(q)
        var gq = gpu.quantile(q)
        var local_atol = 0.08 * max(spread, _abs(cq))
        assert_true(
            _abs(cq - gq) <= local_atol,
            String(
                "decayed q=",
                q,
                ": cpu=",
                cq,
                " gpu=",
                gq,
            ),
        )


def test_add_batch_below_crossover_matches_add() raises:
    # Below the crossover add_batch runs the CPU path, so it must equal an
    # explicit add loop *exactly*.
    var data = _uniform(1000, 0xF6)
    var looped = TDigest[ScaleK1](compression=COMPRESSION)
    for i in range(len(data)):
        looped.add(data[i])
    looped.merge()
    var batched = TDigest[ScaleK1](compression=COMPRESSION)
    batched.add_batch(data)
    assert_true(len(looped) == len(batched), "CPU-path centroid count")
    var qs = _qs()
    for i in range(len(qs)):
        var q = qs[i]
        assert_true(
            looped.quantile(q) == batched.quantile(q),
            "CPU-path quantiles must be identical",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
