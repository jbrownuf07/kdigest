"""Stage 5f: vectorized query (quantile/cdf) and merge-tree reduction parity.

The GPU query kernels interpolate the *same* compressed digest the CPU does, so
the isolating check is: build one digest, then require `gpu_quantile` /
`gpu_cdf` over an array of queries to match `TDigest.quantile` / `.cdf` called
one query at a time, to fp32 rounding. We build the digest from GPU-compressed
centroids (`gpu_compress_batch`) wrapped via `TDigest._from_centroids` so both
sides see byte-identical centroids, min, and max.

The merge test checks the mergeable property end-to-end: compressing the union
of several partial digests (`gpu_merge_digests`) yields the same quantiles as a
single digest built from the concatenated raw data.

Gated on `has_accelerator()`: no GPU -> no-op pass.
"""

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite

from kdigest import TDigest, ScaleK2
from kdigest.gpu import (
    gpu_compress_batch,
    gpu_quantile,
    gpu_cdf,
    gpu_merge_digests,
)


comptime COMPRESSION = 100.0


struct _Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def unit(mut self) -> Float64:
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        var x = (self.state * 0x2545F4914F6CDD1D) >> 11
        return (Float64(x) + 0.5) / Float64(1 << 53)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _build(
    n: Int, seed: UInt64
) raises -> Tuple[List[Float32], List[Float32], Float32, Float32]:
    """GPU-compress `n` uniform samples; return the centroid arrays and
    min/max. Callers wrap these in a CPU TDigest via `_from_centroids`."""
    var rng = _Rng(seed)
    var raw = List[Float32]()
    var w = List[Float32]()
    var lo = Float32(1e30)
    var hi = Float32(-1e30)
    for _ in range(n):
        var v = Float32(rng.unit() * 10.0 - 5.0)
        raw.append(v)
        w.append(Float32(1))
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    var comp = gpu_compress_batch[ScaleK2.F32](raw, w, Float32(COMPRESSION))
    return (comp[0].copy(), comp[1].copy(), lo, hi)


def _f64(xs: List[Float32]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(xs)):
        out.append(Float64(xs[i]))
    return out^


def test_vectorized_quantile_matches_cpu() raises:
    comptime if not has_accelerator():
        return
    var built = _build(50_000, 0x1234)
    var means = built[0].copy()
    var weights = built[1].copy()
    var lo = built[2]
    var hi = built[3]
    var digest = TDigest[ScaleK2]._from_centroids(
        COMPRESSION, _f64(means), _f64(weights), Float64(lo), Float64(hi)
    )
    var qs = List[Float32]()
    var qd = List[Float64]()
    var grid = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    for i in range(len(grid)):
        qs.append(Float32(grid[i]))
        qd.append(grid[i])

    var gpu = gpu_quantile(means, weights, qs, lo, hi)
    assert_true(len(gpu) == len(qs), "quantile output length")
    for i in range(len(qd)):
        var cpu = digest.quantile(qd[i])
        var g = Float64(gpu[i])
        var scale = max(1.0, _abs(cpu))
        assert_true(
            _abs(g - cpu) <= 1e-4 * scale,
            String("q=", qd[i], " gpu=", g, " cpu=", cpu),
        )


def test_vectorized_cdf_matches_cpu() raises:
    comptime if not has_accelerator():
        return
    var built = _build(50_000, 0x9A5E)
    var means = built[0].copy()
    var weights = built[1].copy()
    var lo = built[2]
    var hi = built[3]
    var digest = TDigest[ScaleK2]._from_centroids(
        COMPRESSION, _f64(means), _f64(weights), Float64(lo), Float64(hi)
    )

    # Probe x across the observed range.
    var xs = List[Float32]()
    var xd = List[Float64]()
    var steps = 25
    for s in range(steps):
        var frac = Float64(s) / Float64(steps - 1)
        var x = Float64(lo) + (Float64(hi) - Float64(lo)) * frac
        xs.append(Float32(x))
        xd.append(x)

    var gpu = gpu_cdf(means, weights, xs, lo, hi)
    assert_true(len(gpu) == len(xs), "cdf output length")
    for i in range(len(xd)):
        var cpu = digest.cdf(xd[i])
        var g = Float64(gpu[i])
        assert_true(
            _abs(g - cpu) <= 1e-4,
            String("x=", xd[i], " gpu_cdf=", g, " cpu_cdf=", cpu),
        )


def test_merge_tree_matches_concatenation() raises:
    comptime if not has_accelerator():
        return
    # Four mappers, each compressing a shard; merge their partials and compare
    # quantiles to a single digest over the concatenated raw data.
    comptime shards = 4
    comptime per_shard = 20_000
    var rng = _Rng(0xC0FFEE)
    var part_m = List[List[Float32]]()
    var part_w = List[List[Float32]]()
    var all_raw = List[Float64]()
    var lo = Float32(1e30)
    var hi = Float32(-1e30)

    for _ in range(shards):
        var raw = List[Float32]()
        var w = List[Float32]()
        for _ in range(per_shard):
            var v = Float32(rng.unit() * rng.unit() * 20.0)  # skewed
            raw.append(v)
            w.append(Float32(1))
            all_raw.append(Float64(v))
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        var comp = gpu_compress_batch[ScaleK2.F32](raw, w, Float32(COMPRESSION))
        part_m.append(comp[0].copy())
        part_w.append(comp[1].copy())

    var merged = gpu_merge_digests[ScaleK2.F32](
        part_m, part_w, Float32(COMPRESSION)
    )
    var mm = List[Float64]()
    var mw = List[Float64]()
    for i in range(len(merged[0])):
        mm.append(Float64(merged[0][i]))
        mw.append(Float64(merged[1][i]))
    var merged_digest = TDigest[ScaleK2]._from_centroids(
        COMPRESSION, mm, mw, Float64(lo), Float64(hi)
    )

    # Reference: one digest over all shards' raw data.
    var reference = TDigest[ScaleK2](compression=COMPRESSION)
    reference.add_batch(all_raw)

    var grid = [0.01, 0.1, 0.5, 0.9, 0.99]
    for i in range(len(grid)):
        var q = grid[i]
        var mq = merged_digest.quantile(q)
        var rq = reference.quantile(q)
        var scale = max(1.0, _abs(rq))
        # Merge-tree (parallel) vs a sequential CPU digest over the raw union is
        # the same valid-but-different clustering seen in 5d; allow a small band.
        assert_true(
            _abs(mq - rq) <= 2e-2 * scale,
            String("merge q=", q, " merged=", mq, " ref=", rq),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
