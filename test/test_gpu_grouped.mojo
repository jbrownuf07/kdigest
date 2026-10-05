"""Stage 5e: GroupedTDigest (one GPU block per key) parity.

Each key's GPU digest is produced by the same parallel k-bucketing compress as
Stage 5d, one block per key in a single launch. As in 5d, the parallel merge is
a different-but-valid clustering than a sequential CPU sweep, so we judge
*accuracy against the true per-key quantiles* rather than exact equality with a
CPU digest: for every key, both a per-key CPU digest and the grouped GPU digest
must land within an accuracy band of that key's true quantiles. The keys carry
different distributions, and we confirm the result is independent of key count.

A mixed-size test exercises both routes at once — small keys (one-block path)
beside one oversized key (> BLOCK_CAP → hierarchical fallback).

Accumulation tests (regression for a bug where each GPU `add_batch` rebuilt
every key from the latest batch alone) feed several batches — keys coming and
going, decay on and off, a key crossing BLOCK_CAP mid-stream — and require each
key to equal, exactly, a reference `TDigest` fed the same per-key batches.

Gated on `has_accelerator()`: no GPU → no-op pass.
"""

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite

from kdigest import TDigest, ScaleK2, GroupedTDigest, Decay, GpuSession
from kdigest.gpu.add_batch import gpu_compress_batch_host_tiled
from kdigest.gpu.grouped import (
    gpu_grouped_compress,
    gpu_grouped_compress_weighted,
    GroupedWorkspace,
)
from max.gpu.host import DeviceContext


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


def _qs() -> List[Float64]:
    return [0.01, 0.5, 0.99]


# Accuracy bands (fraction of the local quantile magnitude) at [p1, p50, p99].
def _bands() -> List[Float64]:
    return [0.15, 0.05, 0.15]


def _true_quantiles(data: List[Float64], qs: List[Float64]) -> List[Float64]:
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


def _check_grouped(num_keys: Int, per_key: Int) raises:
    """Build one grouped digest over `num_keys` keys with distinct
    distributions; require each key's grouped quantiles (and, as a control, the
    per-key CPU digest's quantiles) to be within band of the true quantiles."""
    var rng = _Rng(0x51E5 + UInt64(num_keys))
    var keys = List[Int32]()
    var values = List[Float64]()
    var per_key_vals = List[List[Float64]]()
    for _ in range(num_keys):
        per_key_vals.append(List[Float64]())

    for k in range(num_keys):
        for _ in range(per_key):
            var v = rng.unit() * Float64(k + 1) + Float64(k)
            keys.append(Int32(k))
            values.append(v)
            per_key_vals[k].append(v)

    var grouped = GroupedTDigest[ScaleK2](
        num_keys=num_keys, compression=COMPRESSION
    )
    grouped.add_batch(keys, values)

    var qs = _qs()
    var bands = _bands()
    for k in range(num_keys):
        var cpu = TDigest[ScaleK2](compression=COMPRESSION)
        for j in range(per_key):
            cpu.add(per_key_vals[k][j])
        cpu.merge()

        var truth = _true_quantiles(per_key_vals[k], qs)
        for qi in range(len(qs)):
            var q = qs[qi]
            var tq = truth[qi]
            var gq = grouped.quantile(k, q)
            var cq = cpu.quantile(q)
            var band = bands[qi] * max(1.0, _abs(tq))
            assert_true(
                _abs(gq - tq) <= band,
                String("key ", k, " GPU q=", q, " truth=", tq, " gpu=", gq),
            )
            assert_true(
                _abs(cq - tq) <= band,
                String("key ", k, " CPU q=", q, " truth=", tq, " cpu=", cq),
            )
        # Weight is the per-key sample count.
        assert_true(
            _abs(grouped.total_weight(k) - Float64(per_key)) <= 1.0,
            String("key ", k, " weight"),
        )


def test_grouped_small_keycount() raises:
    comptime if not has_accelerator():
        return
    _check_grouped(num_keys=8, per_key=20_000)


def test_grouped_many_keys() raises:
    # Correctness must be independent of key count: many keys, fewer samples
    # each (each still a healthy single-block digest).
    comptime if not has_accelerator():
        return
    _check_grouped(num_keys=256, per_key=800)


def test_grouped_mixed_sizes() raises:
    # One oversized key (> BLOCK_CAP → hierarchical fallback) alongside small
    # keys (one-block path). Both routes must land within band of truth.
    comptime if not has_accelerator():
        return
    var rng = _Rng(0xBEEF)
    comptime num_keys = 4
    var keys = List[Int32]()
    var values = List[Float64]()
    var per_key_vals = List[List[Float64]]()
    for _ in range(num_keys):
        per_key_vals.append(List[Float64]())

    var sizes = [100_000, 500, 500, 500]
    for k in range(num_keys):
        for _ in range(sizes[k]):
            var v = rng.unit() * Float64(k + 1)
            keys.append(Int32(k))
            values.append(v)
            per_key_vals[k].append(v)

    var grouped = GroupedTDigest[ScaleK2](
        num_keys=num_keys, compression=COMPRESSION
    )
    grouped.add_batch(keys, values)

    var qs = _qs()
    var bands = _bands()
    for k in range(num_keys):
        var truth = _true_quantiles(per_key_vals[k], qs)
        for qi in range(len(qs)):
            var q = qs[qi]
            var tq = truth[qi]
            var gq = grouped.quantile(k, q)
            var band = bands[qi] * max(1.0, _abs(tq))
            assert_true(
                _abs(gq - tq) <= band,
                String("mixed key ", k, " q=", q, " truth=", tq, " gpu=", gq),
            )


def test_grouped_matches_per_key_reference_exactly() raises:
    # Each block sorts its own key in shared memory now (no host sort), and
    # writes compactly at its input offset. Each key's centroids must equal the
    # single-digest reference path run on that key alone, bit for bit — across
    # interleaved keys, empty keys, a key of exactly BLOCK_CAP, tied values,
    # an oversized key, and a reused (shrinking then growing) workspace.
    comptime if not has_accelerator():
        return
    var ctx = DeviceContext()
    var ws = GroupedWorkspace(ctx)
    var cfgs = [(50, 300, 0), (40, 1024, 13), (6, 2500, 0), (300, 1, 0)]
    for c in range(len(cfgs)):
        var nk = cfgs[c][0]
        var per = cfgs[c][1]
        var lev = cfgs[c][2]
        var rng = _Rng(UInt64(0xBEE + c))
        var keys = List[Int32]()
        var vals = List[Float64]()
        for i in range(per):
            for k in range(nk):
                if k % 7 == 3:
                    continue  # empty key
                if k % 5 == 1 and i >= per // 3:
                    continue  # smaller key
                var u = rng.unit()
                keys.append(Int32(k))
                vals.append(
                    Float64(Int(u * Float64(lev))) if lev
                    > 0 else u * Float64(k + 1)
                )
        var res = gpu_grouped_compress[ScaleK2.F32](
            ctx, ws, keys, vals, nk, Float32(COMPRESSION)
        )
        for k in range(nk):
            var km = List[Float32]()
            var kw = List[Float32]()
            for i in range(len(keys)):
                if Int(keys[i]) == k:
                    km.append(Float32(vals[i]))
                    kw.append(Float32(1))
            var tag = String("cfg=", c, " key=", k)
            if len(km) == 0:
                assert_true(len(res.means[k]) == 0, "empty key " + tag)
                continue
            var r = gpu_compress_batch_host_tiled[ScaleK2.F32](
                ctx, km, kw, Float32(COMPRESSION)
            )
            assert_true(len(r[0]) == len(res.means[k]), "count " + tag)
            for j in range(len(r[0])):
                assert_true(r[0][j] == res.means[k][j], "mean " + tag)
                assert_true(r[1][j] == res.weights[k][j], "weight " + tag)


def _check_accumulation(decay: Decay) raises:
    """Several batches into one GroupedTDigest, with keys appearing in some
    batches but not others and one key growing past BLOCK_CAP (prior centroids
    + new samples) so it moves to the oversized path mid-stream. Per key, the
    result must equal a reference TDigest fed the same per-key batches through
    the same GPU batch path, exactly."""
    comptime num_keys = 6
    var g = GroupedTDigest[ScaleK2](num_keys, COMPRESSION, decay)
    var refs = List[TDigest[ScaleK2]]()
    for _ in range(num_keys):
        refs.append(TDigest[ScaleK2](compression=COMPRESSION, decay=decay))
    var session = GpuSession()
    var rng = _Rng(0xACC0)
    for batch in range(5):
        var keys = List[Int32]()
        var values = List[Float64]()
        var per_key = List[List[Float64]](length=num_keys, fill=List[Float64]())
        for i in range(400):
            for k in range(num_keys):
                if k == 1 and batch % 2 == 1:
                    continue  # key 1 skips odd batches
                if k == 2 and batch >= 2:
                    continue  # key 2 stops after batch 1
                if k == 4 and i >= 10:
                    continue  # key 4 stays tiny
                if k == 5 and batch == 3 and i >= 300:
                    continue  # key 5: irregular batch sizes
                # key 3 gets 1,200 samples in batch 4 -> oversized path
                var v = rng.unit() * Float64(k + 1) + Float64(batch)
                keys.append(Int32(k))
                values.append(v)
                per_key[k].append(v)
        if batch == 4:
            for _ in range(800):
                var v = rng.unit() * 4.0 + 4.0
                keys.append(Int32(3))
                values.append(v)
                per_key[3].append(v)
        g.add_batch(keys, values)
        for k in range(num_keys):
            if len(per_key[k]) > 0:
                refs[k]._add_batch_gpu(per_key[k], session)

    var qs = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    for k in range(num_keys):
        var tag = String("key=", k, " decay=", decay.rate_lambda)
        assert_true(
            g.centroid_count(k) == len(refs[k]), "centroid count " + tag
        )
        assert_true(
            g.total_weight(k) == refs[k].total_weight(), "total weight " + tag
        )
        for i in range(len(qs)):
            assert_true(
                g.quantile(k, qs[i]) == refs[k].quantile(qs[i]),
                "quantile " + tag,
            )


def test_grouped_accumulates_across_batches() raises:
    # Regression: the GPU path used to rebuild every key from the latest batch
    # only, dropping earlier batches and wiping keys absent from the batch.
    comptime if not has_accelerator():
        return
    _check_accumulation(Decay.none())


def test_grouped_accumulates_with_decay() raises:
    comptime if not has_accelerator():
        return
    _check_accumulation(Decay.half_life(500.0))


def test_grouped_untouched_key_and_total_weight() raises:
    comptime if not has_accelerator():
        return
    var g = GroupedTDigest[ScaleK2](3, COMPRESSION)
    var k1 = List[Int32]()
    var v1 = List[Float64]()
    for i in range(500):
        for k in range(3):
            k1.append(Int32(k))
            v1.append(Float64(i))
    g.add_batch(k1, v1)
    var before_count = g.centroid_count(2)
    var before_p50 = g.quantile(2, 0.5)
    var k2 = List[Int32]()
    var v2 = List[Float64]()
    for i in range(300):
        k2.append(Int32(0))
        v2.append(Float64(1000 + i))
    g.add_batch(k2, v2)
    assert_true(g.total_weight(0) == 800.0, "key 0 accumulated weight")
    assert_true(g.total_weight(2) == 500.0, "untouched key weight")
    assert_true(g.centroid_count(2) == before_count, "untouched key centroids")
    assert_true(g.quantile(2, 0.5) == before_p50, "untouched key quantile")
    # Key 0 now spans both batches: its median sits at the batch boundary.
    var p50 = g.quantile(0, 0.5)
    assert_true(p50 > 300.0 and p50 < 1100.0, "key 0 median spans batches")


def test_grouped_invalid_batch_leaves_state_unchanged() raises:
    var g = GroupedTDigest[ScaleK2](2, COMPRESSION)
    var k1 = List[Int32]()
    var v1 = List[Float64]()
    for i in range(200):
        k1.append(Int32(i % 2))
        v1.append(Float64(i))
    g.add_batch(k1, v1)
    var w0 = g.total_weight(0)
    var w1 = g.total_weight(1)
    var bad_k = List[Int32]()
    var bad_v = List[Float64]()
    for i in range(100):
        bad_k.append(Int32(i % 2))
        bad_v.append(Float64(i))
    bad_v[57] = Float64.MAX_FINITE * 10.0  # inf, near the end of the batch
    var raised = False
    try:
        g.add_batch(bad_k, bad_v)
    except:
        raised = True
    assert_true(raised, "non-finite value must raise")
    assert_true(g.total_weight(0) == w0, "key 0 unchanged after bad batch")
    assert_true(g.total_weight(1) == w1, "key 1 unchanged after bad batch")


def test_grouped_weighted_matches_single_digest_exactly() raises:
    """`gpu_grouped_compress_weighted` with prior centroids + scaled new
    weights must equal, per key, `gpu_compress_batch` on (prior ++ new) —
    including heavy ties between prior means and new values, and a key pushed
    past BLOCK_CAP by its prior centroids."""
    comptime if not has_accelerator():
        return
    var ctx = DeviceContext()
    var ws = GroupedWorkspace(ctx)
    comptime num_keys = 5
    var rng = _Rng(0xBEEF)
    var prior_m = List[List[Float32]](length=num_keys, fill=List[Float32]())
    var prior_w = List[List[Float32]](length=num_keys, fill=List[Float32]())
    var new_w = List[Float32]()
    var prior_sizes = [0, 40, 200, 900, 7]
    for k in range(num_keys):
        new_w.append(Float32(1.0 + 0.25 * Float64(k)))
        for j in range(prior_sizes[k]):
            # Coarse grid -> many exact ties with the new values below.
            prior_m[k].append(Float32(Int(rng.unit() * 20.0)))
            prior_w[k].append(Float32(1.0 + Float64(j % 5)))
    var keys = List[Int32]()
    var values = List[Float64]()
    var per_key = List[List[Float64]](length=num_keys, fill=List[Float64]())
    for i in range(300):
        for k in range(num_keys):
            if k == 4 and i >= 3:
                continue
            var v = Float64(Int(rng.unit() * 20.0))
            keys.append(Int32(k))
            values.append(v)
            per_key[k].append(v)
    var res = gpu_grouped_compress_weighted[ScaleK2.F32](
        ctx,
        ws,
        keys,
        values,
        new_w,
        prior_m,
        prior_w,
        num_keys,
        Float32(COMPRESSION),
    )
    for k in range(num_keys):
        var m = prior_m[k].copy()
        var w = prior_w[k].copy()
        for j in range(len(per_key[k])):
            m.append(Float32(per_key[k][j]))
            w.append(new_w[k])
        var ref_out = gpu_compress_batch_host_tiled[ScaleK2.F32](
            ctx, m, w, Float32(COMPRESSION)
        )
        assert_true(
            len(res.means[k]) == len(ref_out[0]),
            String("weighted grouped count, key ", k),
        )
        for j in range(len(ref_out[0])):
            assert_true(
                res.means[k][j] == ref_out[0][j]
                and res.weights[k][j] == ref_out[1][j],
                String("weighted grouped centroid, key ", k),
            )


def test_grouped_cpu_fallback_runs() raises:
    # The CPU fallback path must produce sane per-key quantiles. Small data so
    # this is cheap and stays below the GPU crossover inside add().
    var keys = List[Int32]()
    var values = List[Float64]()
    for k in range(3):
        for i in range(100):
            keys.append(Int32(k))
            values.append(Float64(i) + Float64(k) * 100.0)
    var g = GroupedTDigest[ScaleK2](num_keys=3, compression=COMPRESSION)
    g.add_batch(keys, values)
    assert_true(_abs(g.quantile(0, 0.5) - 49.5) <= 5.0, "key0 median")
    assert_true(_abs(g.quantile(1, 0.5) - 149.5) <= 5.0, "key1 median")
    assert_true(_abs(g.quantile(2, 0.5) - 249.5) <= 5.0, "key2 median")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
