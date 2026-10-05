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

Gated on `has_accelerator()`: no GPU → no-op pass.
"""

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite

from kdigest import TDigest, ScaleK2, GroupedTDigest


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
