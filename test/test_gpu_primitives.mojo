"""Stage 5b: GPU primitives vs. their CPU references.

Each primitive is checked against a CPU implementation with identical semantics
on a range of lengths, including prime and tile-straddling sizes (13, 63, 64,
65) and the full block capacity. Inputs are generated with a deterministic LCG
so the GPU and CPU paths see byte-identical data.

The whole suite is gated on `has_accelerator()`: with no GPU present every test
is a no-op pass, so CI without an accelerator stays green.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_true,
    assert_almost_equal,
    TestSuite,
)

from kdigest.gpu import (
    BLOCK_CAP,
    gpu_sort_pairs,
    gpu_inclusive_scan,
    gpu_exclusive_scan,
    gpu_segmented_scan,
    gpu_compact,
    cpu_sort_pairs,
    cpu_inclusive_scan,
    cpu_exclusive_scan,
    cpu_segmented_scan,
    cpu_compact,
)


def _lengths() -> List[Int]:
    """Lengths that stress the single-block tiling: empty, singleton, prime,
    and the 63/64/65 boundary, plus full capacity."""
    return [0, 1, 2, 13, 31, 63, 64, 65, 100, 511, 512, 1000, BLOCK_CAP]


# fp32 tree-reductions reorder additions vs. a sequential CPU sum; allow a small
# relative slack on scan outputs. Sort/compaction are exact (reordering only).
comptime _SCAN_RTOL = 1e-5


struct _Rng(Movable):
    """Tiny deterministic LCG so CPU and GPU get identical pseudo-random data.
    """

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u32(mut self) -> UInt32:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return UInt32((self.state >> 33) & 0xFFFFFFFF)

    def unit(mut self) -> Float32:
        # Value in [0, 1).
        return Float32(self.next_u32() % 1_000_000) / Float32(1_000_000)


def _close(a: Float32, b: Float32, rtol: Float32) raises:
    var scale = max(Float32(1.0), max(abs(a), abs(b)))
    assert_true(
        abs(a - b) <= rtol * scale,
        String("expected ", a, " ≈ ", b),
    )


# ----------------------------------------------------------------------
# sort by key
# ----------------------------------------------------------------------


def _check_sort(n: Int) raises:
    var rng = _Rng(0x1234 + UInt64(n))
    var gk = List[Float32]()
    var gv = List[Float32]()
    var ck = List[Float32]()
    var cv = List[Float32]()
    for i in range(n):
        var key = rng.unit() * 1000.0 - 500.0
        gk.append(key)
        ck.append(key)
        # Distinct value payload = original index, to verify it rides along.
        gv.append(Float32(i))
        cv.append(Float32(i))

    gpu_sort_pairs(gk, gv)
    cpu_sort_pairs(ck, cv)

    assert_equal(len(gk), n)
    for i in range(n):
        assert_equal(gk[i], ck[i])
        assert_equal(gv[i], cv[i])
    # Keys must be non-decreasing.
    for i in range(1, n):
        assert_true(gk[i] >= gk[i - 1], "sorted keys must be non-decreasing")


def test_sort_pairs() raises:
    comptime if not has_accelerator():
        return
    var lens = _lengths()
    for i in range(len(lens)):
        _check_sort(lens[i])


def test_cpu_sort_pairs_large_stable() raises:
    # The host sort is on the gpu_compress_batch hot path, so it is a bottom-up
    # merge sort rather than insertion sort. Check it past the insertion-run
    # size and across several merge passes, with heavy ties so stability
    # (equal keys keep input order) is exercised. Runs without a GPU.
    var sizes = [33, 100, 1_000, 4_097, 100_000]
    for si in range(len(sizes)):
        var n = sizes[si]
        var rng = _Rng(0xD00D + UInt64(n))
        var k = List[Float32]()
        var v = List[Float32]()
        for i in range(n):
            # Only 64 distinct keys -> many ties.
            k.append(Float32(Int(rng.unit() * 64.0)))
            v.append(Float32(i))
        cpu_sort_pairs(k, v)
        assert_equal(len(k), n)
        for i in range(1, n):
            assert_true(k[i] >= k[i - 1], "keys must be non-decreasing")
            if k[i] == k[i - 1]:
                assert_true(v[i] > v[i - 1], "ties must keep input order")


# ----------------------------------------------------------------------
# inclusive / exclusive scan
# ----------------------------------------------------------------------


def _check_inclusive(n: Int) raises:
    var rng = _Rng(0x5678 + UInt64(n))
    var g = List[Float32]()
    var c = List[Float32]()
    for _ in range(n):
        var v = rng.unit()
        g.append(v)
        c.append(v)
    gpu_inclusive_scan(g)
    cpu_inclusive_scan(c)
    assert_equal(len(g), n)
    for i in range(n):
        _close(g[i], c[i], _SCAN_RTOL)


def _check_exclusive(n: Int) raises:
    var rng = _Rng(0x9ABC + UInt64(n))
    var g = List[Float32]()
    var c = List[Float32]()
    for _ in range(n):
        var v = rng.unit()
        g.append(v)
        c.append(v)
    gpu_exclusive_scan(g)
    cpu_exclusive_scan(c)
    assert_equal(len(g), n)
    for i in range(n):
        _close(g[i], c[i], _SCAN_RTOL)


def test_inclusive_scan() raises:
    comptime if not has_accelerator():
        return
    var lens = _lengths()
    for i in range(len(lens)):
        _check_inclusive(lens[i])


def test_exclusive_scan() raises:
    comptime if not has_accelerator():
        return
    var lens = _lengths()
    for i in range(len(lens)):
        _check_exclusive(lens[i])


# ----------------------------------------------------------------------
# segmented scan
# ----------------------------------------------------------------------


def _check_segmented(n: Int) raises:
    var rng = _Rng(0xDEF0 + UInt64(n))
    var g = List[Float32]()
    var c = List[Float32]()
    var flags = List[Int32]()
    for i in range(n):
        var v = rng.unit()
        g.append(v)
        c.append(v)
        # First element always a head; then start a new segment ~every 7th,
        # exercising short and long runs within one block.
        var head = Int32(1) if (i == 0 or (rng.next_u32() % 7) == 0) else Int32(
            0
        )
        flags.append(head)
    gpu_segmented_scan(g, flags)
    cpu_segmented_scan(c, flags)
    assert_equal(len(g), n)
    for i in range(n):
        _close(g[i], c[i], _SCAN_RTOL)


def test_segmented_scan() raises:
    comptime if not has_accelerator():
        return
    var lens = _lengths()
    for i in range(len(lens)):
        _check_segmented(lens[i])


def test_segmented_scan_single_segment() raises:
    # A segmented scan with one head at index 0 must equal a plain inclusive
    # scan (sanity bridge between the two primitives).
    comptime if not has_accelerator():
        return
    var g = List[Float32]()
    var expected = List[Float32]()
    var flags = List[Int32]()
    for i in range(50):
        g.append(Float32(i + 1))
        expected.append(Float32(i + 1))
        flags.append(Int32(1) if i == 0 else Int32(0))
    gpu_segmented_scan(g, flags)
    cpu_inclusive_scan(expected)
    for i in range(50):
        _close(g[i], expected[i], _SCAN_RTOL)


# ----------------------------------------------------------------------
# stream compaction
# ----------------------------------------------------------------------


def _check_compact(n: Int) raises:
    var rng = _Rng(0x2468 + UInt64(n))
    var keys = List[Float32]()
    var vals = List[Float32]()
    var flags = List[Int32]()
    for i in range(n):
        keys.append(rng.unit() * 100.0)
        vals.append(Float32(i))
        flags.append(Int32(1) if (rng.next_u32() % 3) != 0 else Int32(0))

    var gpu = gpu_compact(keys, vals, flags)
    var cpu = cpu_compact(keys, vals, flags)
    assert_equal(len(gpu[0]), len(cpu[0]))
    assert_equal(len(gpu[1]), len(cpu[1]))
    for i in range(len(cpu[0])):
        assert_equal(gpu[0][i], cpu[0][i])
        assert_equal(gpu[1][i], cpu[1][i])


def test_compact() raises:
    comptime if not has_accelerator():
        return
    var lens = _lengths()
    for i in range(len(lens)):
        _check_compact(lens[i])


def test_compact_all_and_none() raises:
    comptime if not has_accelerator():
        return
    var keys = List[Float32]()
    var vals = List[Float32]()
    var all_true = List[Int32]()
    var all_false = List[Int32]()
    for i in range(37):
        keys.append(Float32(i) * 2.0)
        vals.append(Float32(i))
        all_true.append(Int32(1))
        all_false.append(Int32(0))

    var kept = gpu_compact(keys, vals, all_true)
    assert_equal(len(kept[0]), 37)
    for i in range(37):
        assert_equal(kept[0][i], keys[i])

    var dropped = gpu_compact(keys, vals, all_false)
    assert_equal(len(dropped[0]), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
