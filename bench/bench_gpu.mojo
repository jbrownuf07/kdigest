"""GPU crossover benchmarks for kdigest (Phase 5, Stage 5f).

Separate from the CPU `bench.mojo` because it requires an accelerator and the
`max` package. Measures *where the GPU wins*:

  1. Batched single-digest: ns/point to fold a batch into one digest, GPU
     (`gpu_compress_batch`) vs CPU (`TDigest.add` loop), swept over batch size.
     Also reports p50/p99 rank error for both so speed is read alongside
     accuracy.
  2. Grouped multi-digest: ns/digest to compress K independent per-key digests,
     GPU (`gpu_grouped_compress`, one block per key, single launch) vs CPU
     (per-key `add` loop), swept over key count.

Run with:
    pixi run bench-gpu

Numbers are hardware-dependent and not an SLA; they identify the break-even
points documented in bench/Benchmark.md.
"""

from std.math import abs
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from kdigest.scale import ScaleK2
from kdigest.tdigest import TDigest
from kdigest.gpu.add_batch import (
    gpu_compress_batch,
    gpu_compress_batch_host_sort,
    gpu_compress_batch_host_tiled,
)
from kdigest.gpu.grouped import gpu_grouped_compress


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


def _sorted(data: List[Float64]) -> List[Float64]:
    # Reference ordering for the exact quantiles (outside the timed regions).
    var xs = data.copy()
    sort(xs)
    return xs^


def _rank_err(
    sorted_data: List[Float64], estimate: Float64, q: Float64
) -> Float64:
    var lo = 0
    var hi = len(sorted_data)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_data[mid] <= estimate:
            lo = mid + 1
        else:
            hi = mid
    return abs(Float64(lo) / Float64(len(sorted_data)) - q)


# --- 1. Batched single-digest crossover --------------------------------------


comptime _REPS = 3  # best-of-N timing per row, to suppress warm-up noise


def _bench_batch_one(ctx: DeviceContext, n: Int) raises:
    var rng = _Rng(0xA11CE + UInt64(n))
    var raw32 = List[Float32]()
    var w32 = List[Float32]()
    var raw64 = List[Float64]()
    var lo = Float32(1e30)
    var hi = Float32(-1e30)
    for _ in range(n):
        var v = rng.unit()
        raw32.append(Float32(v))
        w32.append(Float32(1))
        raw64.append(v)
        if Float32(v) < lo:
            lo = Float32(v)
        if Float32(v) > hi:
            hi = Float32(v)
    var sorted_data = _sorted(raw64)

    var ref_ns = 0
    var a_ns = 0
    var b_ns = 0
    var c_ns = 0
    var comp = (List[Float32](), List[Float32]())
    var cpu = TDigest[ScaleK2](compression=COMPRESSION)
    for rep in range(_REPS):
        # Reference: host sort + per-tile round trips (pre-step-3).
        var t0 = perf_counter_ns()
        _ = gpu_compress_batch_host_tiled[ScaleK2.F32](
            ctx, raw32, w32, Float32(COMPRESSION)
        )
        var t1 = perf_counter_ns()
        # Step 3 part A: host sort + device-resident tiling.
        _ = gpu_compress_batch_host_sort[ScaleK2.F32](
            ctx, raw32, w32, Float32(COMPRESSION)
        )
        var t2 = perf_counter_ns()
        # Step 3 part B: GPU sort + device-resident tiling (the default).
        comp = gpu_compress_batch[ScaleK2.F32](
            ctx, raw32, w32, Float32(COMPRESSION)
        )
        var t3 = perf_counter_ns()
        # CPU add loop + merge.
        var d = TDigest[ScaleK2](compression=COMPRESSION)
        var t4 = perf_counter_ns()
        for i in range(n):
            d.add(raw64[i])
        d.merge()
        var t5 = perf_counter_ns()
        if rep == 0 or t1 - t0 < ref_ns:
            ref_ns = t1 - t0
        if rep == 0 or t2 - t1 < a_ns:
            a_ns = t2 - t1
        if rep == 0 or t3 - t2 < b_ns:
            b_ns = t3 - t2
        if rep == 0 or t5 - t4 < c_ns:
            c_ns = t5 - t4
        cpu = d^

    # Accuracy for the default GPU path vs CPU (p50, p99 rank error vs exact).
    var dm = List[Float64]()
    var dw = List[Float64]()
    for i in range(len(comp[0])):
        dm.append(Float64(comp[0][i]))
        dw.append(Float64(comp[1][i]))
    var gdig = TDigest[ScaleK2]._from_centroids(
        COMPRESSION, dm, dw, Float64(lo), Float64(hi)
    )
    var g_p50 = _rank_err(sorted_data, gdig.quantile(0.5), 0.5)
    var g_p99 = _rank_err(sorted_data, gdig.quantile(0.99), 0.99)
    var c_p50 = _rank_err(sorted_data, cpu.quantile(0.5), 0.5)
    var c_p99 = _rank_err(sorted_data, cpu.quantile(0.99), 0.99)

    var nn = Float64(n)
    print(
        "   N=",
        n,
        ":  ref ",
        Float64(ref_ns) / nn,
        "  A ",
        Float64(a_ns) / nn,
        "  B ",
        Float64(b_ns) / nn,
        "  CPU ",
        Float64(c_ns) / nn,
        " ns/pt   B speedup vs CPU ",
        Float64(c_ns) / Float64(b_ns),
        "x  | p50 err G/C ",
        g_p50,
        "/",
        c_p50,
        "  p99 err G/C ",
        g_p99,
        "/",
        c_p99,
    )


def _bench_batch_crossover(ctx: DeviceContext) raises:
    print("Batched single-digest: GPU gpu_compress_batch vs CPU add-loop")
    print("   ref = host sort + per-tile round trips (before step 3)")
    print("   A   = host sort + device-resident tiling (step 3 part A)")
    print(
        "   B   = GPU sort  + device-resident tiling (step 3 part B, default)"
    )
    print("   (ns per point, best of 3; GPU wins where speedup > 1x)")
    _bench_batch_one(ctx, 1_000)
    _bench_batch_one(ctx, 10_000)
    _bench_batch_one(ctx, 50_000)
    _bench_batch_one(ctx, 100_000)
    _bench_batch_one(ctx, 200_000)
    _bench_batch_one(ctx, 300_000)
    _bench_batch_one(ctx, 500_000)
    _bench_batch_one(ctx, 1_000_000)
    _bench_batch_one(ctx, 2_000_000)
    _bench_batch_one(ctx, 4_000_000)


# --- 2. Grouped multi-digest crossover ---------------------------------------


def _bench_grouped_one(ctx: DeviceContext, num_keys: Int, per_key: Int) raises:
    var rng = _Rng(0x6400 + UInt64(num_keys))
    var keys = List[Int32]()
    var values = List[Float64]()
    for k in range(num_keys):
        for _ in range(per_key):
            keys.append(Int32(k))
            values.append(rng.unit() * Float64(k + 1))

    # GPU: one launch, one block per key.
    var g_start = perf_counter_ns()
    var _res = gpu_grouped_compress[ScaleK2.F32](
        ctx, keys, values, num_keys, Float32(COMPRESSION)
    )
    var g_elapsed = perf_counter_ns() - g_start

    # CPU: per-key add loop.
    var c_start = perf_counter_ns()
    var digests = List[TDigest[ScaleK2]]()
    for _ in range(num_keys):
        digests.append(TDigest[ScaleK2](compression=COMPRESSION))
    for i in range(len(values)):
        digests[Int(keys[i])].add(values[i])
    for k in range(num_keys):
        digests[k].merge()
    var c_elapsed = perf_counter_ns() - c_start

    var g_per = Float64(g_elapsed) / Float64(num_keys)
    var c_per = Float64(c_elapsed) / Float64(num_keys)
    var speedup = c_per / g_per
    print(
        "   keys=",
        num_keys,
        " x ",
        per_key,
        "/key:  GPU ",
        g_per,
        " ns/digest   CPU ",
        c_per,
        " ns/digest   speedup ",
        speedup,
        "x",
    )


def _bench_grouped_crossover(ctx: DeviceContext) raises:
    print(
        "\nGrouped multi-digest: GPU gpu_grouped_compress vs CPU per-key loop"
    )
    print("   (ns per digest; one GPU block per key, single launch)")
    print("   one-block path (samples/key <= BLOCK_CAP = 1024):")
    _bench_grouped_one(ctx, 256, 1_000)
    _bench_grouped_one(ctx, 1_024, 1_000)
    _bench_grouped_one(ctx, 4_096, 1_000)
    _bench_grouped_one(ctx, 4_096, 500)
    _bench_grouped_one(ctx, 16_384, 500)
    _bench_grouped_one(ctx, 16_384, 100)
    _bench_grouped_one(ctx, 65_536, 100)
    print("   oversized-key fallback (samples/key > BLOCK_CAP):")
    _bench_grouped_one(ctx, 64, 2_000)
    _bench_grouped_one(ctx, 256, 2_000)


def main() raises:
    print("=== kdigest GPU crossover benchmarks ===\n")
    if not has_accelerator():
        print("No accelerator present; skipping GPU benchmarks.")
        return
    # One device context for the whole run, reused by every call.
    var ctx = DeviceContext()
    # Warm-up: the first launch of each kernel pays one-time module load cost;
    # keep it out of the timed rows.
    var wm = List[Float32]()
    var ww = List[Float32]()
    for i in range(4_096):
        wm.append(Float32(i))
        ww.append(Float32(1))
    _ = gpu_compress_batch[ScaleK2.F32](ctx, wm, ww, Float32(COMPRESSION))
    var wk = List[Int32]()
    var wv = List[Float64]()
    for i in range(64):
        wk.append(Int32(i % 4))
        wv.append(Float64(i))
    _ = gpu_grouped_compress[ScaleK2.F32](ctx, wk, wv, 4, Float32(COMPRESSION))

    _bench_batch_crossover(ctx)
    _bench_grouped_crossover(ctx)
