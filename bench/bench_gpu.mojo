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

from kdigest.scale import ScaleK2
from kdigest.tdigest import TDigest
from kdigest.gpu.add_batch import gpu_compress_batch
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
    var xs = data.copy()
    var n = len(xs)
    # Simple insertion sort is fine for accuracy probes (<= a few 100k, run once).
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key
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


def _bench_batch_one(n: Int) raises:
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

    # GPU batched compress.
    var g_start = perf_counter_ns()
    var comp = gpu_compress_batch[ScaleK2.F32](raw32, w32, Float32(COMPRESSION))
    var g_elapsed = perf_counter_ns() - g_start

    # CPU add loop + merge.
    var cpu = TDigest[ScaleK2](compression=COMPRESSION)
    var c_start = perf_counter_ns()
    for i in range(n):
        cpu.add(raw64[i])
    cpu.merge()
    var c_elapsed = perf_counter_ns() - c_start

    # Accuracy for each (p50, p99 rank error vs exact).
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

    var g_ns = Float64(g_elapsed) / Float64(n)
    var c_ns = Float64(c_elapsed) / Float64(n)
    var speedup = c_ns / g_ns
    print(
        "   N=",
        n,
        ":  GPU ",
        g_ns,
        " ns/pt   CPU ",
        c_ns,
        " ns/pt   speedup ",
        speedup,
        "x   | p50 err G/C ",
        g_p50,
        "/",
        c_p50,
        "  p99 err G/C ",
        g_p99,
        "/",
        c_p99,
    )


def _bench_batch_crossover() raises:
    print("Batched single-digest: GPU gpu_compress_batch vs CPU add-loop")
    print("   (ns per point; GPU wins where speedup > 1x)")
    _bench_batch_one(1_000)
    _bench_batch_one(10_000)
    _bench_batch_one(50_000)
    _bench_batch_one(100_000)
    _bench_batch_one(500_000)
    _bench_batch_one(1_000_000)


# --- 2. Grouped multi-digest crossover ---------------------------------------


def _bench_grouped_one(num_keys: Int, per_key: Int) raises:
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
        keys, values, num_keys, Float32(COMPRESSION)
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


def _bench_grouped_crossover() raises:
    print(
        "\nGrouped multi-digest: GPU gpu_grouped_compress vs CPU per-key loop"
    )
    print("   (ns per digest; one GPU block per key, single launch)")
    _bench_grouped_one(16, 2_000)
    _bench_grouped_one(64, 2_000)
    _bench_grouped_one(256, 2_000)


def main() raises:
    print("=== kdigest GPU crossover benchmarks ===\n")
    if not has_accelerator():
        print("No accelerator present; skipping GPU benchmarks.")
        return
    _bench_batch_crossover()
    _bench_grouped_crossover()
