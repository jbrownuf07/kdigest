"""GPU crossover benchmarks for kdigest (Phase 5, Stage 5f).

Separate from the CPU `bench.mojo` because it requires an accelerator and the
`max` package. Measures *where the GPU wins*:

  1. Batched single-digest: ns/point to fold a batch into one digest, GPU
     (`gpu_compress_batch`, one-shot and with a reused `GpuSession`) vs CPU
     (`TDigest.add` loop), swept over batch size. Also reports p50/p99/p99.9
     rank error for both so speed is read alongside accuracy.
  2. Grouped multi-digest: ns/digest to compress K independent per-key digests,
     GPU (`gpu_grouped_compress`, one block per key, single launch; one-shot and
     session) vs CPU (per-key `add` loop), plus mean per-key rank error.

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
from kdigest.gpu.add_batch import gpu_compress_batch
from kdigest.gpu.grouped import gpu_grouped_compress, GroupedResult
from kdigest.gpu.session import GpuSession


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


comptime _REPS = 3  # best-of-N timing per row, to suppress warm-up noise


# --- 1. Batched single-digest crossover --------------------------------------


def _bench_batch_one(mut session: GpuSession, n: Int) raises:
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

    var once_ns = 0
    var sess_ns = 0
    var c_ns = 0
    var comp = (List[Float32](), List[Float32]())
    var cpu = TDigest[ScaleK2](compression=COMPRESSION)
    for rep in range(_REPS):
        # One-shot: fresh workspace (device + pinned host buffers) per call.
        var t0 = perf_counter_ns()
        _ = gpu_compress_batch[ScaleK2.F32](
            session.ctx, raw32, w32, Float32(COMPRESSION)
        )
        var t1 = perf_counter_ns()
        # Session: reused context + workspace (the steady-state streaming path).
        comp = gpu_compress_batch[ScaleK2.F32](
            session.ctx, session.ws.batch, raw32, w32, Float32(COMPRESSION)
        )
        var t2 = perf_counter_ns()
        var d = TDigest[ScaleK2](compression=COMPRESSION)
        var t3 = perf_counter_ns()
        for i in range(n):
            d.add(raw64[i])
        d.merge()
        var t4 = perf_counter_ns()
        if rep == 0 or Int(t1 - t0) < once_ns:
            once_ns = Int(t1 - t0)
        if rep == 0 or Int(t2 - t1) < sess_ns:
            sess_ns = Int(t2 - t1)
        if rep == 0 or Int(t4 - t3) < c_ns:
            c_ns = Int(t4 - t3)
        cpu = d^

    var dm = List[Float64]()
    var dw = List[Float64]()
    for i in range(len(comp[0])):
        dm.append(Float64(comp[0][i]))
        dw.append(Float64(comp[1][i]))
    var gdig = TDigest[ScaleK2]._from_centroids(
        COMPRESSION, dm, dw, Float64(lo), Float64(hi)
    )
    var nn = Float64(n)
    print(
        "   N=",
        n,
        ":  one-shot ",
        Float64(once_ns) / nn,
        "  session ",
        Float64(sess_ns) / nn,
        "  CPU ",
        Float64(c_ns) / nn,
        " ns/pt   speedup one-shot ",
        Float64(c_ns) / Float64(once_ns),
        "x  session ",
        Float64(c_ns) / Float64(sess_ns),
        "x  | p50 err G/C ",
        _rank_err(sorted_data, gdig.quantile(0.5), 0.5),
        "/",
        _rank_err(sorted_data, cpu.quantile(0.5), 0.5),
        "  p99 err G/C ",
        _rank_err(sorted_data, gdig.quantile(0.99), 0.99),
        "/",
        _rank_err(sorted_data, cpu.quantile(0.99), 0.99),
        "  p99.9 err G/C ",
        _rank_err(sorted_data, gdig.quantile(0.999), 0.999),
        "/",
        _rank_err(sorted_data, cpu.quantile(0.999), 0.999),
    )


def _bench_batch_crossover(mut session: GpuSession) raises:
    print("Batched single-digest: GPU gpu_compress_batch vs CPU add-loop")
    print("   one-shot = fresh workspace per call; session = reused GpuSession")
    print("   (ns per point, best of 3; GPU wins where speedup > 1x)")
    var sizes = [
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        4_000_000,
    ]
    for i in range(len(sizes)):
        _bench_batch_one(session, sizes[i])


# --- 2. Grouped multi-digest crossover ---------------------------------------


def _bench_grouped_one(
    mut session: GpuSession, num_keys: Int, per_key: Int
) raises:
    var rng = _Rng(0x6400 + UInt64(num_keys))
    var keys = List[Int32]()
    var values = List[Float64]()
    for k in range(num_keys):
        for _ in range(per_key):
            keys.append(Int32(k))
            values.append(rng.unit() * Float64(k + 1))

    var once_ns = 0
    var sess_ns = 0
    var c_ns = 0
    var res = GroupedResult()
    var digests = List[TDigest[ScaleK2]]()
    for rep in range(_REPS):
        var t0 = perf_counter_ns()
        _ = gpu_grouped_compress[ScaleK2.F32](
            session.ctx, keys, values, num_keys, Float32(COMPRESSION)
        )
        var t1 = perf_counter_ns()
        res = gpu_grouped_compress[ScaleK2.F32](
            session.ctx,
            session.ws,
            keys,
            values,
            num_keys,
            Float32(COMPRESSION),
        )
        var t2 = perf_counter_ns()
        var ds = List[TDigest[ScaleK2]]()
        for _ in range(num_keys):
            ds.append(TDigest[ScaleK2](compression=COMPRESSION))
        for i in range(len(values)):
            ds[Int(keys[i])].add(values[i])
        for k in range(num_keys):
            ds[k].merge()
        var t3 = perf_counter_ns()
        if rep == 0 or Int(t1 - t0) < once_ns:
            once_ns = Int(t1 - t0)
        if rep == 0 or Int(t2 - t1) < sess_ns:
            sess_ns = Int(t2 - t1)
        if rep == 0 or Int(t3 - t2) < c_ns:
            c_ns = Int(t3 - t2)
        digests = ds^

    # Accuracy: mean rank error over (up to) 256 sampled keys at p50 / p99.
    var g50 = 0.0
    var c50 = 0.0
    var g99 = 0.0
    var c99 = 0.0
    var step = max(1, num_keys // 256)
    var sampled = 0
    var k = 0
    while k < num_keys:
        var kv = List[Float64]()
        for i in range(per_key):
            kv.append(values[k * per_key + i])
        var sd = _sorted(kv)
        var dm = List[Float64]()
        var dw = List[Float64]()
        for j in range(len(res.means[k])):
            dm.append(Float64(res.means[k][j]))
            dw.append(Float64(res.weights[k][j]))
        var gd = TDigest[ScaleK2]._from_centroids(
            COMPRESSION, dm, dw, Float64(res.mins[k]), Float64(res.maxs[k])
        )
        g50 += _rank_err(sd, gd.quantile(0.5), 0.5)
        c50 += _rank_err(sd, digests[k].quantile(0.5), 0.5)
        g99 += _rank_err(sd, gd.quantile(0.99), 0.99)
        c99 += _rank_err(sd, digests[k].quantile(0.99), 0.99)
        sampled += 1
        k += step
    var s = Float64(sampled)
    var nk = Float64(num_keys)
    print(
        "   keys=",
        num_keys,
        " x ",
        per_key,
        "/key:  one-shot ",
        Float64(once_ns) / nk,
        "  session ",
        Float64(sess_ns) / nk,
        "  CPU ",
        Float64(c_ns) / nk,
        " ns/digest   speedup one-shot ",
        Float64(c_ns) / Float64(once_ns),
        "x  session ",
        Float64(c_ns) / Float64(sess_ns),
        "x  | mean p50 err G/C ",
        g50 / s,
        "/",
        c50 / s,
        "  mean p99 err G/C ",
        g99 / s,
        "/",
        c99 / s,
    )


def _bench_grouped_crossover(mut session: GpuSession) raises:
    print(
        "\nGrouped multi-digest: GPU gpu_grouped_compress vs CPU per-key loop"
    )
    print("   (ns per digest, best of 3; one GPU block per key, single launch)")
    print("   one-block path (samples/key <= BLOCK_CAP = 1024):")
    _bench_grouped_one(session, 256, 1_000)
    _bench_grouped_one(session, 1_024, 1_000)
    _bench_grouped_one(session, 4_096, 1_000)
    _bench_grouped_one(session, 4_096, 500)
    _bench_grouped_one(session, 16_384, 500)
    _bench_grouped_one(session, 16_384, 100)
    _bench_grouped_one(session, 65_536, 100)
    _bench_grouped_one(session, 65_536, 20)
    print("   oversized-key fallback (samples/key > BLOCK_CAP):")
    _bench_grouped_one(session, 64, 2_000)
    _bench_grouped_one(session, 256, 2_000)


def main() raises:
    print("=== kdigest GPU crossover benchmarks ===\n")
    if not has_accelerator():
        print("No accelerator present; skipping GPU benchmarks.")
        return
    # One session (context + workspaces) for the whole run.
    var session = GpuSession()
    # Warm-up: the first launch of each kernel pays one-time module load cost;
    # keep it out of the timed rows.
    var wm = List[Float32]()
    var ww = List[Float32]()
    for i in range(4_096):
        wm.append(Float32(i))
        ww.append(Float32(1))
    _ = gpu_compress_batch[ScaleK2.F32](
        session.ctx, session.ws.batch, wm, ww, Float32(COMPRESSION)
    )
    var wk = List[Int32]()
    var wv = List[Float64]()
    for i in range(64):
        wk.append(Int32(i % 4))
        wv.append(Float64(i))
    _ = gpu_grouped_compress[ScaleK2.F32](
        session.ctx, session.ws, wk, wv, 4, Float32(COMPRESSION)
    )

    _bench_batch_crossover(session)
    _bench_grouped_crossover(session)
