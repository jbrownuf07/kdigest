"""Benchmarks for kdigest (Phase 4).

Reports two things, mirroring the reference's quality/benchmark modules:

  1. Throughput: nanoseconds per `add` across scale functions and compression
     levels (the streaming hot path).
  2. Accuracy: tail and median rank error vs. a brute-force exact quantile,
     swept across compression (delta) and scale function.

Run with:
    pixi run bench
"""

from std.math import log, sqrt, cos, pi, abs
from std.random import random_float64, seed
from std.time import perf_counter_ns

from kdigest.scale import ScaleFunction, ScaleK0, ScaleK1, ScaleK2, ScaleK3
from kdigest.tdigest import TDigest


comptime BENCH_N = 1_000_000
comptime ACC_N = 200_000


def _normal() -> Float64:
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    var u2 = random_float64()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


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
    var lo = 0
    var hi = len(sorted_data)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_data[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return Float64(lo) / Float64(len(sorted_data))


# --- Throughput --------------------------------------------------------------


def _bench_add[S: ScaleFunction](name: String, compression: Float64) raises:
    seed(1)
    # Pre-generate inputs so RNG cost is outside the timed region.
    var data = List[Float64]()
    for _ in range(BENCH_N):
        data.append(random_float64())

    var d = TDigest[S](compression=compression)
    var start = perf_counter_ns()
    for i in range(BENCH_N):
        d.add(data[i])
    d.merge()
    var elapsed = perf_counter_ns() - start

    var ns_per_add = Float64(elapsed) / Float64(BENCH_N)
    print(
        "  ",
        name,
        " delta=",
        Int(compression),
        ":  ",
        ns_per_add,
        " ns/add   (",
        len(d),
        " centroids)",
    )


def _bench_throughput() raises:
    print("Throughput (", BENCH_N, " adds of U(0,1)):")
    _bench_add[ScaleK1]("K1", 100.0)
    _bench_add[ScaleK1]("K1", 200.0)
    _bench_add[ScaleK2]("K2", 100.0)
    _bench_add[ScaleK2]("K2", 200.0)
    _bench_add[ScaleK0]("K0", 100.0)
    _bench_add[ScaleK3]("K3", 100.0)


# --- Accuracy sweep ----------------------------------------------------------


def _acc_one[
    S: ScaleFunction
](
    name: String,
    compression: Float64,
    sorted_data: List[Float64],
    data: List[Float64],
) raises:
    var d = TDigest[S](compression=compression)
    for i in range(len(data)):
        d.add(data[i])

    var p001 = abs(_empirical_cdf(sorted_data, d.quantile(0.001)) - 0.001)
    var p50 = abs(_empirical_cdf(sorted_data, d.quantile(0.5)) - 0.5)
    var p999 = abs(_empirical_cdf(sorted_data, d.quantile(0.999)) - 0.999)
    print(
        "  ",
        name,
        " delta=",
        Int(compression),
        ":  p0.1 err=",
        p001,
        "  p50 err=",
        p50,
        "  p99.9 err=",
        p999,
    )


def _bench_accuracy() raises:
    print("\nAccuracy (rank error vs exact over ", ACC_N, " N(0,1) samples):")
    seed(2)
    var data = List[Float64]()
    for _ in range(ACC_N):
        data.append(_normal())
    var sorted_data = _sorted(data)

    _acc_one[ScaleK1]("K1", 50.0, sorted_data, data)
    _acc_one[ScaleK1]("K1", 100.0, sorted_data, data)
    _acc_one[ScaleK1]("K1", 200.0, sorted_data, data)
    _acc_one[ScaleK2]("K2", 100.0, sorted_data, data)
    _acc_one[ScaleK2]("K2", 200.0, sorted_data, data)
    _acc_one[ScaleK3]("K3", 100.0, sorted_data, data)
    _acc_one[ScaleK0]("K0", 100.0, sorted_data, data)


def main() raises:
    print("=== kdigest benchmarks ===\n")
    _bench_throughput()
    _bench_accuracy()
