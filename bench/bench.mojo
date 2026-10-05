"""Benchmarks for kdigest (Phase 4).

Reports two things, mirroring the reference's quality/benchmark modules:

  1. Throughput: nanoseconds per `add` across scale functions and compression
     levels (the streaming hot path).
  2. Accuracy: tail and median rank error vs. a brute-force exact quantile,
     swept across compression (delta) and scale function.

Run with:
    pixi run bench
"""

from std.math import log, sqrt, cos, sin, pi, abs
from std.random import random_float64, seed
from std.time import perf_counter_ns

from kdigest.decay import Decay
from kdigest.scale import ScaleFunction, ScaleK0, ScaleK1, ScaleK2, ScaleK3
from kdigest.tdigest import TDigest


comptime BENCH_N = 1_000_000
comptime ACC_N = 200_000

# Seasonal benchmark parameters.
comptime SEASON_N = 200_000  # samples in the seasonal stream
comptime SEASON_PERIOD_INT = 20_000  # samples per full sine cycle (int form)
comptime SEASON_PERIOD = 20_000.0  # samples per full sine cycle
comptime SEASON_AMPLITUDE = 10.0  # peak offset of the seasonal mean
comptime SEASON_NOISE = 1.0  # std dev of the additive Gaussian noise


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


# --- Seasonal (non-stationary) stream ----------------------------------------


def _seasonal_mean(t: Int) -> Float64:
    """Instantaneous mean of the seasonal signal at sample index `t`."""
    return SEASON_AMPLITUDE * sin(2.0 * pi * Float64(t) / SEASON_PERIOD)


def _seasonal_tracking_error(half_life: Float64) raises -> Float64:
    """Mean |running median - instantaneous mean| for a given decay half-life.

    half_life <= 0 selects no decay (all-history).
    """
    var decay: Decay
    if half_life <= 0.0:
        decay = Decay.none()
    else:
        decay = Decay.half_life(half_life)
    var d = TDigest[ScaleK1](compression=100.0, decay=decay)

    var err = 0.0
    var samples = 0
    comptime warmup = SEASON_PERIOD_INT  # one full cycle before measuring
    comptime probe_every = 1000

    seed(5)
    for t in range(SEASON_N):
        var x = _seasonal_mean(t) + SEASON_NOISE * _normal()
        d.add(x)
        if t >= warmup and t % probe_every == 0:
            err += abs(d.quantile(0.5) - _seasonal_mean(t))
            samples += 1
    return err / Float64(samples)


def _bench_seasonal() raises:
    # A sinusoidally drifting signal with additive Gaussian noise:
    #     x_t = A·sin(2π·t / period) + N(0, noise²)
    # The distribution's center oscillates, so this is non-stationary. We track
    # how closely each digest's running median follows the *instantaneous* mean
    # as the stream advances. A decayed digest with a half-life short relative
    # to the period rides the wave; a long half-life lags it (phase delay +
    # amplitude attenuation); no decay averages the wave out toward 0.
    print(
        "\nSeasonal stream (x = ",
        SEASON_AMPLITUDE,
        "·sin(2πt/",
        SEASON_PERIOD,
        ") + N(0,",
        SEASON_NOISE,
        "²),  ",
        SEASON_N,
        " samples)",
    )
    print("   mean |running median - instantaneous mean|, lower is better:")
    print(
        "     half-life = period/50 : ",
        _seasonal_tracking_error(SEASON_PERIOD / 50.0),
    )
    print(
        "     half-life = period/20 : ",
        _seasonal_tracking_error(SEASON_PERIOD / 20.0),
    )
    print(
        "     half-life = period/5  : ",
        _seasonal_tracking_error(SEASON_PERIOD / 5.0),
    )
    print(
        "     half-life = period/2  : ",
        _seasonal_tracking_error(SEASON_PERIOD / 2.0),
    )
    print(
        "     no decay (all-history): ",
        _seasonal_tracking_error(0.0),
    )


def main() raises:
    print("=== kdigest benchmarks ===\n")
    _bench_throughput()
    _bench_accuracy()
    _bench_seasonal()
