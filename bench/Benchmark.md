# kdigest Benchmarks

This document describes the benchmarks in [`bench/bench.mojo`](./bench.mojo),
what each one measures and why, and a snapshot of representative results.

Run them with:

```bash
pixi run bench
```

All numbers below were captured on an Apple-silicon (`osx-arm64`) dev box with
the nightly Mojo toolchain. **Throughput is hardware- and build-dependent and is
not a committed SLA**; the accuracy and tracking numbers are deterministic given
the fixed RNG seeds and should reproduce closely across machines.

---

## 1. Throughput

**What it measures.** Wall-clock nanoseconds per `add` on the streaming hot
path, across scale functions (`K0`/`K1`/`K2`/`K3`) and compression levels
(`δ = 100`, `200`).

**Characteristics.**

- Input: 1,000,000 samples from `U(0, 1)`, **pre-generated** so RNG cost sits
  outside the timed region.
- A final `merge()` is included in the timed window so buffered points are
  flushed into centroids.
- The hot path is dominated by buffer appends plus the periodic merge pass; the
  per-`add` cost therefore scales with how often the buffer fills (a function of
  `δ`) and the per-boundary scale-function evaluation (`asin`/`log` for
  `K1`/`K2`/`K3`, arithmetic only for `K0`).

**Representative results.**

| Scale | δ | ns/add | centroids |
| ----- | --- | ------ | --------- |
| K0 | 100 | ~76 | 59 |
| K1 | 100 | ~88 | 62 |
| K2 | 100 | ~90 | 67 |
| K3 | 100 | ~92 | 66 |
| K1 | 200 | ~137 | 120 |
| K2 | 200 | ~139 | 134 |

**Reading the results.** `K0` is fastest (no transcendental in `k(q)`); the
log/asin-based functions add a small constant. Doubling `δ` roughly doubles the
centroid budget and the per-`add` cost, because the merge pass does more work
and runs against a larger centroid array.

---

## 2. Accuracy (rank error vs. exact)

**What it measures.** How close an estimated quantile is to the true quantile,
expressed as **rank error** in quantile space: for an estimate
`v = quantile(q)`, we compute the exact empirical fraction of samples `≤ v`
(via a sorted brute-force reference) and report `|empirical_cdf(v) − q|`. This is
scale-free and comparable across distributions, and is the metric the reference
t-digest literature uses.

**Characteristics.**

- Input: 200,000 samples from `N(0, 1)`, fixed seed.
- Swept across `δ ∈ {50, 100, 200}` and all four scale functions.
- Probes the lower tail (`p0.1`), the median (`p50`), and the upper tail
  (`p99.9`) — the three regimes where the scale-function trade-off is visible.

**Representative results** (rank error; lower is better):

| Scale | δ | p0.1 err | p50 err | p99.9 err |
| ----- | --- | -------- | ------- | --------- |
| K1 | 50 | 3.7e-4 | 3.6e-4 | 2.3e-4 |
| K1 | 100 | 1.1e-4 | 1.9e-4 | 1.3e-4 |
| K1 | 200 | 6.0e-5 | 1.4e-4 | 9.0e-5 |
| K2 | 100 | 2.0e-5 | 9.0e-4 | 5.0e-6 |
| K2 | 200 | 1.5e-5 | 1.1e-4 | 5.0e-6 |
| K3 | 100 | 2.0e-5 | 3.1e-3 | 1.0e-5 |
| K0 | 100 | 9.6e-4 | 2.8e-4 | 1.0e-3 |

**Reading the results.** This is the signature t-digest trade-off made concrete:

- **K2/K3** drive the **tails** into the single-digit-ppm range
  (`p99.9 ≈ 5e-6`) by spending centroids at the extremes — at the cost of a
  **looser median** (`K3`'s `p50 ≈ 3e-3`).
- **K0** (linear) spreads centroids uniformly, so its tails are the weakest
  (`~1e-3`) while its median is competitive.
- **K1** is the balanced default: good everywhere, best-in-class nowhere.
- Increasing `δ` improves accuracy roughly linearly in `1/δ` across the board.

Pick the scale function by which part of the distribution matters: `K2`/`K3`
for extreme-quantile monitoring (p99.9 latencies, tail risk), `K1` for a general
default, `K0` only when speed trumps tail fidelity.

---

## 3. Seasonal / non-stationary stream (decay tracking)

**What it measures.** How well a *recency-weighted* digest follows a
distribution whose center **drifts over time** — the headline use case for the
exponential-decay feature (Phase 3).

**Characteristics.**

- Signal: a sine wave with additive Gaussian noise,

  ```
  x_t = A · sin(2π · t / period) + N(0, noise²)
  ```

  with `A = 10`, `period = 20,000` samples, `noise = 1.0`, over 200,000 samples
  (10 full cycles). The *instantaneous* mean oscillates in `[−10, +10]`, so the
  stream is non-stationary; its long-run mean is `0`.
- Metric: after a one-cycle warm-up, we periodically compare the digest's
  running `quantile(0.5)` against the **known instantaneous mean**
  `A · sin(2π t / period)` and report the mean absolute tracking error.
- We sweep the decay **half-life** relative to the seasonal period, plus a
  no-decay baseline. The noise floor for the median of `N(mean, 1²)` is small
  relative to `A`, so a perfect tracker would score near `0`.

**Representative results** (mean `|running median − instantaneous mean|`; lower
is better):

| Decay half-life | tracking error |
| --------------- | -------------- |
| period / 50 (400 samples) | **0.96** |
| period / 20 (1,000 samples) | 2.02 |
| period / 5 (4,000 samples) | 5.02 |
| period / 2 (10,000 samples) | 6.07 |
| no decay (all-history) | 6.35 |

**Reading the results.** Decay turns a static sketch into a tracker, but only
when the half-life is **short relative to the season**:

- A **short half-life** (`period/50`) rides the wave tightly: error `≈ 0.96`,
  approaching the irreducible noise/compression floor. This is a **~6.6×**
  improvement over no decay.
- As the half-life lengthens, the estimator develops **phase lag and amplitude
  attenuation** (a classic low-pass response to a sinusoid), and tracking error
  climbs toward the no-decay baseline.
- With **no decay**, the median converges to the long-run mean (`≈ 0`) and
  essentially ignores the current phase, so its error is roughly the mean
  absolute value of the sine wave.

**Guidance.** Choose the half-life from the timescale you want to track: short
enough to follow real drift, long enough to average out per-sample noise. For a
periodic signal, a half-life of a few percent of the period tracks well; much
longer than the period and decay buys you almost nothing over a plain digest.

---

## Summary

| Benchmark | Question it answers | Key takeaway |
| --------- | ------------------- | ------------ |
| Throughput | How fast is `add`? | ~75–140 ns/add; `K0` fastest, cost grows with `δ`. |
| Accuracy | How accurate are quantiles? | `K2`/`K3` give ppm-level tails; `K1` balanced; `K0` weak tails. |
| Seasonal | Does decay track drift? | Short half-life (≈ period/50) tracks a drifting signal ~6.6× better than no decay. |

These mirror the quality/benchmark split in the
[tdunning/t-digest](https://github.com/tdunning/t-digest) reference, extended
with the recency-tracking scenario that motivates kdigest's decay feature.
