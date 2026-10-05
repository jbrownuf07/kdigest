# kdigest Benchmarks

This document describes the benchmarks in [`bench/bench.mojo`](./bench.mojo),
what each one measures and why, and a snapshot of representative results.

Run them with:

```bash
pixi run bench       # CPU: throughput, accuracy, seasonal (Sections 1–3)
pixi run bench-gpu   # GPU: accuracy + CPU/GPU crossover (Section 4; needs an accelerator)
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

## 4. GPU backend — accuracy and the CPU/GPU crossover

**What it measures.** The Phase 5 GPU backend (`add_batch`, `GroupedTDigest`,
vectorized query, merge-tree) uses a *parallel* k-bucketing compress instead of
the CPU's sequential greedy sweep. This benchmark
([`bench/bench_gpu.mojo`](./bench_gpu.mojo), `pixi run bench-gpu`) answers two
questions side by side: **is the GPU result as accurate**, and **when (if ever)
is it faster** than the CPU on the current implementation.

All GPU numbers are Float32 (the device dtype); CPU is Float64. Captured on an
Apple-silicon (`osx-arm64`) dev box. **These are not an SLA** — they exist to
locate the break-even points and to be honest about where the backend stands.

The benchmark creates **one `DeviceContext` for the whole run** and passes it to
every call (the `gpu_compress_batch[S](ctx, ...)` /
`gpu_grouped_compress[S](ctx, ...)` overloads), and runs a warm-up call so the
one-time kernel module load stays out of the timed rows.

### 4a. Batched single-digest — `gpu_compress_batch` vs CPU `add` loop

Folding one batch into one digest, swept over batch size. "ns/pt" is wall-clock
nanoseconds per input point (lower is better), best of 3 runs. Four
implementations are timed on identical input:

- **5f** — the original Stage 5f code (O(n²) host sort, five buffers per tile).
- **ref** — after the sort/buffer fixes (4c): host merge sort, then one
  upload → single-block launch → readback *per tile*.
- **A** — step 3 part A: host sort, then **device-resident tiling** (one upload,
  one multi-block launch per pass, one readback).
- **B** — step 3 part B, the default `gpu_compress_batch`: **GPU sort** +
  device-resident tiling. The batch never leaves the GPU until the final
  centroids.

| N | 5f | ref | A | **B (default)** | CPU | B vs CPU |
| --------- | ------- | ------ | ---- | ----- | --- | ----- |
| 1,000 | ~12,100 | ~3,160 | ~3,040 | ~3,420 | ~85 | 0.02× |
| 10,000 | ~5,400 | ~3,710 | ~600 | ~530 | ~89 | 0.17× |
| 50,000 | ~10,700 | ~3,590 | ~236 | ~159 | ~87 | 0.55× |
| 100,000 | ~17,500 | ~3,720 | ~175 | ~100 | ~89 | 0.89× |
| 200,000 | — | ~3,160 | ~148 | **~73** | ~88 | **1.21×** |
| 300,000 | — | ~3,650 | ~142 | **~74** | ~88 | **1.20×** |
| 500,000 | ~73,900 | ~3,290 | ~138 | **~69** | ~89 | **1.29×** |
| 1,000,000 | ~144,700 | ~3,670 | ~139 | **~60** | ~90 | **1.49×** |
| 2,000,000 | — | ~3,620 | ~141 | **~63** | ~88 | **1.40×** |
| 4,000,000 | — | ~3,380 | ~148 | **~71** | ~89 | **1.25×** |

All three GPU paths return **bit-identical centroids** (enforced by
`test_device_pipeline_matches_reference`, including heavy-tie inputs), so the
accuracy is the same for all of them:

| N | GPU p50 err | CPU p50 err | GPU p99 err | CPU p99 err |
| --------- | ------ | ------ | ------ | ------ |
| 10,000 | 1.1e-3 | 2.4e-3 | 1.0e-4 | 3.0e-4 |
| 100,000 | 6.5e-4 | 2.1e-4 | 2.0e-5 | 1.1e-4 |
| 1,000,000 | 7.4e-5 | 1.8e-3 | 5.2e-5 | 3.0e-4 |
| 4,000,000 | 9.8e-5 | 1.3e-4 | 2.0e-5 | 9.9e-5 |

**The GPU now beats the CPU on single-digest batches from ~150k points,
peaking at ~1.5× around 1M points.**

### 4b. Grouped multi-digest — `gpu_grouped_compress` vs CPU per-key loop

Many independent per-key digests. "ns/digest" is wall-clock time divided by the
number of keys. Keys with **≤ `BLOCK_CAP` (1,024) samples** take the
one-block-per-key path (a single launch for all of them); larger keys take the
oversized-key fallback through `gpu_compress_batch`, one key at a time.

| keys × samples/key | path | GPU ns/digest | CPU ns/digest | speedup |
| ------------------ | ---- | ------------- | ------------- | ------- |
| 256 × 1,000 | one block per key | ~98k | ~82k | 0.84× |
| 1,024 × 1,000 | one block per key | ~86k | ~83k | 0.97× |
| 4,096 × 1,000 | one block per key | ~80k | ~82k | **1.03×** |
| 4,096 × 500 | one block per key | ~44k | ~39k | 0.89× |
| 16,384 × 500 | one block per key | ~43k | ~39k | 0.91× |
| 16,384 × 100 | one block per key | ~16.6k | ~5.5k | 0.33× |
| 65,536 × 100 | one block per key | ~15.9k | ~5.6k | 0.35× |
| 64 × 2,000 | oversized fallback | ~5.0M | ~171k | 0.034× |
| 256 × 2,000 | oversized fallback | ~5.2M | ~171k | 0.033× |

**Correction to the earlier version of this table.** The original 5f grouped rows
all used 2,000 samples per key — above `BLOCK_CAP` — so *every* key went through
the oversized fallback and the table never measured the one-block-per-key kernel
at all. The rows above separate the two paths.

### 4c. What was actually slow (and what changed)

The original 5f write-up attributed the slowness to creating a `DeviceContext`
per call. **That was wrong** — measured directly, `DeviceContext()` costs
~**0.7 ms**, and `gpu_compress_batch` only created one per call. Profiling one
`gpu_compress_batch` call (before the fix):

| N | `DeviceContext()` | host sort | total call |
| ------- | ------ | -------- | -------- |
| 10,000 | 0.7 ms | 15 ms | 58 ms |
| 50,000 | 0.7 ms | 345 ms | 527 ms |
| 100,000 | 0.7 ms | 1,394 ms | 1,750 ms |

The real costs, and the fixes:

1. **Quadratic host sort (65–80% of the time).** `cpu_sort_pairs` — written as a
   test oracle — was an O(n²) insertion sort, and it ran on the whole input, on
   every reduction pass, and on every grouped key slice. It is now a stable
   O(n log n) bottom-up merge sort (insertion-sorted runs of 32, then merged).
   This is what turned the growing per-point cost into a flat one.
2. **Per-tile device allocation.** Every 1,024-element tile allocated five fresh
   device buffers. They are now allocated **once per call** and reused by every
   tile.
3. **Context reuse.** The GPU entry points gained overloads that take a caller's
   `DeviceContext`; `GroupedTDigest` holds one long-lived context (created on its
   first GPU batch). This is not a meaningful speed win, but it removes the
   repeated context creation that exhausted Metal command queues
   ("Failed to create Metal command queue") in tight loops — the benchmark now
   runs its full sweep, including 65,536 keys, without crashing.

### 4d. Step 3 — sort and tiling on the GPU

After 4c the batched path was flat at ~3.8 µs/pt. Splitting that cost showed
two separate problems, which is why step 3 has two parts:

| N | total | host sort | per-tile round trips | tiles | per round trip |
| ------- | ------ | ------ | ------ | ---- | ------ |
| 100,000 | 3,804 ns/pt | 101 ns/pt | 3,704 ns/pt | 98 | ~3.8 ms |
| 1,000,000 | 3,859 ns/pt | 120 ns/pt | 3,739 ns/pt | 977 | ~3.8 ms |

- **Round trips were ~97% of the cost.** Every `map_to_host` waits for the GPU,
  and each tile did several of them.
- **The host sort alone (100–120 ns/pt) already exceeded the CPU digest's whole
  ~90 ns/pt.** The CPU digest only sorts small buffers; the GPU path sorts the
  entire batch. So the sort *had* to move to the GPU to win.

**Part A — device-resident tiling** (`gpu_compress_batch_host_sort`). The
compress kernel was generalized so block `b` compresses tile `b`; each pass is
now **one multi-block launch** over all tiles. A small `_gather_tiles_kernel`
packs each tile's survivors densely on the device for the next pass. The host
only reads the per-tile survivor counts each pass (a few hundred integers) and
uploads their prefix sums. Result: **~3,670 → ~139 ns/pt at 1M** — now bounded
by the host sort, exactly as predicted.

**Part B — GPU sort** (`kdigest.gpu.sort`, used by the default
`gpu_compress_batch`). A stable sort entirely on the device:

1. `_tile_sort_kernel` — one block per 1,024-element tile, bitonic sort in
   shared memory of (mean, original index) pairs. Comparing on (mean, index)
   makes every element distinct, so the result is exactly a *stable* sort.
2. `_merge_pass_kernel` — merges adjacent sorted runs, one thread per element:
   its output position is its rank in its own run plus a binary-search count of
   preceding elements in the partner run. ~10 passes for 1M points, enqueued
   back to back with **no host synchronization** between them.
3. `_permute_kernel` — gathers the weights through the final permutation.

Result: **~139 → ~60 ns/pt at 1M**, and the GPU passes the CPU.

**Two correctness details that fell out of this.**

- **Sort once, not per pass.** After a compress pass, each tile's centroids are
  in mean order and tiles are in global order, so the survivors are already
  sorted for the next pass. The old path re-sorted between passes.
- **Clamping centroid means.** That "already sorted" argument relies on every
  centroid's mean lying within its segment's [first, last] value. In Float32 the
  weighted mean Σw·x/Σw can round a hair outside that range, and with heavily
  tied inputs this produced adjacent centroids out of order by one ulp. The
  kernel now clamps each mean to its segment's range. This makes the ordering
  guarantee exact, and the three paths bit-identical, ties included.

**What's left.** Below ~150k points the GPU still loses because of fixed
per-call cost. A call allocates six device buffers, and each compress pass does
one count readback and one offsets upload that both wait for the GPU. That is
~3 ms in total, which is why N = 1,000 still costs ~3,400 ns/pt. Possible next
steps are computing the tile offsets on the device (removing the per-pass sync),
reusing device buffers across calls, and a faster bitonic merge for the sort.

**Reading the results.**

- **Accuracy is the GPU's clear win.** The parallel compress is at least as
  accurate as the CPU digest and consistently **sharper in the tail**: GPU `p99`
  rank error ~`2e-5–5e-5` vs `1e-4–3e-4` for the sequential CPU sweep. The `p50`
  is comparable.
- **Grouped (one block per key) is at break-even.** With ~500–1,000 samples per
  key the GPU runs at **0.84–1.03×** the CPU and edges ahead at 4,096 × 1,000.
  With few samples per key (100) the CPU wins ~3×, because per-key host work
  (scatter, per-slice sort, building the launch) dominates the tiny GPU job.
- **Batched single-digest now wins above ~150k points** (1.2–1.5×). Below that
  the fixed per-call cost (~3 ms: buffer allocation plus a couple of GPU waits
  per pass) dominates.
- **The oversized-key fallback is still the slow path.** It runs
  `gpu_compress_batch` once *per oversized key*, so it pays that ~3–5 ms
  per-call cost per key (~5M ns/digest, down from ~10M). Avoid it by keeping
  keys ≤ `BLOCK_CAP` samples per batch.

**Crossover verdict.**

- **Single-digest `add_batch`:** crossover at **~150k points per batch**; the GPU
  is 1.2–1.5× faster above it.
- **Grouped one-block-per-key:** at break-even (~1×) with ~1,000 samples per key
  and thousands of keys.

**Dispatch guidance.** `TDigest.add_batch` now switches to the GPU at **150k**
points (`_GPU_CROSSOVER`, raised from the provisional 50k based on these
measurements); below that it uses the CPU path. For `GroupedTDigest`, the GPU
path is competitive when keys carry hundreds to ~1,000 samples per batch; with
very small per-key batches the CPU is faster.

---

## Summary

| Benchmark | Question it answers | Key takeaway |
| --------- | ------------------- | ------------ |
| Throughput | How fast is `add`? | ~75–140 ns/add; `K0` fastest, cost grows with `δ`. |
| Accuracy | How accurate are quantiles? | `K2`/`K3` give ppm-level tails; `K1` balanced; `K0` weak tails. |
| Seasonal | Does decay track drift? | Short half-life (≈ period/50) tracks a drifting signal ~6.6× better than no decay. |
| GPU crossover | Is the GPU as accurate / faster? | **Accuracy:** GPU matches CPU and is sharper in the tail (p99 err ~2–5e-5 vs ~1–3e-4). **Speed:** single-digest `add_batch` beats the CPU above ~150k points (1.2–1.5×, ~60 ns/pt at 1M after moving sort + tiling onto the GPU); grouped one-block-per-key is at break-even (0.84–1.03× at ~1,000 samples/key). |

These mirror the quality/benchmark split in the
[tdunning/t-digest](https://github.com/tdunning/t-digest) reference, extended
with the recency-tracking scenario that motivates kdigest's decay feature and a
GPU crossover study that locates where the Phase 5 accelerator does — and does
not — pay off.
