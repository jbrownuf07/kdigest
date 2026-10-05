# kdigest Benchmarks

This document describes the benchmarks in [`bench/bench.mojo`](./bench.mojo),
what each one measures and why, and a snapshot of representative results.

Run them with:

```bash
pixi run bench       # CPU: throughput, accuracy, seasonal (Sections 1–3)
pixi run bench-gpu   # GPU: accuracy + CPU/GPU crossover (Section 4; needs an accelerator)
```

All numbers below were captured on the test system described next. **Throughput
is hardware- and build-dependent and is not a committed SLA**; the accuracy and
tracking numbers are deterministic given the fixed RNG seeds and should
reproduce closely across machines (see
[How portable are these numbers?](#how-portable-are-these-numbers)).

### Test system

| Component | Value |
| --------- | ----- |
| Machine | MacBook Pro (`MacBookPro18,2`), 16-inch, 2021 |
| SoC | Apple M1 Max |
| CPU | 10 cores (8 performance + 2 efficiency); the CPU baselines are **single-threaded** |
| GPU | Apple M1 Max integrated GPU, 24 cores, Metal 4 |
| Memory | 32 GB unified LPDDR5 (shared by CPU and GPU, ~400 GB/s peak) |
| OS | macOS 26.7.1 (build 25G241) |
| Toolchain | Mojo 1.1.0 (`8189361e`), MAX `max-core` 26.6.0 (release), via pixi |
| Platform | `osx-arm64` |

Timings are wall-clock, best of 3 runs per row (Section 4), on a laptop. Power
source and thermal state were not controlled.

### How portable are these numbers?

**Accuracy: mostly portable. Speed: not portable. On different hardware the
absolute numbers will differ, and the crossovers and trends can shift too.**

- **Accuracy** comes from the algorithm and the fixed seeds, so the rank-error
  tables should reproduce closely anywhere. The CPU path is Float64 and
  deterministic. The GPU path is Float32 and runs the kernels' operations in a
  fixed order, but `log`/`asin` implementations differ between GPU vendors and
  drivers. Cluster boundaries can therefore move slightly on another device:
  expect *close*, not bit-identical, GPU accuracy numbers. (The bit-identical
  guarantees in Section 4 compare GPU paths with each other *on the same
  device*.)
- **Absolute throughput** depends on CPU single-core speed, GPU size, memory
  bandwidth and the driver. A newer CPU core lowers the CPU baseline and pushes
  the crossovers up. A bigger GPU lowers the GPU numbers and pulls them down.
- **Crossover points and trends can change, not just scale.** The GPU curves
  here are shaped by costs specific to this setup:
  - **Fixed cost per call (~0.9 ms)** is mostly Metal launch and
    synchronization latency (4e measures it: ~30–50 µs per launch, ~170 µs per
    synchronize). CUDA on a discrete GPU typically has much lower launch
    latency, which would move the small-batch crossover (~40k here) *down*.
  - **Unified memory.** On Apple silicon a "host↔device copy" is a memcpy
    within the same RAM. A discrete GPU has to cross PCIe, so per-byte transfer
    costs are higher and large batches could become transfer-bound. That could
    flatten or reverse the large-N trend.
  - **The taper above ~1M points** comes from the GPU merge-sort passes and
    memory bandwidth (~400 GB/s here). A GPU with much higher bandwidth (e.g.
    HBM) would taper later or not at all.
  - **The grouped path** runs one block per key, so its speedup scales with how
    many blocks the GPU runs concurrently (24 cores here; data-center GPUs have
    100+ SMs). A bigger GPU should widen the 2.5–4.3× lead. The ≤ ~100
    samples/key rows are limited by CPU-side bookkeeping, so they would
    change less.
  - **Kernel shape.** The kernels assume `BLOCK_CAP = 1024` threads per block
    and ~20 KB of shared memory per block. On GPUs with different warp or
    wavefront widths, occupancy limits or shared-memory sizes, the
    per-block cost (and possibly the best `BLOCK_CAP`) would differ.
- **The dispatch thresholds** (`_GPU_CROSSOVER` = 60k,
  `_GPU_CROSSOVER_SESSION` = 40k) were tuned on this machine only. Before
  relying on them on other hardware, rerun `pixi run bench-gpu` there and
  re-derive them.

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

All GPU numbers are Float32 (the device dtype); CPU is Float64 and
single-threaded. Captured on the Apple M1 Max described in
[Test system](#test-system) (24-core integrated GPU, Metal, unified memory).
**These are not an SLA**: they exist to locate the break-even points on this
machine and to be honest about where the backend stands. The crossovers in
particular are expected to move on other hardware (see
[How portable are these numbers?](#how-portable-are-these-numbers)).

The benchmark holds **one `GpuSession`** (device context + reusable workspaces)
for the whole run and times each GPU path two ways: **one-shot** (a fresh
workspace per call — what `TDigest.add_batch(values)` does) and **session** (the
reused workspace — `TDigest.add_batch(values, session)`, and what
`GroupedTDigest` does internally). A warm-up call keeps the one-time kernel
module load out of the timed rows.

### 4a. Batched single-digest — `gpu_compress_batch` vs CPU `add` loop

Folding one batch into one digest, swept over batch size. "ns/pt" is wall-clock
nanoseconds per input point (lower is better), best of 3 runs. History columns
are earlier implementations on identical input:

- **5f** — the original Stage 5f code (O(n²) host sort, five buffers per tile).
- **ref** — after the sort/buffer fixes (4c): host merge sort, then one
  upload → single-block launch → readback *per tile*.
- **step 3** — GPU sort + device-resident tiling (4d), before the fixed-cost
  work in 4e.
- **one-shot / session** — current code (4e).

| N | 5f | ref | step 3 | one-shot | **session** | CPU | one-shot vs CPU | **session vs CPU** |
| --------- | ------- | ------ | ---- | ---- | ----- | --- | ----- | ----- |
| 1,000 | ~12,100 | ~3,160 | ~3,420 | ~1,110 | ~979 | ~86 | 0.08× | 0.09× |
| 5,000 | — | — | — | ~404 | ~350 | ~86 | 0.21× | 0.24× |
| 10,000 | ~5,400 | ~3,710 | ~530 | ~291 | ~272 | ~87 | 0.30× | 0.32× |
| 20,000 | — | — | — | ~185 | ~157 | ~91 | 0.49× | 0.58× |
| 30,000 | — | — | — | ~157 | ~114 | ~89 | 0.57× | 0.78× |
| 50,000 | ~10,700 | ~3,590 | ~159 | ~92 | **~71** | ~90 | 0.98× | **1.27×** |
| 100,000 | ~17,500 | ~3,720 | ~100 | **~56** | **~47** | ~87 | **1.54×** | **1.85×** |
| 200,000 | — | ~3,160 | ~73 | **~51** | **~44** | ~90 | **1.76×** | **2.04×** |
| 500,000 | ~73,900 | ~3,290 | ~69 | **~49** | **~47** | ~88 | **1.79×** | **1.89×** |
| 1,000,000 | ~144,700 | ~3,670 | ~60 | **~55** | **~51** | ~89 | **1.62×** | **1.73×** |
| 2,000,000 | — | ~3,620 | ~63 | **~64** | **~57** | ~88 | **1.38×** | **1.55×** |
| 4,000,000 | — | ~3,380 | ~71 | **~69** | **~64** | ~88 | **1.29×** | **1.38×** |

All GPU variants return **bit-identical centroids** (enforced by
`test_device_pipeline_matches_reference` and
`test_session_reuse_matches_one_shot`, including heavy-tie inputs), so accuracy
is the same for all of them. Rank error vs. the exact quantile (lower is
better):

| N | GPU p50 | CPU p50 | GPU p99 | CPU p99 | GPU p99.9 | CPU p99.9 |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 10,000 | 1.1e-3 | 2.4e-3 | 1.0e-4 | 3.0e-4 | 1.0e-4 | 1.0e-4 |
| 100,000 | 6.5e-4 | 2.1e-4 | 2.0e-5 | 1.1e-4 | 1.0e-5 | 3.0e-5 |
| 200,000 | 1.0e-4 | 2.4e-4 | 9.5e-5 | 3.4e-4 | 5.0e-6 | 3.5e-5 |
| 1,000,000 | 7.4e-5 | 1.8e-3 | 5.2e-5 | 3.0e-4 | 1.7e-5 | 1.4e-5 |
| 4,000,000 | 9.8e-5 | 1.3e-4 | 2.0e-5 | 9.9e-5 | 1.8e-6 | 2.2e-5 |

**The GPU now beats the CPU on single-digest batches from ~40k points with a
session (~60k one-shot), peaking at ~2× around 100k–500k points.** The speedup
tapers at multi-million-point batches as the GPU merge-sort passes grow
(log₂(N/1024) passes over the whole batch).

### 4b. Grouped multi-digest — `gpu_grouped_compress` vs CPU per-key loop

Many independent per-key digests. "ns/digest" is wall-clock time divided by the
number of keys, best of 3. Keys with **≤ `BLOCK_CAP` (1,024) samples** take the
one-block-per-key path (a single launch for all of them); larger keys take the
oversized-key fallback through `gpu_compress_batch`, one key at a time.
"before" is the previous run (pre-4e).

| keys × samples/key | path | before | one-shot | **session** | CPU | **session vs CPU** |
| ------------------ | ---- | ------ | -------- | ----------- | --- | ------- |
| 256 × 1,000 | one block per key | ~98k | ~25k | ~31k | ~81k | **2.6×** |
| 1,024 × 1,000 | one block per key | ~86k | ~28k | ~19k | ~82k | **4.3×** |
| 4,096 × 1,000 | one block per key | ~80k | ~22k | ~19k | ~82k | **4.3×** |
| 4,096 × 500 | one block per key | ~44k | ~18k | ~15k | ~38k | **2.6×** |
| 16,384 × 500 | one block per key | ~43k | ~16k | ~15k | ~38k | **2.5×** |
| 16,384 × 100 | one block per key | ~16.6k | ~12.1k | ~11.6k | ~5.5k | 0.48× |
| 65,536 × 100 | one block per key | ~15.9k | ~11.6k | ~11.2k | ~5.5k | 0.49× |
| 65,536 × 20 | one block per key | — | ~8.4k | ~8.1k | ~1.7k | 0.21× |
| 64 × 2,000 | oversized fallback | ~5.0M | ~921k | ~889k | ~169k | 0.19× |
| 256 × 2,000 | oversized fallback | ~5.2M | ~924k | ~912k | ~172k | 0.19× |

(256 × 1,000 is noisy at this size — one-shot and session swap places run to
run; the steady-state rows with ≥ 1,024 keys are the reliable ones.)

Per-key accuracy — mean rank error over up to 256 sampled keys (GPU / CPU):

| keys × samples/key | p50 err | p99 err |
| ------------------ | ------- | ------- |
| 4,096 × 1,000 | 2.9e-3 / 2.5e-3 | 4.4e-4 / 3.3e-4 |
| 16,384 × 500 | 4.1e-3 / 3.2e-3 | 0 / 0 |
| 65,536 × 100 | 6.4e-3 / 5.9e-3 | 0 / 0 |
| 256 × 2,000 (fallback) | 2.2e-3 / 2.1e-3 | 4.4e-4 / 3.5e-4 |

With ≤ 1,000 samples a key's digest is near-exact anyway (rank-error
granularity is 1/n), so both sides are comparable; per-key accuracy is
unaffected by the speed work (`test_grouped_matches_per_key_reference_exactly`
checks the grouped centroids equal the single-digest reference per key, bit for
bit).

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

**What was left after step 3** — fixed per-call cost kept batches under
~150k points on the CPU. That is the subject of 4e.

### 4e. Removing the fixed per-call cost

Measured directly on this machine, the per-call fixed costs were:

| operation | cost |
| --------- | ---- |
| `map_to_host` (blocking upload / readback) | ~670 µs each |
| pinned host buffer + async `enqueue_copy` + one `synchronize` | ~250 µs |
| bare `synchronize` | ~170 µs |
| kernel launch | ~30–50 µs |
| create a pinned host buffer | ~200 µs |
| create a device buffer | ~15 µs |

A small step-3 call did ~4 `map_to_host`s (~2.7 ms) plus allocations: that
*was* the ~3 ms floor. Three changes:

1. **Tile offsets on the device.** A one-block `_scan_counts_kernel` computes
   each pass's survivor prefix sums (any number of tiles: per-thread serial
   chunks + a block scan) and writes the survivor total at `offsets[tiles]`.
   Per intermediate pass the host now copies back **one integer** and
   synchronizes once, instead of a blocking count readback plus a blocking
   offsets upload.
2. **Reusable workspaces + async copies.** `CompressWorkspace` owns every device
   buffer and pinned host staging buffer, growing geometrically; uploads and
   readbacks are `enqueue_copy` on sub-buffers. A call that fits one block now
   synchronizes **once**. `GpuSession` (context + workspaces) is the public
   handle: `TDigest.add_batch(values, session)` reuses it, and
   `GroupedTDigest` holds one for its lifetime.
3. **Grouped path.**
   - Each block **bitonic-sorts its own key in shared memory**, removing the
     host per-key sort (which also allocated two lists per key).
   - Output is written **compactly at each key's input offset** instead of
     `num_keys × BLOCK_CAP` slots — at 65,536 keys the old layout read back
     ~512 MB.
   - Host work is two linear passes (count, scatter straight into pinned
     staging), then one upload, one launch, one readback, one synchronize.
   - The oversized-key fallback reuses the session's `CompressWorkspace`.

Effect (session numbers): single-digest **~530 → ~272 ns/pt at 10k**, **~100 →
~47 at 100k**, ~60 → ~51 at 1M; fixed cost per call **~3.4 ms → ~0.9 ms**.
Grouped **~80k → ~19k ns/digest** at 4,096 × 1,000 (≈1× → 4.3× the CPU);
oversized fallback ~5.0M → ~0.9M ns/digest.

**Reading the results.**

- **Accuracy is unchanged and remains the GPU's strength.** Every speed change
  is bit-identical to the reference path. The parallel compress is consistently
  **sharper in the tail**: single-digest p99 rank error ~`2e-5–1e-4` vs
  `1e-4–3e-4` on the CPU, p99.9 typically several times lower.
- **Single-digest `add_batch` wins from ~40k points (session) / ~60k
  (one-shot)**, peaking at ~2× at 100k–500k. Below that the remaining ~0.9 ms
  per call (a handful of launches plus one or two synchronizes) dominates.
- **Grouped one-block-per-key now wins clearly at hundreds+ samples per key**:
  **2.5–4.3×** at 500–1,000 samples/key. With ≤ 100 samples per key the CPU is
  still ~2–5× faster: each key's host bookkeeping (building its result lists)
  and a 1,024-thread block for ~100 values cost more than the CPU's tiny merge.
- **The oversized-key fallback is still the slow path** (~0.19×): it runs one
  `gpu_compress_batch` per oversized key, ~0.9 ms each. Batching all oversized
  keys into one segmented launch would fix it; until then keep keys ≤
  `BLOCK_CAP` samples per batch.

**Crossover verdict.**

- **Single-digest `add_batch`:** crossover **~40k points with a `GpuSession`,
  ~60k one-shot**; the GPU is 1.3–2.0× faster above it.
- **Grouped one-block-per-key:** **2.5–4.3×** faster at 500–1,000 samples per
  key; the CPU still wins at ≤ ~100 samples per key.

**Dispatch guidance.** `TDigest.add_batch(values)` switches to the GPU at
**60k** points (`_GPU_CROSSOVER`); `TDigest.add_batch(values, session)` at
**40k** (`_GPU_CROSSOVER_SESSION`). Streaming callers should hold one
`GpuSession`. For `GroupedTDigest`, prefer the GPU when keys carry hundreds to
~1,000 samples per batch; with very small per-key batches the CPU is faster.

---

## Summary

| Benchmark | Question it answers | Key takeaway |
| --------- | ------------------- | ------------ |
| Throughput | How fast is `add`? | ~75–140 ns/add; `K0` fastest, cost grows with `δ`. |
| Accuracy | How accurate are quantiles? | `K2`/`K3` give ppm-level tails; `K1` balanced; `K0` weak tails. |
| Seasonal | Does decay track drift? | Short half-life (≈ period/50) tracks a drifting signal ~6.6× better than no decay. |
| GPU crossover | Is the GPU as accurate / faster? | **Accuracy:** GPU matches CPU and is sharper in the tail (p99 err ~2–5e-5 vs ~1–3e-4). **Speed:** single-digest `add_batch` beats the CPU from ~40k points with a reused `GpuSession` (~60k one-shot), up to ~2× (~44–51 ns/pt vs ~88); grouped one-block-per-key is 2.5–4.3× faster at 500–1,000 samples/key; CPU still wins for ≤ ~100 samples/key and for oversized keys. |

These mirror the quality/benchmark split in the
[tdunning/t-digest](https://github.com/tdunning/t-digest) reference, extended
with the recency-tracking scenario that motivates kdigest's decay feature and a
GPU crossover study that locates where the Phase 5 accelerator does — and does
not — pay off.
