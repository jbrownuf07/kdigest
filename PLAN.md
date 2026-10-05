# kdigest — Implementation Plan

Goal: a Mojo t-digest library specialized for accurate **extreme-tail**
quantiles over streaming data, with (1) **exponential decay** of centroid
weights and (2) **pluggable scale functions** matching the
[tdunning/t-digest](https://github.com/tdunning/t-digest) reference.

The plan is phased so each phase produces a compiling, tested artifact. We build
the exact-but-slow pieces first, validate accuracy against ground truth, then
layer on decay and performance.

---

## Phase 0 — Project scaffolding

**Deliverables**

- `src/kdigest/` package with an empty `__init__.mojo`.
- `test/` directory.
- `pixi.toml` tasks: `test`, `fmt`, `build`.
- Confirm `pixi install` resolves and a trivial `mojo` file compiles.

**Steps**

1. Add `[tasks]` to `pixi.toml`:
   - `test` — runs `scripts/run_tests.sh`, which `mojo run`s each
     `test/test_*.mojo` against a `TestSuite` runner (the `mojo test`
     subcommand was removed in Mojo 1.1).
   - `fmt  = "mojo format src/ test/"`
   - `build` — compiles the package to verify it builds.
2. Create package skeleton and a smoke test that imports the package.
3. Verify: `pixi run test` passes on an empty suite.

**Done when:** repo compiles and the test task runs green.

---

## Phase 1 — Centroid + scale functions (the math core)

This is the foundation and the part most unique to t-digest. It is pure,
side-effect-free, and highly testable, so we validate it in isolation first.

**Deliverables**

- `centroid.mojo`: `Centroid(mean, weight)` value type with an `add_sample`
  merge helper (weighted mean update).
- `scale.mojo`: a `ScaleFunction` trait and four implementations.

**Scale functions to implement** (from the reference):

| Name | Forward `k(q)` | Notes |
| ---- | -------------- | ----- |
| `ScaleK0` | `q * (δ / 2)` | Linear; inverse is trivial. |
| `ScaleK1` | `(δ / 2π) · asin(2q − 1)` | Classic; inverse uses `sin`. |
| `ScaleK2` | normalized `log(q/(1−q))` with count term | Proven size bounds. |
| `ScaleK3` | piecewise `K2`-like | Most aggressive tail control. |

Each implements `k(q, normalizer)`, `q(k, normalizer)` (inverse), and
`normalizer(compression, n)`.

**Steps**

1. Implement `Centroid` with a numerically stable weighted-mean update:
   `mean += (x − mean) · (w / (weight + w))`.
2. Implement each scale function and its inverse.
3. Port the exact constants/branching from the reference `ScaleFunction` enum.

**Tests (`test_scale.mojo`)**

- Round-trip identity: `q(k(q)) ≈ q` for `q ∈ {0.001, 0.01, 0.5, 0.99, 0.999}`.
- Monotonicity: `k` is strictly increasing in `q`.
- Tail steepness: verify Δk per Δq is larger near the tails than the middle
  (this is the property that drives tail accuracy).

**Done when:** all four scale functions pass round-trip + monotonicity tests.

---

## Phase 2 — Core MergingDigest (no decay)

Implement the `MergingDigest` variant from the reference (array-backed, no
dynamic allocation on the hot path after construction).

**Deliverables**

- `tdigest.mojo`: `TDigest` struct holding sorted centroids + buffers.
- Operations: `add(x, weight=1.0)`, `merge()` (compress buffer into main),
  `quantile(q)`, `cdf(x)`, `merge_into(other)`.

**Algorithm sketch**

1. **Buffer:** incoming samples accumulate in an unmerged buffer.
2. **Merge pass:** when the buffer fills, sort the union of buffer + centroids
   by mean, then sweep left→right accumulating weight into the current centroid
   until adding more would exceed the k-size limit (`k(q_right) − k(q_left) > 1`);
   then start a new centroid. This is where the scale function restricts
   cluster sizes.
3. **Query:** `quantile(q)` interpolates between centroid means using cumulative
   weights; **singleton centroids at the tails get special interpolation** so the
   extreme quantiles stay sharp (per the reference's "better interpolation").

**Steps**

1. Implement the merge/compress pass parameterized by the chosen scale function.
2. Implement `quantile` with tail-aware interpolation (handle singletons at
   `q=0`/`q=1` explicitly).
3. Implement `cdf` as the inverse interpolation.
4. Implement `merge_into` (concatenate centroids, re-run merge pass).

**Tests (`test_tdigest.mojo`)**

- Accuracy vs. a brute-force exact quantile over N=1e5 samples from:
  normal, uniform, exponential, and a heavy-tailed (Pareto) distribution.
- Assert tail error (p0.1 / p99.9) is within target ppm bounds; assert median
  error within the looser middle bound.
- Merge equivalence: `digest(A ∪ B)` ≈ `digest(A).merge_into(digest(B))`.
- Compression bound: centroid count stays `O(δ)`.

**Done when:** tail quantile errors beat a plain histogram by orders of
magnitude and merge equivalence holds.

---

## Phase 3 — Exponential decay

Add time/age weighting so recent data dominates. This is the second headline
feature.

**Deliverables**

- `decay.mojo`: a `Decay` policy — `Decay.none()`, `Decay.rate(λ)`,
  `Decay.half_life(h)` (with `λ = ln 2 / h`).
- `TDigest` integration: decay applied against either sample count or
  user-supplied timestamps.

**Design decisions**

- **Lazy decay via a running scale factor.** Rather than rescaling every
  centroid on each insert (O(n) per add), maintain a global multiplier
  `S = e^{λ t}` and store each centroid's weight *pre-divided* by the scale at
  insertion time. The effective weight is `stored_weight · e^{−λ t_now}`. This
  keeps `add` O(1) amortized.
- **Overflow guard:** `S` grows unbounded; periodically renormalize all stored
  weights (divide by current `S`, reset `S = 1`) when `S` exceeds a threshold.
- **Pruning:** drop centroids whose effective weight falls below a floor
  `ε` to keep memory bounded on long-running streams.
- **Interaction with the scale function:** total weight `n` (used by the
  normalizer) becomes the *effective* decayed total, so k-size limits adapt as
  old mass evaporates.

**Steps**

1. Add decayed-weight bookkeeping (`S`, per-centroid stored weight).
2. Apply decay in `add` (count-based) and in a `add(x, timestamp=t)` overload
   (time-based).
3. Implement renormalization + pruning.
4. Make `quantile`/`cdf`/`merge` use effective weights.

**Tests (`test_decay.mojo`)**

- **Drift test:** stream from `N(0,1)` for a while, then switch to `N(10,1)`.
  With a short half-life, `quantile(0.5)` should converge toward 10; with
  `Decay.none()` it should stay near the historical blend.
- **Half-life correctness:** after `h` samples of no new data in a region, that
  region's contribution to total weight has halved (within tolerance).
- **Memory bound:** centroid count stays bounded under indefinite streaming with
  pruning enabled.

**Done when:** drifting-stream quantiles track the recent distribution and
memory stays bounded.

---

## Phase 4 — Hardening, benchmarks, docs

**Deliverables**

- Edge cases: empty digest, single point, all-identical values, NaN/inf
  rejection, `q` out of `[0,1]`.
- Serialization: `to_bytes` / `from_bytes` for persisting/merging across
  processes.
- Benchmarks: ns/add and accuracy sweeps across `δ` and scale functions
  (mirrors the reference's quality/benchmark modules). Candidate for the
  `benchmark-model` skill workflow.
- README examples kept in sync with the real API; add an accuracy plot.

**Done when:** public API is stable, documented, benchmarked, and all tests
green.

---

## Phase 5 — GPU acceleration (optional batched backend)

The classic single-point `add` path is sequential and latency-bound — a poor GPU
fit. But several surrounding workloads are excellent fits. This phase adds a GPU
backend *behind the same public API*, leaving the Phase 1–3 CPU implementation as
the correctness reference. Mojo targets GPUs directly (see the
`mojo-gpu-fundamentals` skill), so the port stays in-language.

**Guiding principles**

- **CPU stays the oracle.** The GPU path never replaces the CPU `TDigest`; every
  GPU result is validated against it within an fp tolerance. The sequential
  single-point `add` remains CPU-only.
- **Parity before speed.** Each stage ships a GPU-vs-CPU parity test first; the
  throughput win is the goal but correctness is the gate.
- **Generic kernels.** The kernels here are built from generic primitives
  (`thread_idx`, `barrier`, `warp.sum`, shared memory, `TileTensor`) — not vendor
  intrinsics — so they are not specific to the GPU they were written on and must
  not be gated to one vendor. (Dev box is Apple silicon; CI should also exercise
  NVIDIA/AMD lanes.)
- **Order of attack.** Build single-digest `add_batch` first (it establishes the
  primitives and the parity harness on the simplest shape), *then* the grouped
  multi-digest kernel.

**Float32 path (new decision).** Phase 1–4 are `Float64`. Consumer and
Apple-silicon GPUs deliver substantially higher `Float32` throughput than
`Float64` — on Apple silicon, `Float64` is not natively supported in hardware
and is emulated, so `Float32` is effectively required — so the GPU backend
targets `Float32` first. To avoid forking the math, parameterize the scale
functions and centroid over a `DType` (defaulting to `float64` on CPU, `float32`
on GPU). The CPU reference keeps `Float64`; parity tolerances account for the
fp32 vs fp64 gap (looser than the fp64-vs-fp64 bound, tracked per scale
function). If the GPU path does not pan out, the `DType` parameter is dropped and
we revert to `Float64`.

**Where the GPU does *not* help**

- Single-point streaming `add` at low rates — host↔device transfer dwarfs the work.
- Small digests queried occasionally — a few hundred centroids fit in cache; a CPU
  scan beats a kernel launch. The crossover is throughput (large batches or many
  concurrent digests).

---

### Stage 5a — DType generalization (CPU, no GPU yet) ✅ DONE

Prepare the math core for a device port without touching behavior.

- Parameterize `Centroid` and the `ScaleFunction` implementations over a
  `DType` (`comptime dtype: DType = .float64`). Keep `Float64` as the default so
  all existing tests pass unchanged.
- Add a `Float32` instantiation and a CPU test that the two agree within an fp32
  tolerance on the scale round-trip / monotonicity suite.

**Done when:** `TDigest` still passes all Phase 1–4 tests at `float64`, and a
`float32` instantiation passes a looser-tolerance parity test on CPU.

**Outcome.** `Centroid[dtype: DType = .float64]` and the generic scale structs
`ScaleK0Impl..K3Impl[dtype: DType = .float64]` are DType-parameterized; the
public names `ScaleK0..K3` are Float64 `comptime` aliases of those structs, so
`TDigest[ScaleK1]` and every other call site compile unchanged. The
`ScaleFunction` trait carries an associated `comptime DT: DType`; floating-point
math inside each method is unlocked with `comptime assert DT.is_floating_point()`,
and `_clamp_q` uses a dtype-sized epsilon (1e-7 for fp32, 1e-15 for fp64).
`TDigest` stays Float64-only via `where S.DT == DType.float64`, bridging the
nominal `S.DT` to concrete `Float64` through two tiny `_normalizer`/`_k` wrappers
(justified `rebind`s under the `where` proof). All 53 Phase 1–4 tests pass
unchanged; 10 new float32 parity tests (round-trip, monotonicity, tail
steepness, and a direct f32-vs-f64 forward-map check) pass at atol=1e-4, for 63
tests total.

---

### Stage 5b — GPU primitives library ✅ DONE

The reusable device building blocks, each tested in isolation against a CPU
reference so later stages compose known-good parts.

- `gpu/primitives.mojo`:
  - **Parallel sort** by key (centroid mean) — the dominant merge-pass cost.
  - **Prefix sum (inclusive/exclusive scan)** over weights.
  - **Segmented scan** (scan that resets at segment boundaries) — the key to
    parallel boundary assignment.
  - **Stream compaction** (drop elements failing a predicate) — for pruning.
- A `DeviceContext` guard: all GPU tests `comptime if has_accelerator()` and
  no-op (pass) when no GPU is present, so CI without a GPU stays green.

**Tests (`test_gpu_primitives.mojo`):** each primitive vs. a CPU implementation
on random inputs incl. prime / tile-straddling lengths (13, 63/64/65).

**Done when:** every primitive matches its CPU reference within tolerance.

**Outcome.** Added the `max` package (pulls `max.gpu` + `layout`) as the GPU
dependency and a new `kdigest.gpu` subpackage. All four primitives are
**single-block** (one thread block, shared memory, up to `BLOCK_CAP = 1024`
elements) — exactly the granularity the digest pipeline needs (one block per
digest in 5d, one block per key in 5e); larger arrays tile in the backend.
Algorithms are correctness-first: **bitonic sort** over a power-of-two padded
shared array (keys padded with +inf; a value payload rides along), and
**Hillis-Steele** inclusive/exclusive/segmented scans; compaction is an
exclusive-scan-of-flags + stable scatter. Each GPU entry point has a matching
`cpu_*` reference with identical semantics (used by tests and as the no-GPU
fallback). Host wrappers upload `List`s, launch, and download. `fn` is gone in
this toolchain — kernels are non-raising `def`s; scalar kernel args must be
fixed-width (`Int32`, not `Int`). 7 new tests pass against the CPU references
across lengths {0,1,2,13,31,63,64,65,100,511,512,1000,1024} (fp32 scans within
1e-5 rtol; sort/compaction exact), for **70 tests total**. All tests are gated on
`has_accelerator()` so GPU-less CI stays green.

---

### Stage 5c — Scale functions as device functions ✅ DONE

- Express `k(q)`, `q(k)`, `normalizer(...)` as per-thread device functions
  reusing the exact Phase 1 formulas (one source of truth, two call sites).
- Parity test on-device: evaluate `k`/`q` across a grid of `q` values on the GPU
  and compare against the CPU `ScaleFunction` for all of K0–K3.

**Done when:** on-device scale evaluation matches the CPU trait within fp32
tolerance for all four functions.

**Outcome.** No new math: the DType-generic `ScaleFunction` structs from Phase 5a
compile to device code unchanged, so a kernel just instantiates
`ScaleKnImpl[float32]` and calls `.normalizer()/.k()/.q()` directly — literally
one source of truth, two call sites. `gpu/scale_device.mojo` adds a generic
kernel `_scale_eval_kernel[S: ScaleFunction, Layout]` (bound before
`enqueue_function`) and a host wrapper `gpu_scale_eval[S](qs, compression,
count)` returning `(k(q), q(k(q)))`. As in the CPU digest, the nominal `S.DT` is
bridged to concrete `float32` with `rebind`s under `where S.DT == float32`.
`test_gpu_scale.mojo` compares the device output to the **float32** CPU trait
(apples-to-apples) across a dense q-grid incl. near-tail points for all four
functions: forward `k` within 1e-3, on-device inverse round-trip `q(k(q))` within
2e-3. 4 new tests, **74 total**, all GPU tests gated on `has_accelerator()`.

---

### Stage 5d — Single-digest `add_batch` (first GPU backend)

The core single-digest win: amortize one host→device transfer over a large
buffer, then run the whole compress on-device.

- Public surface: `add_batch(host_buffer)` on the digest, dispatching to a GPU
  kernel when a device is present and the batch exceeds the crossover (else fall
  back to the CPU buffer path).
- Kernel pipeline: **sort** (buffer ∪ existing centroids by mean) → **parallel
  `k`-evaluation** at candidate boundaries → **segmented-scan boundary
  assignment** → write compacted centroids back. Decay, if active, is a scalar
  broadcast-multiply over weights before the pass (reuses the lazy-decay design).

**Tests (`test_gpu_add_batch.mojo`):** GPU `add_batch` vs. CPU `TDigest` on the
same input — centroid count and quantiles at p0.1/p50/p99.9 within fp32
tolerance, across all scale functions and across uniform/normal/exponential/
Pareto; include a decayed-stream parity case.

**Done when:** single-digest GPU results match the CPU reference within tolerance
for all scale functions, with and without decay.

**Outcome.** `TDigest.add_batch(values)` dispatches to the GPU when
`has_accelerator()` and `len >= _GPU_CROSSOVER` (50k, a safe default pending the
Stage 5f crossover sweep), else runs the exact CPU `add` loop. A batch shares one
clock advance of `len(values)` ticks, so count-based decay matches the
sample-at-a-time path; new-sample weights get the current decay scale factor.

The GPU merge uses the **parallel k-bucketing** formulation (chosen over the
sequential sweep): prefix-sum weights → `cluster_id = floor(k(cum/W))` →
head-flag → segmented reduce (Σw, Σw·mean) → compact. This is a different but
equally valid clustering, so it does not reproduce the CPU centroids bit for bit.
Implemented as one fused single-block kernel (`gpu/add_batch.mojo`); the host
wrapper sorts + tiles arbitrary-length batches and reduces partials
hierarchically until a final block fits. The float64 digest drives the float32
GPU scale via a new `ScaleFunction.F32` associated alias (`S.F32` =
`ScaleKnImpl[float32]`), bridged with `rebind` under `where S.F32.DT == float32`.

Parity policy (per the chosen "truth" framing): both the CPU **and** GPU
estimates are checked against the *true* empirical quantile at p0.1/p50/p99.9,
each required to fall inside a per-scale accuracy band. This surfaced a real,
documented property — the parallel merge resolves the extreme tail a bit more
loosely than the sequential sweep, worst for weak-tail scales. Coverage: K1/K2/K3
over uniform/normal/exponential/Pareto; K0 (linear, weakest tails *by design* —
even its CPU digest is ~8× off at Pareto p99.9) tested on uniform where it is
meaningful; a decayed-stream case; and an exact CPU==add-loop check below the
crossover. 6 new tests, **80 total**, GPU tests gated on `has_accelerator()`.

---

### Stage 5e — Grouped multi-digest (the headline win)

One digest *per key* (per metric / customer / sensor / window) = thousands of
independent small digests → **one GPU block per digest**. Mirrors RAPIDS/libcudf
powering Spark's `approx_percentile`. The parallelism is *across* digests, which
sidesteps the single-digest sequential bottleneck entirely.

- Public surface: `GroupedTDigest` ingesting `(key, value)` batches; per-key
  `quantile`/`cdf`; a `to_bytes` per key for shipping.
- Kernel: segment the batch by key, assign one block per key, run the 5d pipeline
  in block-local shared memory, write per-key centroid arrays back.

**Tests (`test_gpu_grouped.mojo`):** per-key GPU digests match per-key CPU
digests on a synthetic multi-key stream (keys with differing distributions);
correctness holds as the number of keys scales.

**Done when:** per-key GPU digests match per-key CPU digests within tolerance,
and correctness is independent of key count.

**Outcome (done):** Implemented and green (84/84 suite).

- `src/kdigest/gpu/grouped.mojo` — `_grouped_compress_kernel` launches **one
  block per key**, each running the exact Stage 5d parallel k-bucketing compress
  on its key's slice in block-local shared memory. Host wrapper
  `gpu_grouped_compress[S]` counting-sorts the `(key, value)` batch into per-key
  contiguous slices, value-sorts each slice, builds a compact one-block-per-key
  launch for keys that fit (≤ `BLOCK_CAP`), and routes oversized keys to the
  Stage 5d single-digest `gpu_compress_batch` (hierarchical tiling) — so
  correctness is independent of *per-key* volume as well as key count.
- `src/kdigest/grouped.mojo` — `GroupedTDigest[S]` holds one `TDigest[S]` per key
  and wraps the GPU result via a new `TDigest._from_centroids` factory, so per-key
  `quantile`/`cdf`/`to_bytes` reuse the full query surface for free. Falls back to
  a per-key CPU `add` loop when no accelerator is present.
- Parity, as in 5d, is judged *against true per-key quantiles* (the parallel
  merge is a valid-but-different clustering than a sequential sweep, so exact
  centroid-count equality with a CPU digest is not the right bar). Each key's GPU
  digest and its CPU control both land within a per-rank accuracy band of truth.
- `test_gpu_grouped.mojo`: 8 keys × 20k, **256 keys × 800** (verifying
  key-count-independence — compresses in ~59 ms), a mixed-size case (one 100k
  oversized key via fallback beside small one-block keys), and the CPU fallback.

---

### Stage 5f — Query fan-out, merge-tree, and crossover benchmark

- **Vectorized `quantile`/`cdf`** over a large query array — one thread per
  query (parallel map).
- **Merge-tree reduction** combining many partial digests (parallel reduction,
  exploits the mergeable property) for map-reduce finalization.
- **Crossover benchmark** (`bench/bench_gpu.mojo`): ns/point for batched GPU vs.
  CPU as a function of batch size, and ns/digest for grouped GPU vs. CPU as a
  function of key count. Identify and document the break-even points; decide
  whether `add_batch` / `GroupedTDigest` auto-dispatch to CPU below crossover
  (Open Design Question #5).

**Done when:** query/merge kernels match the CPU reference, and the benchmark
documents the batch-size and key-count crossovers where GPU beats CPU.

**Outcome (done):** Implemented and green (87/87 suite).

- `src/kdigest/gpu/query.mojo` —
  - `gpu_quantile` / `gpu_cdf`: a **parallel map**, one GPU thread per query,
    interpolating one compressed digest. Centroids (<= `BLOCK_CAP`) are staged
    into shared memory and reused by every thread in the block. The per-thread
    interpolation mirrors `TDigest.quantile` / `.cdf` exactly, so results agree
    with the CPU to fp32 rounding (tests: 1e-4 band on a full p0.1→p99.9 panel
    and a 25-point CDF sweep).
  - `gpu_merge_digests[S]`: the **mergeable reduce** — gather all partial
    digests' centroids and compress the union once via `gpu_compress_batch`.
    This is the associative fold behind map-reduce finalization. Test merges 4
    shard digests and matches a single digest over the concatenated raw data.
- `bench/bench_gpu.mojo` (`pixi run bench-gpu`) — the crossover study, swept over
  batch size (1k→1M) and key count (16→256). **Findings, documented honestly in
  `bench/Benchmark.md` §4:**
  - *Accuracy:* the parallel GPU compress matches the CPU and is **sharper in the
    tail** (p99 rank error ~2e-5–5e-5 vs the CPU sweep's ~1e-4–3e-4).
  - *Speed:* the current correctness-first backend is **~60–1600× slower** than
    the CPU and shows **no crossover** in range — it is setup/transfer-bound
    (per-call `DeviceContext`, host↔device `map_to_host`, serial host-side
    tiling), not compute-bound. The per-call context creation also leaks Metal
    command queues under tight loops.
  - *Verdict / Open Design Question #5:* `add_batch` / `GroupedTDigest` keep
    their conservative CPU fallback (50k crossover for `add_batch`) because the
    GPU does not yet win on throughput. The value today is accuracy and a
    correct, mergeable, GPU-resident data path (compress → query → merge).
    Winning on speed is deferred future work: a **persistent `DeviceContext`**,
    **device-resident buffers** across compress/query/merge, and a **parallel
    multi-block sort** to replace the host-side tiling.

---

**Phase-level done when:** GPU results match the CPU reference within tolerance
across all scale functions (single and grouped), and batched/grouped throughput
beats CPU above the measured crossover points.

---

## Phase 6 — DDSketch backend (guaranteed relative-value error)

t-digest gives empirically excellent tail accuracy but no hard guarantee.
[DDSketch](https://arxiv.org/abs/1908.10693) (Datadog) is its closest sibling
with a **provable relative-value error** bound and **lossless merge**, making it
a strong complement — and a cleaner fit for the Phase 5 GPU fan-out (merging is
just adding bucket counts). This phase adds it as an *alternative backend behind
a shared trait*, mirroring how scale functions are already pluggable.

**Core idea**

Values are placed into logarithmic buckets with index `i = ceil(log_γ(x))`,
where `γ = (1 + α) / (1 − α)` for a target relative accuracy `α`. Every value
then sits within an `α`-relative band of its bucket boundary, so for any `q` the
returned value `v` satisfies `|v − v_true| ≤ α · v_true`. Negative values go in a
mirrored bucket store; zeros (and a configurable near-zero band) are counted
separately.

**Deliverables**

- `quantile_sketch.mojo`: a `QuantileSketch` trait capturing the common surface
  — `add(x, weight)`, `quantile(q)`, `cdf(x)`, `merge_into(other)`,
  `total_weight()`, `to_bytes()`. Make the existing `TDigest` conform (no
  behavior change) so the two backends are interchangeable.
- `ddsketch.mojo`: `DDSketch` with a positive and a negative `BucketStore`
  (a dense array keyed by bucket index with a tracked offset), a zero counter,
  and the mapping `α ↔ γ`.
- Decay integration: reuse the Phase 3 lazy running-scale-factor over bucket
  *counts* (buckets are just weighted counts), including renormalization and
  pruning of buckets whose effective count falls below `ε`.
- Collapsing bound: cap the number of buckets (like the reference's
  `collapsingLowest/Highest` stores) so memory stays bounded on wide ranges;
  collapsing degrades the guarantee gracefully at the far tail only.

**Algorithm sketch**

1. **add:** route by sign, compute `i = ceil(log(|x|) / log(γ))`, increment that
   bucket's (possibly decayed) count; update min/max and total.
2. **quantile:** scan buckets accumulating count until reaching `q · n`, then
   return the bucket's representative value `2·γ^i / (γ + 1)` (the band
   midpoint), negated for the negative store.
3. **cdf:** inverse scan.
4. **merge_into:** align bucket-index offsets and add counts — exact, no
   re-binning.

**Tests (`test_ddsketch.mojo`)**

- **Guarantee check:** for `α = 0.01`, assert every estimated quantile is within
  `α` *relative* error of the brute-force exact quantile across uniform, normal,
  exponential, and Pareto — a stronger assertion than t-digest's empirical
  bound.
- **Lossless merge:** `ddsketch(A ∪ B)` is *bucket-for-bucket identical* to
  `ddsketch(A).merge_into(ddsketch(B))` (not just approximately equal).
- **Decay drift:** same `N(0,1) → N(10,1)` drift test as Phase 3.
- **Memory bound:** bucket count stays within the collapsing cap under a
  wide-range stream.
- **Trait interchange:** the same generic test driver runs over both `TDigest`
  and `DDSketch` through `QuantileSketch`.

**Done when:** DDSketch meets its relative-error guarantee on all test
distributions, merges losslessly, and is drop-in via the `QuantileSketch` trait.

---

## Phase 7 — KLL sketch (provable rank error, optimal space)

For workloads needing a *provable rank-error* bound (`ε` such that the returned
item's true rank is within `εN`) on arbitrary ordered data — no value-range or
sign assumptions — [KLL](https://arxiv.org/abs/1603.05346) is the
asymptotically space-optimal comparison-based sketch and the basis of Apache
DataSketches. It is a genuinely different structure (randomized compactors), so
it ships as a third backend under the same `QuantileSketch` trait.

**Core idea**

A stack of **compactors** (sorted buffers), one per level. When a level fills,
it *compacts*: sort, then keep either the even- or odd-indexed items (chosen by a
fair coin) and promote them to the next level with doubled weight. Lower levels
use smaller capacities (geometric decay with factor `c ≈ 2/3`), concentrating
resolution where it is cheap. Space is
`O((1/ε) · log log (1/ε))`, with randomized rank error `εN` w.h.p.

**Deliverables**

- `kll.mojo`: `KLLSketch` holding a `List` of compactor levels, the per-level
  capacity schedule derived from a target `k` (⇔ `ε`), and the running item
  count.
- Operations: `add(x, weight)` (unit weight; fractional/weighted via repetition
  or a weighted-KLL extension noted below), `quantile(q)`, `cdf(x)`,
  `merge_into(other)` (level-wise compactor merge), `to_bytes()`.
- A deterministic-seed RNG hook so tests are reproducible.

**Algorithm sketch**

1. **add:** push into level 0; if full, compact upward, cascading as needed.
2. **quantile/cdf:** gather all retained items with their level weights
   (`2^level`), sort by value, and walk the weighted cumulative distribution.
3. **merge_into:** concatenate same-level compactors and re-compact overflowing
   levels; grow the level stack to the max of the two.

**Open sub-questions (resolve in-phase)**

- **Weighted/decayed inputs.** KLL is natively unit-weight. Options: (a) expand
  integer weights, (b) a weighted-KLL variant, (c) document KLL as the
  unweighted/undecayed backend and keep decay a t-digest/DDSketch feature. Pick
  one; (c) is the likely default.

**Tests (`test_kll.mojo`)**

- **Rank-error guarantee:** over many trials with a fixed seed, assert the
  empirical rank error stays within `εN` at the target failure probability for
  uniform/normal/exponential/Pareto.
- **Merge equivalence:** `kll(A ∪ B)` ≈ `kll(A).merge_into(kll(B))` within the
  rank bound.
- **Space bound:** total retained items match the `k`-derived capacity schedule.
- **Trait interchange:** runs through the shared `QuantileSketch` driver.

**Done when:** KLL meets its rank-error guarantee within the target failure
probability, merges correctly, and is drop-in via the `QuantileSketch` trait.

---

## Open design questions

1. **MergingDigest vs. AVLTreeDigest.** Plan targets `MergingDigest` (simpler,
   allocation-free hot path). An AVL-tree variant could follow if we need
   per-insert queries without a merge pass.
2. **Decay semantics for `merge_into`.** Merging two digests with different time
   origins needs a shared clock or a rebasing step — define this precisely in
   Phase 3.
3. **Asymmetric / single-tail scale functions.** The reference notes these are
   promising for one-sided accuracy; consider as a later extension.
4. **Generic element type.** Start with `Float64`; a `DType` parameter for
   `Float32` streams is adopted in Phase 5 (Stage 5a) to ease the GPU port. CPU
   reference stays `Float64`; GPU targets `Float32` first. Drop the parameter and
   revert to `Float64` if the GPU path is abandoned.
5. **GPU crossover threshold.** Determine the batch size / digest count at which
   the Phase 5 GPU path beats CPU (measured in Stage 5f), and whether `add_batch`
   / `GroupedTDigest` auto-dispatch to CPU below crossover.
6. **Error semantics across backends.** t-digest = empirical tail accuracy;
   DDSketch = guaranteed relative-*value* error; KLL = guaranteed *rank* error.
   The `QuantileSketch` trait unifies the API but not the guarantee — document
   per-backend semantics and let callers choose by requirement.
7. **Decay × backend.** Exponential decay is natural for t-digest and DDSketch
   (weighted counts) but awkward for KLL (unit-weight compactors). Decide whether
   decay is a backend-specific capability or a uniform promise (see Phase 7).
8. **Hard sliding window vs. smooth decay.** Current decay smoothly down-weights
   the past; it cannot express "exactly the last N / last T". If users need a
   hard window, that is a separate mechanism (exponential-histogram style), not a
   decay rate — scope it only if demand appears.

---

## Milestone summary

| Phase | Outcome | Headline feature |
| ----- | ------- | ---------------- |
| 0 | Compiling scaffold + CI task | — |
| 1 | Centroid + 4 scale functions, tested | Pluggable scale functions |
| 2 | Working t-digest with tail-accurate quantiles | Extreme-quantile accuracy |
| 3 | Exponential decay (rate / half-life) | Recency weighting |
| 4 | Hardened, serialized, benchmarked | Production readiness |
| 5 | Batched + grouped GPU backend behind same API | GPU acceleration |
| 6 | DDSketch backend behind a `QuantileSketch` trait | Guaranteed relative-value error |
| 7 | KLL sketch backend behind the same trait | Provable rank error, optimal space |
