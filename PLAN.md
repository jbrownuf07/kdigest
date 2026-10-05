# stream-fire — Implementation Plan

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

- `src/stream_fire/` package with an empty `__init__.mojo`.
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

**Where the GPU wins**

1. **Batched merge pass (core win).** The merge pass is naturally batch-oriented:
   - **Parallel sort** of buffer ∪ centroids by mean (GPU radix/merge sort) — the
     single most expensive step.
   - **Parallel scale-function evaluation** — `k(q)` (`asin`/`log` for
     `K1`/`K2`/`K3`) is embarrassingly parallel, one thread per candidate boundary.
   - **Prefix-sum (scan)** over weights for cumulative quantile positions.
   - Boundary assignment, though sequentially dependent, reformulates as a
     **segmented scan** so it parallelizes too.
2. **Grouped / multi-digest (often the biggest real-world win).** One digest *per
   key* (per metric / customer / sensor / window) = thousands of independent small
   digests → one GPU block per digest. This mirrors RAPIDS/libcudf powering Spark's
   `approx_percentile`.
3. **Query fan-out.** `quantile()` / `cdf()` over a large array of query points is a
   parallel map — one thread per query.
4. **Merge trees for map-reduce.** Combining many partial digests is a parallel
   reduction tree, exploiting the mergeable property.

**Decay interaction (GPU-friendly).** The lazy-decay design survives the port
cleanly: applying decay is a scalar broadcast-multiply over centroid weights;
renormalization is a parallel map; pruning (weights below ε) is stream compaction.

**Where the GPU does *not* help**

- Single-point streaming `add` at low rates — host↔device transfer dwarfs the work.
- Small digests queried occasionally — a few hundred centroids fit in cache; a CPU
  scan beats a kernel launch. The crossover is throughput (large batches or many
  concurrent digests).

**Deliverables**

- `add_batch(host_buffer)` — device-side sort + scale-eval + segmented-scan merge.
- Grouped API — one GPU block per key for many independent digests.
- Vectorized `quantile`/`cdf` over query arrays.
- GPU primitives: parallel sort, prefix/segmented scan, stream compaction.

**Steps**

1. Implement scale functions as per-thread device functions (reuse Phase 1 math).
2. Implement the batched merge kernel (sort → k-eval → segmented-scan boundaries).
3. Add the grouped multi-digest kernel.
4. Add vectorized query kernels and the parallel merge-tree reduction.

**Tests**

- GPU vs. CPU parity: identical centroids/quantiles for the same input (within fp
  tolerance) across all scale functions.
- Grouped correctness: per-key GPU digests match per-key CPU digests.
- Throughput benchmark: ns/point for batched GPU vs. CPU; identify the crossover
  batch size.

**Done when:** GPU results match the CPU reference within tolerance and batched
throughput beats CPU above the crossover point.

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
4. **Generic element type.** Start with `Float64`; consider a `DType` parameter
   later for `Float32` streams (also eases the GPU port in Phase 5).
5. **GPU crossover threshold.** Determine the batch size / digest count at which
   the Phase 5 GPU path beats CPU, and whether to auto-dispatch based on it.

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
