# kdigest

**Accurate streaming quantiles at the extremes — with time decay.**

`kdigest` is a [Mojo](https://docs.modular.com/mojo/) library for estimating
quantiles over unbounded data streams using the
[**t-digest**](https://github.com/tdunning/t-digest) data structure. It is
optimized for the hard part of the distribution: the **extreme tails** (near
\(q = 0\) and \(q = 1\)), where naive sketches lose accuracy fastest.

On top of the standard t-digest, `kdigest` adds two capabilities that make
it well suited to real-time, non-stationary streams:

1. **Exponential decay of centroid weights** — recent observations count more
   than old ones, so the estimated quantiles track a drifting distribution.
2. **Pluggable scale functions** — choose the mathematical scaling function
   (`k_0`, `k_1`, `k_2`, `k_3`) that controls how cluster sizes are restricted,
   trading median accuracy for tail accuracy.

> Status: **Phases 0–4 implemented.** The library ships the `MergingDigest`
> core (`add`/`merge`/`quantile`/`cdf`/`merge_into`), four scale functions,
> exponential decay (rate / half-life), serialization, edge-case hardening, and
> a benchmark suite. GPU acceleration (Phase 5) is the remaining optional work;
> see [`PLAN.md`](./PLAN.md). The API below reflects the real shipping API.

---

## Why t-digest?

A t-digest is a compact summary of a distribution built from a set of
**centroids**. Each centroid is a `(mean, weight)` pair summarizing a cluster of
nearby samples. The key insight is that clusters are allowed to be large in the
middle of the distribution but are forced to stay small near the tails. This is
what gives t-digests their signature property:

- **Part-per-million accuracy for extreme quantiles** (e.g. p99.9, p0.1)
- **< ~1000 ppm accuracy for middle quantiles** (e.g. the median)
- **Bounded memory** regardless of stream length
- **Mergeable** — two digests can be combined, enabling map-reduce / parallel
  aggregation

The size restriction on clusters is governed by a **scale function** \(k(q)\),
which maps a quantile \(q \in [0, 1]\) to a "k-scale" coordinate. A cluster is
allowed to grow only until its span in k-space reaches 1.0. Because the scale
function is steep near \(q=0\) and \(q=1\), tail clusters stay small (often a
single sample), which is exactly what yields accurate tail estimates.

---

## Core concepts

### Centroid

```mojo
struct Centroid(ImplicitlyCopyable, Movable, Writable):
    var mean: Float64    # cluster mean
    var weight: Float64  # effective count (may be fractional after decay)
```

### Scale functions

The scale function `k(q, delta, n)` and its inverse are the "heart of the
t-digest" ([tdunning/t-digest](https://github.com/tdunning/t-digest)). We expose
the four standard functions from the reference implementation as an enum-like
trait so users can pick the accuracy trade-off that fits their workload:

| Function | Behavior | Best for |
| -------- | -------- | -------- |
| `K0` | Linear in \(q\). Uniform cluster sizes. | Simple, fast; weakest tails. |
| `K1` | \(k \propto \arcsin(2q-1)\). Classic t-digest. | Balanced accuracy. |
| `K2` | Uses \(\log\) of \(q/(1-q)\) with a count term. | Strong, proven size bounds. |
| `K3` | Piecewise variant of `K2`. | Aggressive tail control. |

```mojo
trait ScaleFunction:
    # Forward: quantile q -> k-scale coordinate
    def k(self, q: Float64, normalizer: Float64) -> Float64: ...
    # Inverse: k-scale coordinate -> quantile q
    def q(self, k: Float64, normalizer: Float64) -> Float64: ...
    # Normalizer derived from compression (delta) and total weight (n)
    def normalizer(self, compression: Float64, n: Float64) -> Float64: ...
```

### Exponential decay

To favor recent data, every existing centroid weight is multiplied by a decay
factor before (or as) new data arrives. Given a decay rate \(\lambda\) and
elapsed time \(\Delta t\) (or elapsed sample count), the weight update is:

\[
w \leftarrow w \cdot e^{-\lambda \, \Delta t}
\]

Equivalently, you can specify a **half-life** \(h\): the time (or number of
samples) after which an observation's influence halves. The two are related by
\(\lambda = \ln(2)/h\).

Because weights become fractional and shrink over time, decay is implemented
lazily and the digest periodically prunes centroids whose weight has decayed
below a floor, keeping memory bounded.

---

## Usage

The scale function is a compile-time type parameter; decay and compression are
runtime arguments.

```mojo
from kdigest import TDigest, ScaleK2, Decay

def main() raises:
    # Compression (delta): higher = more centroids = more accuracy.
    var digest = TDigest[ScaleK2](
        compression=200.0,
        decay=Decay.half_life(10_000.0),   # influence halves every 10k samples
    )

    # Stream data in (add raises on NaN/inf or non-positive weight)
    for x in stream:
        digest.add(x)                      # weight 1.0, one clock tick
        # digest.add(x, weight=3.0)        # weighted sample
        # digest.add_at(x, timestamp=t)    # time-based decay

    # Query quantiles — accurate even at the extremes
    var p50 = digest.quantile(0.50)
    var p999 = digest.quantile(0.999)
    var p001 = digest.quantile(0.001)

    # Inverse query: what fraction of mass is <= x?
    var cdf = digest.cdf(5.0)

    # Merge two digests (map-reduce friendly)
    other.merge_into(digest)

    # Persist / ship across processes
    var bytes = digest.to_bytes()
    var restored = TDigest[ScaleK2].from_bytes(bytes)

    print("p99.9 =", p999)
```

### Choosing parameters

- **`compression` (δ):** Controls the number of centroids (≈ `δ`). Typical
  values: 100–500. Higher = more accurate and larger.
- **`scale`:** `ScaleK1` is the balanced default; `ScaleK2`/`ScaleK3` give the
  tightest, provably-bounded tails and are recommended when you care most about
  extreme quantiles.
- **`decay`:** Omit for a classic (all-history) t-digest. Use
  `Decay.half_life(h)` or `Decay.rate(λ)` to emphasize recent data.

---

## Project layout

```
kdigest/
├── README.md              # this file
├── PLAN.md                # phased implementation plan
├── pixi.toml              # Mojo toolchain (Modular MAX channel)
├── scripts/
│   └── run_tests.sh           # test runner (mojo run over test/test_*.mojo)
├── bench/
│   └── bench.mojo             # throughput + accuracy benchmarks         [done]
├── src/
│   └── kdigest/
│       ├── centroid.mojo      # Centroid value type                     [done]
│       ├── scale.mojo         # ScaleFunction trait + K0/K1/K2/K3        [done]
│       ├── decay.mojo         # Decay policy (rate / half-life)          [done]
│       ├── tdigest.mojo       # MergingDigest core + serialization       [done]
│       └── __init__.mojo      # public exports
└── test/
    ├── test_centroid.mojo     # weighted-mean correctness
    ├── test_scale.mojo        # scale function identities & inverses
    ├── test_tdigest.mojo      # accuracy vs. exact quantiles             [done]
    ├── test_decay.mojo        # decay behavior on drifting streams       [done]
    └── test_edge.mojo         # edge cases + serialization round-trip    [done]
```

---

## Building & testing

This project uses [`pixi`](https://pixi.sh) with the Modular MAX conda channel.

```bash
pixi install          # resolve the Mojo toolchain
pixi run test         # run the test suite (mojo run over test/test_*.mojo)
pixi run fmt          # format sources
pixi run build        # verify the package compiles
pixi run bench        # throughput + accuracy benchmarks
```

### Benchmark snapshot

Measured over N(0,1) on an Apple-silicon dev box (`pixi run bench`; indicative,
not a committed SLA):

| Scale | δ | ns/add | p0.1 rank err | p50 rank err | p99.9 rank err |
| ----- | --- | ------ | ------------- | ------------ | -------------- |
| K0 | 100 | ~76 | 9.6e-4 | 2.8e-4 | 1.0e-3 |
| K1 | 100 | ~89 | 1.1e-4 | 1.9e-4 | 1.3e-4 |
| K2 | 100 | ~90 | 2.0e-5 | 9.0e-4 | 5.0e-6 |
| K2 | 200 | ~140 | 1.5e-5 | 1.1e-4 | 5.0e-6 |

K2/K3 drive tail error into the single-digit-ppm range (trading some median
accuracy); K0 (linear) has the weakest tails. This is the t-digest trade-off the
scale function selects.

See [`PLAN.md`](./PLAN.md) for the step-by-step build-out.

---

## References

- Dunning & Ertl, *Computing Extremely Accurate Quantiles Using t-Digests* —
  [tdunning/t-digest](https://github.com/tdunning/t-digest) (reference
  implementation, scale functions, proofs, and the `MergingDigest` /
  `AVLTreeDigest` designs).
- Scale functions and tail-accuracy discussion:
  [t-digest README — Scale Functions](https://github.com/tdunning/t-digest#scale-functions).

## License

[Apache-2.0](./LICENSE) — matching the
[tdunning/t-digest](https://github.com/tdunning/t-digest) reference
implementation.
