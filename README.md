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

> Status: **early design / scaffolding.** This repo currently contains the
> project toolchain (pixi + Mojo), this README, and a phased build plan in
> [`PLAN.md`](./PLAN.md). The API shown below is the target design, not yet a
> shipping implementation.

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
struct Centroid(Copyable, Movable):
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

## Target API

```mojo
from kdigest import TDigest, ScaleK2, Decay

def main() raises:
    # Compression (delta): higher = more centroids = more accuracy.
    var digest = TDigest(
        compression=200.0,
        scale=ScaleK2(),
        decay=Decay.half_life(10_000.0),   # influence halves every 10k samples
    )

    # Stream data in
    for x in stream:
        digest.add(x)                      # weight 1.0
        # digest.add(x, weight=3.0)        # weighted sample
        # digest.add(x, timestamp=t)       # time-based decay

    # Query quantiles — accurate even at the extremes
    var p50  = digest.quantile(0.50)
    var p999 = digest.quantile(0.999)
    var p001 = digest.quantile(0.001)

    # Inverse query: what fraction of mass is <= x?
    var cdf  = digest.cdf(x)

    # Merge two digests (map-reduce friendly)
    other.merge_into(digest)

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

## Project layout (planned)

```
kdigest/
├── README.md              # this file
├── PLAN.md                # phased implementation plan
├── pixi.toml              # Mojo toolchain (Modular MAX channel)
├── scripts/
│   └── run_tests.sh           # test runner (mojo run over test/test_*.mojo)
├── src/
│   └── kdigest/
│       ├── centroid.mojo      # Centroid value type                    [done]
│       ├── scale.mojo         # ScaleFunction trait + K0/K1/K2/K3       [done]
│       ├── decay.mojo         # Decay policy (rate / half-life)         [planned]
│       ├── tdigest.mojo       # MergingDigest core (add/merge/quantile) [planned]
│       └── __init__.mojo      # public exports
└── test/
    ├── test_centroid.mojo     # weighted-mean correctness
    ├── test_scale.mojo        # scale function identities & inverses
    ├── test_tdigest.mojo      # accuracy vs. exact quantiles            [planned]
    └── test_decay.mojo        # decay behavior on drifting streams      [planned]
```

---

## Building & testing

This project uses [`pixi`](https://pixi.sh) with the Modular MAX conda channel.

```bash
pixi install          # resolve the Mojo toolchain
pixi run test         # run the test suite (mojo run over test/test_*.mojo)
pixi run fmt          # format sources
pixi run build        # verify the package compiles
```

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
