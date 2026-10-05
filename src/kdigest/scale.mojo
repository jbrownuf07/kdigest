"""Scale functions k(q): the heart of the t-digest.

A scale function maps a quantile q in [0, 1] to a "k-scale" coordinate. During
the merge pass, a cluster is allowed to grow only until its span in k-space
reaches 1.0, i.e. while  k(q_right) - k(q_left) <= 1. Because these functions
are steep near q=0 and q=1, tail clusters are forced to stay small (often a
single sample), which yields accurate tail quantiles.

Each scale function provides:
  - normalizer(compression, n): a scalar derived from the compression
    parameter (delta) and the total weight n, folded into k() and q().
  - k(q, normalizer):   forward map, quantile -> k-scale.
  - q(k, normalizer):   inverse map, k-scale -> quantile.

The four functions mirror the reference implementation
(https://github.com/tdunning/t-digest):
  K0  linear in q                      (uniform cluster sizes)
  K1  (1/2pi) * asin(2q - 1)           (classic t-digest)
  K2  log(q/(1-q)) with a count term   (proven size bounds)
  K3  piecewise log variant of K2      (aggressive tail control)

DType parameterization (Phase 5a). Each scale function is generic over a
`DType` so the same math serves the Float64 CPU reference and the Float32 GPU
backend. The four public names `ScaleK0..K3` are Float64 aliases of the generic
`ScaleK0Impl..K3Impl` structs, so existing `TDigest[ScaleK1]` call sites are
unchanged; use e.g. `ScaleK1Impl[DType.float32]` for the Float32 instantiation.
"""

from std.math import asin, sin, log, exp, pi


def _q_eps[dtype: DType]() -> Scalar[dtype]:
    """Clamp distance from the 0/1 singularities, sized to the dtype's
    precision: tiny for fp64, coarser for fp32 (whose epsilon is ~1e-7)."""
    comptime assert dtype.is_floating_point()
    comptime if dtype == DType.float32:
        return 1e-7
    return 1e-15


def _clamp_q[dtype: DType](q: Scalar[dtype]) -> Scalar[dtype]:
    var eps = _q_eps[dtype]()
    if q < eps:
        return eps
    if q > 1.0 - eps:
        return 1.0 - eps
    return q


trait ScaleFunction(Deinitable, ImplicitlyCopyable, Movable):
    """A t-digest scale function and its inverse, generic over `DT`."""

    comptime DT: DType
    # The Float32 counterpart of this scale function (same formulas, fp32). Used
    # to drive the GPU backend from a Float64 digest without hard-coding which
    # Kn is in play.
    comptime F32: ScaleFunction

    def __init__(out self):
        ...

    def normalizer(
        self, compression: Scalar[Self.DT], n: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        ...

    def k(
        self, q: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        ...

    def q(
        self, k: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        ...


@fieldwise_init
struct ScaleK0Impl[dtype: DType = DType.float64](ScaleFunction):
    """Linear scale: k = normalizer * q. Uniform cluster sizes, weakest tails.
    """

    comptime DT = Self.dtype
    comptime F32 = ScaleK0Impl[DType.float32]

    def normalizer(
        self, compression: Scalar[Self.DT], n: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        return compression / 2.0

    def k(
        self, q: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        return normalizer * q

    def q(
        self, k: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        return k / normalizer


@fieldwise_init
struct ScaleK1Impl[dtype: DType = DType.float64](ScaleFunction):
    """Classic t-digest scale: k = normalizer * asin(2q - 1)."""

    comptime DT = Self.dtype
    comptime F32 = ScaleK1Impl[DType.float32]

    def normalizer(
        self, compression: Scalar[Self.DT], n: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        return compression / (2.0 * pi)

    def k(
        self, q: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        return normalizer * asin(2.0 * q - 1.0)

    def q(
        self, k: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        return (sin(k / normalizer) + 1.0) / 2.0


@fieldwise_init
struct ScaleK2Impl[dtype: DType = DType.float64](ScaleFunction):
    """Log-odds scale: k = normalizer * log(q / (1 - q)).

    The normalizer includes a count term 4*log(n/compression) + 24 so that the
    k-scale adapts to the amount of data seen, giving provably tight size
    bounds. The inverse is the logistic function.
    """

    comptime DT = Self.dtype
    comptime F32 = ScaleK2Impl[DType.float32]

    def normalizer(
        self, compression: Scalar[Self.DT], n: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        return compression / (4.0 * log(n / compression) + 24.0)

    def k(
        self, q: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        var qc = _clamp_q(q)
        return normalizer * log(qc / (1.0 - qc))

    def q(
        self, k: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        var w = exp(k / normalizer)
        return w / (1.0 + w)


@fieldwise_init
struct ScaleK3Impl[dtype: DType = DType.float64](ScaleFunction):
    """Piecewise log scale, most aggressive tail control.

    k = normalizer * log(2q)        for q <= 0.5
    k = normalizer * -log(2(1-q))   for q >  0.5
    """

    comptime DT = Self.dtype
    comptime F32 = ScaleK3Impl[DType.float32]

    def normalizer(
        self, compression: Scalar[Self.DT], n: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        return compression / (4.0 * log(n / compression) + 21.0)

    def k(
        self, q: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        var qc = _clamp_q(q)
        if qc <= 0.5:
            return normalizer * log(2.0 * qc)
        return -normalizer * log(2.0 * (1.0 - qc))

    def q(
        self, k: Scalar[Self.DT], normalizer: Scalar[Self.DT]
    ) -> Scalar[Self.DT]:
        comptime assert Self.DT.is_floating_point()
        if k <= 0.0:
            return exp(k / normalizer) / 2.0
        return 1.0 - exp(-k / normalizer) / 2.0


# Float64 aliases: the public, default-precision scale functions. Existing
# `TDigest[ScaleK1]` call sites bind to these unchanged.
comptime ScaleK0 = ScaleK0Impl[DType.float64]
comptime ScaleK1 = ScaleK1Impl[DType.float64]
comptime ScaleK2 = ScaleK2Impl[DType.float64]
comptime ScaleK3 = ScaleK3Impl[DType.float64]
