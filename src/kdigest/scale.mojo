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
"""

from std.math import asin, sin, log, exp, pi


# Clamp used by the log-based functions to avoid the singularities at q=0/1.
comptime _Q_EPS = 1e-15


def _clamp_q(q: Float64) -> Float64:
    if q < _Q_EPS:
        return _Q_EPS
    if q > 1.0 - _Q_EPS:
        return 1.0 - _Q_EPS
    return q


trait ScaleFunction(Copyable, Movable):
    """A t-digest scale function and its inverse."""

    def normalizer(self, compression: Float64, n: Float64) -> Float64:
        ...

    def k(self, q: Float64, normalizer: Float64) -> Float64:
        ...

    def q(self, k: Float64, normalizer: Float64) -> Float64:
        ...


@fieldwise_init
struct ScaleK0(ScaleFunction):
    """Linear scale: k = normalizer * q. Uniform cluster sizes, weakest tails.
    """

    def normalizer(self, compression: Float64, n: Float64) -> Float64:
        return compression / 2.0

    def k(self, q: Float64, normalizer: Float64) -> Float64:
        return normalizer * q

    def q(self, k: Float64, normalizer: Float64) -> Float64:
        return k / normalizer


@fieldwise_init
struct ScaleK1(ScaleFunction):
    """Classic t-digest scale: k = normalizer * asin(2q - 1)."""

    def normalizer(self, compression: Float64, n: Float64) -> Float64:
        return compression / (2.0 * pi)

    def k(self, q: Float64, normalizer: Float64) -> Float64:
        return normalizer * asin(2.0 * q - 1.0)

    def q(self, k: Float64, normalizer: Float64) -> Float64:
        return (sin(k / normalizer) + 1.0) / 2.0


@fieldwise_init
struct ScaleK2(ScaleFunction):
    """Log-odds scale: k = normalizer * log(q / (1 - q)).

    The normalizer includes a count term 4*log(n/compression) + 24 so that the
    k-scale adapts to the amount of data seen, giving provably tight size
    bounds. The inverse is the logistic function.
    """

    def normalizer(self, compression: Float64, n: Float64) -> Float64:
        return compression / (4.0 * log(n / compression) + 24.0)

    def k(self, q: Float64, normalizer: Float64) -> Float64:
        var qc = _clamp_q(q)
        return normalizer * log(qc / (1.0 - qc))

    def q(self, k: Float64, normalizer: Float64) -> Float64:
        var w = exp(k / normalizer)
        return w / (1.0 + w)


@fieldwise_init
struct ScaleK3(ScaleFunction):
    """Piecewise log scale, most aggressive tail control.

    k = normalizer * log(2q)        for q <= 0.5
    k = normalizer * -log(2(1-q))   for q >  0.5
    """

    def normalizer(self, compression: Float64, n: Float64) -> Float64:
        return compression / (4.0 * log(n / compression) + 21.0)

    def k(self, q: Float64, normalizer: Float64) -> Float64:
        var qc = _clamp_q(q)
        if qc <= 0.5:
            return normalizer * log(2.0 * qc)
        return -normalizer * log(2.0 * (1.0 - qc))

    def q(self, k: Float64, normalizer: Float64) -> Float64:
        if k <= 0.0:
            return exp(k / normalizer) / 2.0
        return 1.0 - exp(-k / normalizer) / 2.0
