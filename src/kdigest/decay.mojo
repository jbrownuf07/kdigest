"""Exponential decay policy for recency-weighted t-digests (Phase 3).

A `Decay` describes how fast old mass evaporates. A sample observed at time `t_i`
contributes effective weight

    w · e^(-λ · (t_now - t_i))

at a later time `t_now`, where `λ >= 0` is the decay rate. Larger `λ` means the
digest forgets faster. `λ = 0` recovers the undecayed Phase 2 behavior.

"Time" is whatever clock the digest advances: either the sample count (one tick
per `add`) or a user-supplied timestamp. Both reduce to the same math.

Construct a policy with one of:
    Decay.none()          # λ = 0, no forgetting
    Decay.rate(λ)         # explicit rate
    Decay.half_life(h)    # mass halves every h time units: λ = ln 2 / h
"""

from std.math import log


@fieldwise_init
struct Decay(ImplicitlyCopyable, Movable, Writable):
    """An exponential-decay rate `λ` (per unit of the digest's clock)."""

    var rate_lambda: Float64

    @staticmethod
    def none() -> Decay:
        """No decay: all samples keep full weight forever (λ = 0)."""
        return Decay(rate_lambda=0.0)

    @staticmethod
    def rate(lambda_: Float64) -> Decay:
        """Decay at an explicit rate `λ >= 0` per unit time."""
        return Decay(rate_lambda=lambda_)

    @staticmethod
    def half_life(h: Float64) raises -> Decay:
        """Decay such that effective weight halves every `h` time units.

        λ = ln 2 / h. Requires h > 0.
        """
        if h <= 0.0:
            raise Error("half_life must be positive")
        return Decay(rate_lambda=log(2.0) / h)

    def is_active(self) -> Bool:
        """True when any forgetting happens (λ > 0)."""
        return self.rate_lambda > 0.0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Decay(lambda=", self.rate_lambda, ")")
