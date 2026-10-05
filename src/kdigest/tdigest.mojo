"""Core MergingDigest: array-backed t-digest with tail-accurate quantiles.

This is the Phase 2 variant from the reference (tdunning/t-digest): samples
accumulate in an unmerged buffer, and when the buffer fills we sort the union of
buffer + existing centroids by mean and sweep left -> right, accumulating weight
into the current centroid until adding more would exceed the k-size limit
    (k(q_right) - k(q_left) > 1). The steep tails of the scale function force
the extreme clusters to stay small (often singletons), which is what keeps the
extreme quantiles sharp.

The digest is parameterized over a `ScaleFunction` (K0..K3). All weights are
`Float64` so Phase 3 (exponential decay) can carry fractional weights without a
representation change.

Example:
    var d = TDigest[ScaleK1](compression=100.0)
    for x in samples:
        d.add(x)
    var p999 = d.quantile(0.999)
"""

from std.math import exp

from kdigest.centroid import Centroid
from kdigest.decay import Decay
from kdigest.scale import ScaleFunction, ScaleK1


# Buffer size as a multiple of compression. The reference uses a buffer several
# times the centroid budget so each merge pass amortizes well.
comptime _BUFFER_FACTOR = 5

# Renormalize all stored weights once the running decay exponent (λ·t) grows
# past this, to keep e^(λ·t) far from Float64 overflow (~709).
comptime _RENORM_LOG_THRESHOLD = 400.0

# Effective-weight floor: centroids lighter than this are pruned after a merge
# when decay is active, bounding memory on indefinite streams.
comptime _PRUNE_EPSILON = 1e-6


def _sort_centroids(mut cs: List[Centroid]):
    """In-place sort of centroids by mean (insertion sort).

    Insertion sort is used because after a merge the kept centroids are already
    sorted and only the freshly buffered points are out of order, so runs are
    nearly sorted; a reference-quality fast sort is a Phase 4/5 concern.
    """
    for i in range(1, len(cs)):
        var key = cs[i]
        var j = i - 1
        while j >= 0 and cs[j].mean > key.mean:
            cs[j + 1] = cs[j]
            j -= 1
        cs[j + 1] = key


struct TDigest[S: ScaleFunction](Copyable, Movable, Sized):
    """A merging t-digest with a pluggable scale function `S`.

    When a `Decay` with λ > 0 is supplied, centroid weights decay exponentially
    with the digest's clock. Decay is applied lazily: each centroid stores its
    weight scaled by e^(λ·t_insert), and the effective weight at query time is
    `stored · e^(-λ·t_now)`. The running exponent `λ·t_now` is tracked in
    `_log_scale`; `add` stays O(1) amortized. We renormalize when the exponent
    grows large (overflow guard) and prune centroids below an effective-weight
    floor (memory bound).
    """

    var compression: Float64
    var _scale: Self.S
    var _decay: Decay
    # Merged, mean-sorted centroids. Their `weight` fields hold *scaled* weights
    # (actual · e^(λ·t_insert)) when decay is active, raw weights otherwise.
    var _centroids: List[Centroid]
    # Unmerged incoming samples awaiting the next merge pass (also scaled).
    var _buffer: List[Centroid]
    var _buffer_limit: Int
    # Sum of scaled weights across centroids + buffer; effective total is
    # `_stored_total · e^(-_log_scale)`.
    var _stored_total: Float64
    # Running decay exponent λ·t_now.
    var _log_scale: Float64
    # Current clock value t_now (sample count or last timestamp).
    var _clock: Float64
    var _min: Float64
    var _max: Float64

    def __init__(
        out self,
        compression: Float64 = 100.0,
        scale: Self.S = Self.S(),
        decay: Decay = Decay.none(),
    ):
        self.compression = compression
        self._scale = scale
        self._decay = decay
        self._centroids = List[Centroid]()
        self._buffer = List[Centroid]()
        self._buffer_limit = Int(compression) * _BUFFER_FACTOR
        self._stored_total = 0.0
        self._log_scale = 0.0
        self._clock = 0.0
        self._min = 0.0
        self._max = 0.0

    def add(mut self, value: Float64, weight: Float64 = 1.0):
        """Add one (optionally weighted) sample, advancing the clock by one tick.

        Use this for count-based decay (one unit of time per sample).
        """
        self._advance_clock(self._clock + 1.0)
        self._insert(value, weight)

    def add_at(
        mut self, value: Float64, timestamp: Float64, weight: Float64 = 1.0
    ):
        """Add a sample observed at `timestamp` for time-based decay.

        Timestamps must be non-decreasing across calls; the clock jumps to
        `timestamp` and decays prior mass by the elapsed interval.
        """
        if timestamp > self._clock:
            self._advance_clock(timestamp)
        self._insert(value, weight)

    def _advance_clock(mut self, new_clock: Float64):
        """Move the clock to `new_clock`, growing the decay exponent."""
        if self._decay.is_active() and new_clock > self._clock:
            self._log_scale += self._decay.rate_lambda * (
                new_clock - self._clock
            )
            if self._log_scale > _RENORM_LOG_THRESHOLD:
                self._renormalize()
        self._clock = new_clock

    def _insert(mut self, value: Float64, weight: Float64):
        """Buffer one sample using the current scale factor."""
        if self._stored_total == 0.0 and len(self._buffer) == 0:
            self._min = value
            self._max = value
        else:
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

        var scaled = weight * self._scale_factor()
        self._buffer.append(Centroid(mean=value, weight=scaled))
        self._stored_total += scaled

        if len(self._buffer) >= self._buffer_limit:
            self.merge()

    def _scale_factor(self) -> Float64:
        """e^(λ·t_now): multiplier applied to a weight inserted *now* so that it
        decays correctly relative to future queries."""
        if not self._decay.is_active():
            return 1.0
        return exp(self._log_scale)

    def _renormalize(mut self):
        """Divide all stored weights by e^(_log_scale) and reset the exponent.

        This keeps the stored weights equal to their current effective weights
        and prevents e^(λ·t) from overflowing on long streams.
        """
        var factor = exp(-self._log_scale)
        for i in range(len(self._centroids)):
            self._centroids[i].weight *= factor
        for i in range(len(self._buffer)):
            self._buffer[i].weight *= factor
        self._stored_total *= factor
        self._log_scale = 0.0

    def _effective_count(self) -> Float64:
        """Effective (decayed) total weight the normalizer should see."""
        if not self._decay.is_active():
            return self._stored_total
        return self._stored_total * exp(-self._log_scale)

    def merge(mut self):
        """Compress the buffer into the main centroid list via the merge pass.
        """
        if len(self._buffer) == 0:
            return

        # Union of merged centroids + buffered points, sorted by mean.
        var work = self._centroids.copy()
        for i in range(len(self._buffer)):
            work.append(self._buffer[i])
        self._buffer = List[Centroid]()
        _sort_centroids(work)

        if len(work) == 0:
            return

        # k-size limits use the *effective* total (so limits adapt as old mass
        # decays), but the q positions are ratios of *stored* weights, in which
        # the decay scale factor cancels. Use the stored total for the ratios.
        var effective_n = self._effective_count()
        var stored_n = self._stored_total
        var nrm = self._scale.normalizer(self.compression, effective_n)

        var merged = List[Centroid]()
        var current = work[0]
        # Stored weight to the left of (and not including) the current cluster.
        var weight_so_far = 0.0

        for i in range(1, len(work)):
            var candidate = work[i]
            # q at the left edge and the right edge if `candidate` were absorbed.
            var q_left = weight_so_far / stored_n
            var q_right = (
                weight_so_far + current.weight + candidate.weight
            ) / stored_n
            var k_span = self._scale.k(q_right, nrm) - self._scale.k(
                q_left, nrm
            )
            if k_span <= 1.0:
                current.merge(candidate)
            else:
                merged.append(current)
                weight_so_far += current.weight
                current = candidate
        merged.append(current)

        self._centroids = merged^
        if self._decay.is_active():
            self._prune()

    def _prune(mut self):
        """Drop centroids whose effective weight falls below `_PRUNE_EPSILON`.

        Keeps memory bounded on long-running decayed streams. Stored weights
        relate to effective weights by the constant factor e^(-_log_scale), so
        we compare stored weight against a scaled floor.
        """
        var stored_floor = _PRUNE_EPSILON * exp(self._log_scale)
        var kept = List[Centroid]()
        var removed = 0.0
        for i in range(len(self._centroids)):
            if self._centroids[i].weight >= stored_floor:
                kept.append(self._centroids[i])
            else:
                removed += self._centroids[i].weight
        if removed > 0.0:
            self._stored_total -= removed
            self._centroids = kept^

    def __len__(self) -> Int:
        """Number of merged centroids (after flushing the buffer, call merge).
        """
        return len(self._centroids)

    def total_weight(self) -> Float64:
        """Effective (decayed) total weight currently represented."""
        return self._effective_count()

    def min(self) -> Float64:
        return self._min

    def max(self) -> Float64:
        return self._max

    def _effective_weight(self, stored: Float64) -> Float64:
        """Convert a stored centroid weight to its current effective weight."""
        if not self._decay.is_active():
            return stored
        return stored * exp(-self._log_scale)

    def _is_singleton(self, stored: Float64) -> Bool:
        """True when a centroid holds a single (possibly decayed) sample."""
        return self._effective_weight(stored) <= 1.0 + 1e-9

    def quantile(mut self, q: Float64) raises -> Float64:
        """Estimate the value at quantile `q` in [0, 1].

        Uses linear interpolation between centroid means, with special handling
        at the extremes: singleton centroids at the tails interpolate against
        the tracked min/max so p0.x / p99.x stay sharp.
        """
        if q < 0.0 or q > 1.0:
            raise Error("quantile q must be in [0, 1]")
        self.merge()

        var cs = self._centroids.copy()
        if len(cs) == 0:
            raise Error("quantile of an empty digest is undefined")
        if len(cs) == 1:
            return cs[0].mean

        var n = self._stored_total
        if q <= 0.0:
            return self._min
        if q >= 1.0:
            return self._max

        var target = q * n

        # Below the first centroid's center of mass: interpolate from min.
        var first_half = cs[0].weight / 2.0
        if target < first_half:
            if self._is_singleton(cs[0].weight):
                return self._min
            return self._min + (cs[0].mean - self._min) * (target / first_half)

        # Above the last centroid's center of mass: interpolate to max.
        var last = cs[len(cs) - 1]
        var last_half = last.weight / 2.0
        if target > n - last_half:
            if self._is_singleton(last.weight):
                return self._max
            var frac = (target - (n - last_half)) / last_half
            return last.mean + (self._max - last.mean) * frac

        # Interior: walk cumulative weight and interpolate between the centers
        # of mass of adjacent centroids.
        var weight_so_far = cs[0].weight / 2.0
        for i in range(len(cs) - 1):
            var left = cs[i]
            var right = cs[i + 1]
            var dw = (left.weight + right.weight) / 2.0
            if target <= weight_so_far + dw:
                var frac = (target - weight_so_far) / dw
                return left.mean + (right.mean - left.mean) * frac
            weight_so_far += dw

        return last.mean

    def cdf(mut self, x: Float64) raises -> Float64:
        """Estimate P(X <= x): the inverse interpolation of `quantile`."""
        self.merge()

        var cs = self._centroids.copy()
        if len(cs) == 0:
            raise Error("cdf of an empty digest is undefined")
        if x < self._min:
            return 0.0
        if x > self._max:
            return 1.0
        if len(cs) == 1:
            if x < cs[0].mean:
                return 0.0
            if x > cs[0].mean:
                return 1.0
            return 0.5

        var n = self._stored_total

        # Left of the first center of mass.
        if x < cs[0].mean:
            if self._min == cs[0].mean:
                return 0.0
            var frac = (x - self._min) / (cs[0].mean - self._min)
            return (cs[0].weight / 2.0) * frac / n

        var last = cs[len(cs) - 1]
        if x > last.mean:
            if self._max == last.mean:
                return 1.0
            var frac = (x - last.mean) / (self._max - last.mean)
            return (n - last.weight / 2.0 + (last.weight / 2.0) * frac) / n

        # Interior.
        var weight_so_far = cs[0].weight / 2.0
        for i in range(len(cs) - 1):
            var left = cs[i]
            var right = cs[i + 1]
            if x <= right.mean:
                var dw = (left.weight + right.weight) / 2.0
                if right.mean == left.mean:
                    return (weight_so_far + dw / 2.0) / n
                var frac = (x - left.mean) / (right.mean - left.mean)
                return (weight_so_far + dw * frac) / n
            weight_so_far += (left.weight + right.weight) / 2.0

        return 1.0

    def merge_into(mut self, mut other: Self):
        """Fold `other`'s centroids into this digest, then re-run the merge pass.

        Mergeability is the defining t-digest property: the result is close to
        a digest built from the concatenation of both input streams.

        Decay rebasing (Open Design Question #2): both digests are first
        renormalized so every stored weight equals its current *effective*
        weight (`_log_scale = 0`). Folding then happens on a common
        "effective now" basis, independent of each digest's time origin. The
        clock advances to the later of the two.
        """
        other.merge()
        if other._effective_count() == 0.0:
            return

        self.merge()
        if self._decay.is_active():
            self._renormalize()
        if other._decay.is_active():
            other._renormalize()

        var was_empty = self._stored_total == 0.0
        for i in range(len(other._centroids)):
            self._buffer.append(other._centroids[i])
        self._stored_total += other._stored_total

        if was_empty:
            self._min = other._min
            self._max = other._max
        else:
            self._min = min(self._min, other._min)
            self._max = max(self._max, other._max)
        self._clock = max(self._clock, other._clock)
        self.merge()
