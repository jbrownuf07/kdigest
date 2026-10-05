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

from kdigest.centroid import Centroid
from kdigest.scale import ScaleFunction, ScaleK1


# Buffer size as a multiple of compression. The reference uses a buffer several
# times the centroid budget so each merge pass amortizes well.
comptime _BUFFER_FACTOR = 5


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
    """A merging t-digest with a pluggable scale function `S`."""

    var compression: Float64
    var _scale: Self.S
    # Merged, mean-sorted centroids.
    var _centroids: List[Centroid]
    # Unmerged incoming samples awaiting the next merge pass.
    var _buffer: List[Centroid]
    var _buffer_limit: Int
    # Running totals (merged + buffered) kept so `add` stays O(1).
    var _total_weight: Float64
    var _min: Float64
    var _max: Float64

    def __init__(
        out self, compression: Float64 = 100.0, scale: Self.S = Self.S()
    ):
        self.compression = compression
        self._scale = scale
        self._centroids = List[Centroid]()
        self._buffer = List[Centroid]()
        self._buffer_limit = Int(compression) * _BUFFER_FACTOR
        self._total_weight = 0.0
        self._min = 0.0
        self._max = 0.0

    def add(mut self, value: Float64, weight: Float64 = 1.0):
        """Add one (optionally weighted) sample to the digest."""
        if self._total_weight == 0.0:
            self._min = value
            self._max = value
        else:
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

        self._buffer.append(Centroid(mean=value, weight=weight))
        self._total_weight += weight

        if len(self._buffer) >= self._buffer_limit:
            self.merge()

    def _effective_count(self) -> Float64:
        """Total weight the normalizer should see (overridden by decay later).
        """
        return self._total_weight

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

        var n = self._effective_count()
        var nrm = self._scale.normalizer(self.compression, n)

        var merged = List[Centroid]()
        var current = work[0]
        # Weight to the left of (and not including) the current cluster.
        var weight_so_far = 0.0

        for i in range(1, len(work)):
            var candidate = work[i]
            # q at the left edge and the right edge if `candidate` were absorbed.
            var q_left = weight_so_far / n
            var q_right = (
                weight_so_far + current.weight + candidate.weight
            ) / n
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

    def __len__(self) -> Int:
        """Number of merged centroids (after flushing the buffer, call merge).
        """
        return len(self._centroids)

    def total_weight(self) -> Float64:
        return self._total_weight

    def min(self) -> Float64:
        return self._min

    def max(self) -> Float64:
        return self._max

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

        var n = self._total_weight
        if q <= 0.0:
            return self._min
        if q >= 1.0:
            return self._max

        var target = q * n

        # Below the first centroid's center of mass: interpolate from min.
        var first_half = cs[0].weight / 2.0
        if target < first_half:
            if cs[0].weight == 1.0:
                return self._min
            return self._min + (cs[0].mean - self._min) * (target / first_half)

        # Above the last centroid's center of mass: interpolate to max.
        var last = cs[len(cs) - 1]
        var last_half = last.weight / 2.0
        if target > n - last_half:
            if last.weight == 1.0:
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

        var n = self._total_weight

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
        """
        other.merge()
        if other._total_weight == 0.0:
            return

        var was_empty = self._total_weight == 0.0
        for i in range(len(other._centroids)):
            self._buffer.append(other._centroids[i])
        self._total_weight += other._total_weight

        if was_empty:
            self._min = other._min
            self._max = other._max
        else:
            self._min = min(self._min, other._min)
            self._max = max(self._max, other._max)
        self.merge()
