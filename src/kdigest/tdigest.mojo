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

from std.math import exp, isnan, isinf
from std.memory import bitcast
from std.sys import has_accelerator

from kdigest.centroid import Centroid

# Stage 5a: TDigest remains Float64-only; the GPU Float32 port lands in Stage
# 5d. Bind the (now DType-generic) Centroid to float64 for use here.
comptime _Centroid = Centroid[DType.float64]
from kdigest.decay import Decay
from kdigest.scale import ScaleFunction, ScaleK1
from kdigest.gpu.add_batch import gpu_compress_batch
from kdigest.gpu.session import GpuSession


# Buffer size as a multiple of compression. The reference uses a buffer several
# times the centroid budget so each merge pass amortizes well.
comptime _BUFFER_FACTOR = 5

# Renormalize all stored weights once the running decay exponent (λ·t) grows
# past this, to keep e^(λ·t) far from Float64 overflow (~709).
comptime _RENORM_LOG_THRESHOLD = 400.0

# Effective-weight floor: centroids lighter than this are pruned after a merge
# when decay is active, bounding memory on indefinite streams.
comptime _PRUNE_EPSILON = 1e-6

# add_batch dispatches to the GPU only above this batch size; below it the fixed
# per-call cost (device context + workspace allocation, a few GPU syncs) wins
# and the CPU buffer path is faster. Measured in bench/bench_gpu.mojo on an
# Apple M1 Max (Metal, unified memory): one-shot GPU is 0.98x the CPU at 50k
# points and 1.54x at 100k. These thresholds are hardware-specific; re-derive
# them with `pixi run bench-gpu` on other machines (see bench/Benchmark.md).
comptime _GPU_CROSSOVER = 60_000
# With a caller-held `GpuSession` the context and workspace are reused, so the
# GPU wins sooner: 0.78x at 30k, 1.27x at 50k.
comptime _GPU_CROSSOVER_SESSION = 40_000

# Serialization format version, written as the first byte of `to_bytes`.
comptime _SERDE_VERSION: UInt8 = 1


def _sort_centroids(mut cs: List[_Centroid]):
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


def _put_f64(mut out: List[UInt8], value: Float64):
    """Append `value` as 8 little-endian bytes of its IEEE-754 bit pattern."""
    var bits = UInt64(value.to_bits())
    for i in range(8):
        out.append(UInt8((bits >> UInt64(8 * i)) & UInt64(0xFF)))


def _get_f64(data: List[UInt8], offset: Int) raises -> Float64:
    """Read a little-endian Float64 at `offset` (8 bytes)."""
    if offset + 8 > len(data):
        raise Error("truncated digest: not enough bytes for a Float64")
    var bits = UInt64(0)
    for i in range(8):
        bits |= UInt64(data[offset + i]) << UInt64(8 * i)
    return bitcast[DType.float64, 1](bits)


struct TDigest[S: ScaleFunction](Copyable, Movable, Sized) where (
    S.DT == DType.float64
):
    """A merging t-digest with a pluggable scale function `S`.

    Stage 5a note: this CPU digest is Float64-only (`where S.DT == float64`).
    The scale function is DType-generic for the Float32 GPU backend (Stage 5d),
    but here `S` is constrained to Float64 so the hot-path math stays in one
    precision.

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
    var _centroids: List[_Centroid]
    # Unmerged incoming samples awaiting the next merge pass (also scaled).
    var _buffer: List[_Centroid]
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
        self._centroids = List[_Centroid]()
        self._buffer = List[_Centroid]()
        self._buffer_limit = Int(compression) * _BUFFER_FACTOR
        self._stored_total = 0.0
        self._log_scale = 0.0
        self._clock = 0.0
        self._min = 0.0
        self._max = 0.0

    def add(mut self, value: Float64, weight: Float64 = 1.0) raises:
        """Add one (optionally weighted) sample, advancing the clock by one tick.

        Use this for count-based decay (one unit of time per sample). Rejects
        NaN/inf values and non-positive weights.
        """
        self._validate(value, weight)
        self._advance_clock(self._clock + 1.0)
        self._insert(value, weight)

    def add_at(
        mut self, value: Float64, timestamp: Float64, weight: Float64 = 1.0
    ) raises:
        """Add a sample observed at `timestamp` for time-based decay.

        Timestamps must be non-decreasing across calls; the clock jumps to
        `timestamp` and decays prior mass by the elapsed interval. Rejects
        NaN/inf values and non-positive weights.
        """
        self._validate(value, weight)
        if timestamp > self._clock:
            self._advance_clock(timestamp)
        self._insert(value, weight)

    def add_batch(
        mut self, values: List[Float64]
    ) raises where Self.S.F32.DT == DType.float32:
        """Fold a whole batch of unit-weight samples in at once.

        Semantically equivalent to calling `add(v)` for each `v` in `values`:
        the batch advances the clock by `len(values)` ticks (so count-based
        decay matches the one-at-a-time path), and the resulting quantiles agree
        with the scalar path within tolerance.

        When an accelerator is present and the batch is large enough to amortize
        the host<->device transfer (`len(values) >= _GPU_CROSSOVER`), the merge
        runs on the GPU with the parallel k-bucketing formulation. Otherwise it
        falls back to the proven CPU `add` loop. The GPU path is Float32, so its
        quantiles match the Float64 CPU path to fp32 tolerance rather than bit
        for bit.
        """
        var n = len(values)
        if n == 0:
            return

        comptime gpu_available = has_accelerator()
        if not gpu_available or n < _GPU_CROSSOVER:
            for i in range(n):
                self.add(values[i])
            return
        var session = GpuSession()
        self._add_batch_gpu(values, session)

    def add_batch(
        mut self, values: List[Float64], mut session: GpuSession
    ) raises where Self.S.F32.DT == DType.float32:
        """`add_batch` reusing a caller-held `GpuSession` (device context +
        workspaces), so repeated batches pay no per-call setup. Because that
        fixed cost is gone, the GPU is used from a lower batch size
        (`_GPU_CROSSOVER_SESSION`)."""
        var n = len(values)
        if n == 0:
            return
        comptime gpu_available = has_accelerator()
        if not gpu_available or n < _GPU_CROSSOVER_SESSION:
            for i in range(n):
                self.add(values[i])
            return
        self._add_batch_gpu(values, session)

    def _add_batch_gpu(
        mut self, values: List[Float64], mut session: GpuSession
    ) raises where Self.S.F32.DT == DType.float32:
        var n = len(values)
        # Validate up front (the GPU kernel assumes finite inputs).
        for i in range(n):
            self._validate(values[i], 1.0)

        # The batch occupies one clock span; advance once and scale the new
        # weights by the current decay factor, exactly as `_insert` would.
        self._advance_clock(self._clock + Float64(n))
        var scale_factor = self._scale_factor()

        # Assemble (mean, weight) in Float32: existing centroids (already
        # scaled) + the new samples (scaled on the way in).
        var means = List[Float32]()
        var weights = List[Float32]()
        for i in range(len(self._centroids)):
            means.append(Float32(self._centroids[i].mean))
            weights.append(Float32(self._centroids[i].weight))
        for i in range(len(self._buffer)):
            means.append(Float32(self._buffer[i].mean))
            weights.append(Float32(self._buffer[i].weight))
        self._buffer = List[_Centroid]()
        var vmin = values[0]
        var vmax = values[0]
        for i in range(n):
            var v = values[i]
            means.append(Float32(v))
            weights.append(Float32(scale_factor))
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        self._stored_total += Float64(n) * scale_factor

        # Track running min/max across the whole stream.
        if (
            self._stored_total == Float64(n) * scale_factor
            and len(self._centroids) == 0
        ):
            self._min = vmin
            self._max = vmax
        else:
            if vmin < self._min:
                self._min = vmin
            if vmax > self._max:
                self._max = vmax

        var out = gpu_compress_batch[Self.S.F32](
            session.ctx,
            session.ws.batch,
            means,
            weights,
            Float32(self.compression),
        )
        var new_centroids = List[_Centroid]()
        var total = 0.0
        for i in range(len(out[0])):
            var wt = Float64(out[1][i])
            new_centroids.append(_Centroid(mean=Float64(out[0][i]), weight=wt))
            total += wt
        self._centroids = new_centroids^
        # Reconcile the stored total with the compressed weights (fp32 rounding
        # can nudge it slightly).
        self._stored_total = total
        if self._decay.is_active():
            self._prune()

    @staticmethod
    def _validate(value: Float64, weight: Float64) raises:
        if isnan(value) or isinf(value):
            raise Error("value must be finite (got NaN or inf)")
        if weight <= 0.0 or isnan(weight) or isinf(weight):
            raise Error("weight must be positive and finite")

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
        self._buffer.append(_Centroid(mean=value, weight=scaled))
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

    # Float64 wrappers over the DType-generic scale function. `where S.DT ==
    # float64` guarantees these rebinds are identities (not reinterpretations).
    def _normalizer(self, compression: Float64, n: Float64) -> Float64:
        var c = rebind[Scalar[Self.S.DT]](compression)
        var nn = rebind[Scalar[Self.S.DT]](n)
        return rebind[Float64](self._scale.normalizer(c, nn))

    def _k(self, q: Float64, nrm: Float64) -> Float64:
        var qq = rebind[Scalar[Self.S.DT]](q)
        var nn = rebind[Scalar[Self.S.DT]](nrm)
        return rebind[Float64](self._scale.k(qq, nn))

    def merge(mut self):
        """Compress the buffer into the main centroid list via the merge pass.
        """
        if len(self._buffer) == 0:
            return

        # Union of merged centroids + buffered points, sorted by mean.
        var work = self._centroids.copy()
        for i in range(len(self._buffer)):
            work.append(self._buffer[i])
        self._buffer = List[_Centroid]()
        _sort_centroids(work)

        if len(work) == 0:
            return

        # k-size limits use the *effective* total (so limits adapt as old mass
        # decays), but the q positions are ratios of *stored* weights, in which
        # the decay scale factor cancels. Use the stored total for the ratios.
        var effective_n = self._effective_count()
        var stored_n = self._stored_total
        var nrm = self._normalizer(self.compression, effective_n)

        var merged = List[_Centroid]()
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
            var k_span = self._k(q_right, nrm) - self._k(q_left, nrm)
            # Always coalesce exact-tie means (e.g. all-identical streams), and
            # otherwise grow the cluster while it fits in the k-size budget.
            if candidate.mean == current.mean or k_span <= 1.0:
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
        var kept = List[_Centroid]()
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

    # --- Serialization -------------------------------------------------------

    def to_bytes(mut self) -> List[UInt8]:
        """Serialize the digest to a portable little-endian byte buffer.

        The buffer can be persisted or shipped to another process and restored
        with `from_bytes`. The scale function has no runtime state, so it is not
        stored; the reader must supply the same scale type parameter.

        Layout: [version:u8] then little-endian Float64s:
          compression, decay λ, _log_scale, _clock, _stored_total, min, max,
          centroid_count, then (mean, weight) per centroid.
        """
        self.merge()
        var out = List[UInt8]()
        out.append(_SERDE_VERSION)
        _put_f64(out, self.compression)
        _put_f64(out, self._decay.rate_lambda)
        _put_f64(out, self._log_scale)
        _put_f64(out, self._clock)
        _put_f64(out, self._stored_total)
        _put_f64(out, self._min)
        _put_f64(out, self._max)
        _put_f64(out, Float64(len(self._centroids)))
        for i in range(len(self._centroids)):
            _put_f64(out, self._centroids[i].mean)
            _put_f64(out, self._centroids[i].weight)
        return out^

    @staticmethod
    def from_bytes(data: List[UInt8], scale: Self.S = Self.S()) raises -> Self:
        """Reconstruct a digest from `to_bytes` output.

        `scale` must be the same scale function the digest was built with (its
        type is a compile-time parameter and is not stored in the buffer).
        """
        if len(data) == 0 or data[0] != _SERDE_VERSION:
            raise Error("unrecognized digest serialization version")

        var off = 1
        var compression = _get_f64(data, off)
        off += 8
        var rate_lambda = _get_f64(data, off)
        off += 8
        var log_scale = _get_f64(data, off)
        off += 8
        var clock = _get_f64(data, off)
        off += 8
        var stored_total = _get_f64(data, off)
        off += 8
        var min_v = _get_f64(data, off)
        off += 8
        var max_v = _get_f64(data, off)
        off += 8
        var count = Int(_get_f64(data, off))
        off += 8

        var result = Self(
            compression=compression,
            scale=scale,
            decay=Decay.rate(rate_lambda),
        )
        for _ in range(count):
            var mean = _get_f64(data, off)
            off += 8
            var weight = _get_f64(data, off)
            off += 8
            result._centroids.append(_Centroid(mean=mean, weight=weight))
        result._log_scale = log_scale
        result._clock = clock
        result._stored_total = stored_total
        result._min = min_v
        result._max = max_v
        return result^

    @staticmethod
    def _from_centroids(
        compression: Float64,
        means: List[Float64],
        weights: List[Float64],
        min_v: Float64,
        max_v: Float64,
        scale: Self.S = Self.S(),
    ) -> Self:
        """Build a (decay-free) digest directly from merged centroid arrays.

        Used by `GroupedTDigest` to wrap per-key centroids produced on the GPU
        so they inherit the full query surface (quantile/cdf/to_bytes). The
        centroids must already be mean-sorted and compressed.
        """
        var result = Self(compression=compression, scale=scale)
        var total = 0.0
        for i in range(len(means)):
            result._centroids.append(
                _Centroid(mean=means[i], weight=weights[i])
            )
            total += weights[i]
        result._stored_total = total
        result._clock = total
        result._min = min_v
        result._max = max_v
        return result^
