"""GroupedTDigest: one t-digest per key, compressed in a single GPU launch.

The headline Phase 5 feature. Ingest a `(keys, values)` batch where each key is
a dense id in `[0, num_keys)`; each key's samples are compressed into their own
t-digest with **one GPU block per key** (Stage 5e). Per-key `quantile` / `cdf` /
`to_bytes` reuse the full `TDigest` query surface.

Falls back to a per-key CPU `add` loop when no accelerator is present, so the
same code runs (correctly, if slower) everywhere.

Example:
    var g = GroupedTDigest[ScaleK2](num_keys=1000, compression=100.0)
    g.add_batch(keys, values)           # keys: List[Int32], values: List[Float64]
    g.add_batch(keys2, values2)         # accumulates into the same digests
    var p99 = g.quantile(key=7, q=0.99)

    # Recency-weighted per key (each key's clock advances by its sample count):
    var r = GroupedTDigest[ScaleK2](1000, 100.0, Decay.half_life(10_000))
"""

from std.sys import has_accelerator

from kdigest.gpu.session import GpuSession
from kdigest.decay import Decay
from kdigest.scale import ScaleFunction, ScaleK1
from kdigest.tdigest import TDigest
from kdigest.gpu.grouped import gpu_grouped_compress_weighted


struct GroupedTDigest[S: ScaleFunction](Movable) where (
    S.DT == DType.float64 and S.F32.DT == DType.float32
):
    """A collection of `num_keys` independent t-digests sharing a scale `S` and
    compression, updated together from batched `(key, value)` input."""

    var num_keys: Int
    var compression: Float64
    var decay: Decay
    var _digests: List[TDigest[Self.S]]
    # One long-lived GPU session (device context + reusable workspaces),
    # created lazily on the first GPU batch and reused for every later batch.
    var _session: Optional[GpuSession]

    def __init__(
        out self,
        num_keys: Int,
        compression: Float64 = 100.0,
        decay: Decay = Decay.none(),
    ):
        self.num_keys = num_keys
        self.compression = compression
        self.decay = decay
        self._digests = List[TDigest[Self.S]]()
        for _ in range(num_keys):
            self._digests.append(
                TDigest[Self.S](compression=compression, decay=decay)
            )
        self._session = None

    def add_batch(mut self, keys: List[Int32], values: List[Float64]) raises:
        """Fold a batch of `(key, value)` samples into their per-key digests.

        `keys[i]` must be a dense id in `[0, num_keys)`. Batches accumulate:
        each key's digest keeps everything it has seen (subject to its decay),
        and keys absent from this batch are unchanged. Per key this matches
        `TDigest.add_batch` on that key's samples (batch order preserved): with
        decay, each key's clock advances by its own sample count.

        On a GPU every touched key is compressed in a single launch (one block
        per key, its existing centroids folded in with its new samples);
        otherwise each sample goes through the per-key CPU `add` path. Either
        way the whole batch is validated before any digest changes.
        """
        if len(keys) != len(values):
            raise Error("keys and values must have equal length")
        var n = len(values)
        if n == 0:
            return

        comptime gpu_available = has_accelerator()
        if not gpu_available:
            for i in range(n):
                var k = Int(keys[i])
                if k < 0 or k >= self.num_keys:
                    raise Error("key out of range [0, num_keys)")
                TDigest[Self.S]._validate(values[i], 1.0)
            for i in range(n):
                self._digests[Int(keys[i])].add(values[i])
            return

        for i in range(n):
            var k = Int(keys[i])
            if k < 0 or k >= self.num_keys:
                raise Error("key out of range [0, num_keys)")
            # Validate everything before mutating any digest, so a bad value
            # cannot leave the batch half-applied.
            TDigest[Self.S]._validate(values[i], 1.0)

        # Group the batch per key (batch order preserved within a key).
        var per_key = List[List[Float64]](
            length=self.num_keys, fill=List[Float64]()
        )
        for i in range(n):
            per_key[Int(keys[i])].append(values[i])

        # Each touched key's digest prepares its own GPU input exactly as
        # `TDigest.add_batch` would: advance its clock by its sample count,
        # scale new weights by its decay factor, and prepend its existing
        # centroids + buffer. Untouched keys pass an empty input and are left
        # as they are (not launched, not reset). A key with no prior state and
        # unit weight passes its samples as plain new samples instead, so a
        # fresh (or undecayed first) batch takes the cheaper unweighted kernel.
        var prior_m = List[List[Float32]](
            length=self.num_keys, fill=List[Float32]()
        )
        var prior_w = List[List[Float32]](
            length=self.num_keys, fill=List[Float32]()
        )
        var new_w = List[Float32](length=self.num_keys, fill=Float32(1))
        var fresh_keys = List[Int32]()
        var fresh_vals = List[Float64]()
        for k in range(self.num_keys):
            if len(per_key[k]) == 0:
                continue
            var prior_n = len(self._digests[k]._centroids) + len(
                self._digests[k]._buffer
            )
            var prepared = self._digests[k]._gpu_batch_prepare(per_key[k])
            var m = len(per_key[k])
            var all_unit = True
            var pw = prepared[1].copy()
            for j in range(prior_n, len(pw)):
                if pw[j] != Float32(1):
                    all_unit = False
                    break
            if prior_n == 0 and all_unit:
                for j in range(m):
                    fresh_keys.append(Int32(k))
                    fresh_vals.append(per_key[k][j])
            else:
                prior_m[k] = prepared[0].copy()
                prior_w[k] = pw^

        if not self._session:
            self._session = GpuSession()
        ref session = self._session.value()
        var res = gpu_grouped_compress_weighted[Self.S.F32](
            session.ctx,
            session.ws,
            fresh_keys,
            fresh_vals,
            new_w,
            prior_m,
            prior_w,
            self.num_keys,
            Float32(self.compression),
        )
        # Install each touched key's compressed centroids into its digest.
        for k in range(self.num_keys):
            if len(per_key[k]) == 0:
                continue
            self._digests[k]._gpu_batch_finish(res.means[k], res.weights[k])

    def quantile(mut self, key: Int, q: Float64) raises -> Float64:
        """Estimate the value at quantile `q` for `key`."""
        if key < 0 or key >= self.num_keys:
            raise Error("key out of range [0, num_keys)")
        return self._digests[key].quantile(q)

    def cdf(mut self, key: Int, x: Float64) raises -> Float64:
        """Estimate P(X <= x) for `key`."""
        if key < 0 or key >= self.num_keys:
            raise Error("key out of range [0, num_keys)")
        return self._digests[key].cdf(x)

    def to_bytes(mut self, key: Int) raises -> List[UInt8]:
        """Serialize one key's digest for shipping."""
        if key < 0 or key >= self.num_keys:
            raise Error("key out of range [0, num_keys)")
        return self._digests[key].to_bytes()

    def centroid_count(self, key: Int) raises -> Int:
        if key < 0 or key >= self.num_keys:
            raise Error("key out of range [0, num_keys)")
        return len(self._digests[key])

    def total_weight(self, key: Int) raises -> Float64:
        if key < 0 or key >= self.num_keys:
            raise Error("key out of range [0, num_keys)")
        return self._digests[key].total_weight()
