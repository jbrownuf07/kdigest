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
    var p99 = g.quantile(key=7, q=0.99)
"""

from std.sys import has_accelerator

from kdigest.scale import ScaleFunction, ScaleK1
from kdigest.tdigest import TDigest
from kdigest.gpu.grouped import gpu_grouped_compress


struct GroupedTDigest[S: ScaleFunction](Movable) where (
    S.DT == DType.float64 and S.F32.DT == DType.float32
):
    """A collection of `num_keys` independent t-digests sharing a scale `S` and
    compression, updated together from batched `(key, value)` input."""

    var num_keys: Int
    var compression: Float64
    var _digests: List[TDigest[Self.S]]

    def __init__(out self, num_keys: Int, compression: Float64 = 100.0):
        self.num_keys = num_keys
        self.compression = compression
        self._digests = List[TDigest[Self.S]]()
        for _ in range(num_keys):
            self._digests.append(TDigest[Self.S](compression=compression))

    def add_batch(mut self, keys: List[Int32], values: List[Float64]) raises:
        """Fold a batch of `(key, value)` samples into their per-key digests.

        `keys[i]` must be a dense id in `[0, num_keys)`. On a GPU this compresses
        every key in a single launch (one block per key); otherwise it routes
        each sample through the per-key CPU `add` path.
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
                self._digests[k].add(values[i])
            return

        for i in range(n):
            var k = Int(keys[i])
            if k < 0 or k >= self.num_keys:
                raise Error("key out of range [0, num_keys)")

        var res = gpu_grouped_compress[Self.S.F32](
            keys, values, self.num_keys, Float32(self.compression)
        )
        # Replace each key's digest with one wrapping the compressed centroids.
        var rebuilt = List[TDigest[Self.S]]()
        for k in range(self.num_keys):
            var cnt = len(res.means[k])
            if cnt == 0:
                rebuilt.append(TDigest[Self.S](compression=self.compression))
                continue
            var dm = List[Float64]()
            var dw = List[Float64]()
            for j in range(cnt):
                dm.append(Float64(res.means[k][j]))
                dw.append(Float64(res.weights[k][j]))
            rebuilt.append(
                TDigest[Self.S]._from_centroids(
                    self.compression,
                    dm,
                    dw,
                    Float64(res.mins[k]),
                    Float64(res.maxs[k]),
                )
            )
        self._digests = rebuilt^

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
