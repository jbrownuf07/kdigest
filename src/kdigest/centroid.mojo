"""Centroid: a weighted cluster summarizing nearby samples in a t-digest.

Each centroid tracks the mean of a cluster of samples and the total effective
weight (count) of that cluster. Weights may be fractional once exponential
decay is applied (Phase 3).

Generic over a `DType` (Phase 5a) so the same type serves the Float64 CPU
reference and the Float32 GPU backend; the default is `float64`, so bare
`Centroid` is `Centroid[float64]` and existing call sites are unchanged.
"""


@fieldwise_init
struct Centroid[dtype: DType = DType.float64](
    ImplicitlyCopyable, Movable, Writable
):
    var mean: Scalar[Self.dtype]
    var weight: Scalar[Self.dtype]

    @staticmethod
    def singleton(value: Scalar[Self.dtype]) -> Centroid[Self.dtype]:
        """A centroid holding a single unit-weight sample."""
        return Centroid[Self.dtype](mean=value, weight=1.0)

    def add_sample(
        mut self, value: Scalar[Self.dtype], sample_weight: Scalar[Self.dtype]
    ):
        """Fold one weighted sample into this centroid.

        Uses the numerically stable incremental weighted-mean update
            mean += (value - mean) * (w / (weight + w))
        which avoids forming the large intermediate sum (mean * weight).
        """
        var new_weight = self.weight + sample_weight
        self.mean += (value - self.mean) * (sample_weight / new_weight)
        self.weight = new_weight

    def merge(mut self, other: Centroid[Self.dtype]):
        """Fold another centroid's mass into this one."""
        self.add_sample(other.mean, other.weight)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Centroid(mean=", self.mean, ", weight=", self.weight, ")")
