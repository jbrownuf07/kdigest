"""Centroid: a weighted cluster summarizing nearby samples in a t-digest.

Each centroid tracks the mean of a cluster of samples and the total effective
weight (count) of that cluster. Weights may be fractional once exponential
decay is applied (Phase 3).
"""


@fieldwise_init
struct Centroid(ImplicitlyCopyable, Movable, Writable):
    var mean: Float64
    var weight: Float64

    @staticmethod
    def singleton(value: Float64) -> Centroid:
        """A centroid holding a single unit-weight sample."""
        return Centroid(mean=value, weight=1.0)

    def add_sample(mut self, value: Float64, sample_weight: Float64):
        """Fold one weighted sample into this centroid.

        Uses the numerically stable incremental weighted-mean update
            mean += (value - mean) * (w / (weight + w))
        which avoids forming the large intermediate sum (mean * weight).
        """
        var new_weight = self.weight + sample_weight
        self.mean += (value - self.mean) * (sample_weight / new_weight)
        self.weight = new_weight

    def merge(mut self, other: Centroid):
        """Fold another centroid's mass into this one."""
        self.add_sample(other.mean, other.weight)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Centroid(mean=", self.mean, ", weight=", self.weight, ")")
