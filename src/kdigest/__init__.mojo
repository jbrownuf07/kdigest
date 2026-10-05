"""Accurate streaming quantiles at the extremes, with time decay (kdigest).

A Mojo implementation of the t-digest data structure, specialized for accurate
extreme-tail quantile estimation over unbounded data streams. Adds exponential
decay of centroid weights (recency weighting) and pluggable scale functions.

See README.md for an overview and PLAN.md for the implementation roadmap.
"""

from kdigest.centroid import Centroid
from kdigest.decay import Decay
from kdigest.scale import (
    ScaleFunction,
    ScaleK0,
    ScaleK1,
    ScaleK2,
    ScaleK3,
)
from kdigest.tdigest import TDigest
