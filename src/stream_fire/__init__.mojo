"""Stream-fire: accurate streaming quantiles at the extremes, with time decay.

A Mojo implementation of the t-digest data structure, specialized for accurate
extreme-tail quantile estimation over unbounded data streams. Adds exponential
decay of centroid weights (recency weighting) and pluggable scale functions.

See README.md for an overview and PLAN.md for the implementation roadmap.
"""

from stream_fire.centroid import Centroid
from stream_fire.scale import (
    ScaleFunction,
    ScaleK0,
    ScaleK1,
    ScaleK2,
    ScaleK3,
)
