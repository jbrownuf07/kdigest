"""stream-fire: accurate streaming quantiles at the extremes, with time decay.

A Mojo implementation of the t-digest data structure, specialized for accurate
extreme-tail quantile estimation over unbounded data streams. Adds exponential
decay of centroid weights (recency weighting) and pluggable scale functions.

See README.md for an overview and PLAN.md for the implementation roadmap.
"""
