"""GPU building blocks for kdigest (Phase 5, Stage 5b).

This subpackage is only meaningful when a GPU accelerator is present. The
device code imports `max.gpu` / `layout`, which are available through the `max`
package. Host callers must gate GPU use behind `std.sys.has_accelerator()`; the
tests no-op (pass) on GPU-less machines so CI stays green everywhere.
"""

from kdigest.gpu.primitives import (
    BLOCK_CAP,
    gpu_sort_pairs,
    gpu_inclusive_scan,
    gpu_exclusive_scan,
    gpu_segmented_scan,
    gpu_compact,
    cpu_sort_pairs,
    cpu_inclusive_scan,
    cpu_exclusive_scan,
    cpu_segmented_scan,
    cpu_compact,
)
from kdigest.gpu.scale_device import gpu_scale_eval
from kdigest.gpu.add_batch import gpu_compress_batch
from kdigest.gpu.grouped import gpu_grouped_compress, GroupedResult
from kdigest.gpu.query import gpu_quantile, gpu_cdf, gpu_merge_digests
