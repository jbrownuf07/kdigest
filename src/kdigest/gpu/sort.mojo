"""Device-side stable sort for the GPU compress pipeline (step 3, part B).

Sorts `n` Float32 keys (centroid means) on the GPU and returns the permutation,
so the batch never has to be sorted on the host:

  1. `_tile_sort_kernel`  — one block per BLOCK_CAP tile: bitonic sort in shared
     memory of (key, original index) pairs. Comparing lexicographically on
     (key, index) makes every element distinct, so the order is exactly that of
     a *stable* sort by key.
  2. `_merge_pass_kernel` — merges adjacent sorted runs of width `w` into runs
     of width `2w`, one thread per element: an element's output position is its
     rank within its own run plus the number of elements in the partner run that
     precede it (binary search). Repeated with doubling `w` until one run
     remains; passes are enqueued back to back with no host synchronization.
  3. `_permute_kernel`    — gathers the weights through the final permutation.

Result: device buffers holding the mean-sorted (mean, weight) pairs, ready for
the device-resident compress passes in `add_batch.mojo`.
"""

from std.math import ceildiv

from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from kdigest.gpu.primitives import BLOCK_CAP


comptime _DT = DType.float32
comptime _IDX = DType.int32
comptime _cap = row_major[BLOCK_CAP]()
comptime _MERGE_BLOCK = 256


@always_inline
def _before(ka: Float32, ia: Int32, kb: Float32, ib: Int32) -> Bool:
    """Strict (key, index) lexicographic order."""
    return ka < kb or (ka == kb and ia < ib)


def _tile_sort_kernel[
    L: TensorLayout
](
    keys: TileTensor[_DT, L, MutAnyOrigin],
    idx: TileTensor[_IDX, L, MutAnyOrigin],
    n: Int32,
):
    """Block `b` sorts tile `[b*BLOCK_CAP, ...)` of `keys` in place and writes
    the matching original indices to `idx`. Padding uses (+MAX, INT32_MAX) so
    it sorts after every real element."""
    comptime assert keys.flat_rank == 1 and idx.flat_rank == 1
    var sk = stack_allocation[_DT, address_space=AddressSpace.SHARED](_cap)
    var si = stack_allocation[_IDX, address_space=AddressSpace.SHARED](_cap)
    var tid = Int(thread_idx.x)
    var base = Int(block_idx.x) * BLOCK_CAP
    var cnt = min(BLOCK_CAP, Int(n) - base)

    if tid < cnt:
        sk[tid] = keys[base + tid]
        si[tid] = Int32(base + tid)
    else:
        sk[tid] = Scalar[_DT].MAX_FINITE
        si[tid] = Int32(2147483647)
    barrier()

    var k = 2
    while k <= BLOCK_CAP:
        var j = k // 2
        while j > 0:
            var ixj = tid ^ j
            if ixj > tid:
                var ka = rebind[Scalar[_DT]](sk[tid])
                var kb = rebind[Scalar[_DT]](sk[ixj])
                var ia = rebind[Scalar[_IDX]](si[tid])
                var ib = rebind[Scalar[_IDX]](si[ixj])
                var ascending = (tid & k) == 0
                var b_first = _before(kb, ib, ka, ia)
                var a_first = _before(ka, ia, kb, ib)
                if (ascending and b_first) or (not ascending and a_first):
                    sk[tid] = kb
                    sk[ixj] = ka
                    si[tid] = ib
                    si[ixj] = ia
            barrier()
            j //= 2
        k *= 2

    if tid < cnt:
        keys[base + tid] = sk[tid]
        idx[base + tid] = si[tid]


def _merge_pass_kernel[
    L: TensorLayout
](
    src_k: TileTensor[_DT, L, MutAnyOrigin],
    src_i: TileTensor[_IDX, L, MutAnyOrigin],
    dst_k: TileTensor[_DT, L, MutAnyOrigin],
    dst_i: TileTensor[_IDX, L, MutAnyOrigin],
    n: Int32,
    width: Int32,
):
    """Merge sorted runs of `width` pairwise into runs of `2*width`."""
    comptime assert src_k.flat_rank == 1 and src_i.flat_rank == 1
    comptime assert dst_k.flat_rank == 1 and dst_i.flat_rank == 1
    var i = Int(global_idx.x)
    var nn = Int(n)
    if i >= nn:
        return
    var w = Int(width)
    var run = i // w
    var own_lo = run * w
    var partner = run ^ 1
    var other_lo = partner * w
    var key = rebind[Scalar[_DT]](src_k[i])
    var ix = rebind[Scalar[_IDX]](src_i[i])
    if other_lo >= nn:
        # Trailing run with no partner this pass: already in place.
        dst_k[i] = key
        dst_i[i] = ix
        return
    var other_hi = min(other_lo + w, nn)
    # Count elements in the partner run that come before (key, ix).
    var lo = other_lo
    var hi = other_hi
    while lo < hi:
        var mid = (lo + hi) // 2
        var mk = rebind[Scalar[_DT]](src_k[mid])
        var mi = rebind[Scalar[_IDX]](src_i[mid])
        if _before(mk, mi, key, ix):
            lo = mid + 1
        else:
            hi = mid
    var pair_lo = min(own_lo, other_lo)
    var pos = pair_lo + (i - own_lo) + (lo - other_lo)
    dst_k[pos] = key
    dst_i[pos] = ix


def _permute_kernel[
    L: TensorLayout
](
    src: TileTensor[_DT, L, MutAnyOrigin],
    perm: TileTensor[_IDX, L, MutAnyOrigin],
    dst: TileTensor[_DT, L, MutAnyOrigin],
    n: Int32,
):
    """`dst[i] = src[perm[i]]`."""
    comptime assert src.flat_rank == 1 and perm.flat_rank == 1
    comptime assert dst.flat_rank == 1
    var i = Int(global_idx.x)
    if i < Int(n):
        dst[i] = src[Int(rebind[Scalar[_IDX]](perm[i]))]


def gpu_sort_by_mean(
    ctx: DeviceContext,
    keys_a: DeviceBuffer[_DT],
    keys_b: DeviceBuffer[_DT],
    idx_a: DeviceBuffer[_IDX],
    idx_b: DeviceBuffer[_IDX],
    weights_in: DeviceBuffer[_DT],
    weights_out: DeviceBuffer[_DT],
    n: Int,
) raises -> Bool:
    """Stable-sort the `n` keys in `keys_a` on the device and gather
    `weights_in` through the permutation into `weights_out`.

    Uses `keys_b` / `idx_a` / `idx_b` as scratch. Enqueues work only (no host
    synchronization). Returns True if the sorted keys ended in `keys_a`, False
    if they ended in `keys_b`."""
    var layout = row_major(n)
    var ka = TileTensor(keys_a, layout)
    var kb = TileTensor(keys_b, layout)
    var ia = TileTensor(idx_a, layout)
    var ib = TileTensor(idx_b, layout)
    var win = TileTensor(weights_in, layout)
    var wout = TileTensor(weights_out, layout)

    comptime tile_sort = _tile_sort_kernel[type_of(layout)]
    comptime merge = _merge_pass_kernel[type_of(layout)]
    comptime permute = _permute_kernel[type_of(layout)]

    ctx.enqueue_function[tile_sort](
        ka,
        ia,
        Int32(n),
        grid_dim=ceildiv(n, BLOCK_CAP),
        block_dim=BLOCK_CAP,
    )
    var in_a = True
    var width = BLOCK_CAP
    var grid = ceildiv(n, _MERGE_BLOCK)
    while width < n:
        if in_a:
            ctx.enqueue_function[merge](
                ka,
                ia,
                kb,
                ib,
                Int32(n),
                Int32(width),
                grid_dim=grid,
                block_dim=_MERGE_BLOCK,
            )
        else:
            ctx.enqueue_function[merge](
                kb,
                ib,
                ka,
                ia,
                Int32(n),
                Int32(width),
                grid_dim=grid,
                block_dim=_MERGE_BLOCK,
            )
        in_a = not in_a
        width *= 2

    if in_a:
        ctx.enqueue_function[permute](
            win, ia, wout, Int32(n), grid_dim=grid, block_dim=_MERGE_BLOCK
        )
    else:
        ctx.enqueue_function[permute](
            win, ib, wout, Int32(n), grid_dim=grid, block_dim=_MERGE_BLOCK
        )
    return in_a
