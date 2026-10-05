"""Single-digest GPU `add_batch` backend (Stage 5d).

Parallel merge pass. The CPU digest compresses with a sequential greedy sweep;
the GPU uses the equivalent *parallel* formulation (prefix-sum + k-bucketing),
which yields an equally valid t-digest clustering — quantiles agree with the CPU
within fp32 tolerance, though the exact centroid count may differ by a few.

Pipeline for one block of up to `BLOCK_CAP` sorted (mean, weight) pairs:

  1. prefix-sum the weights          → cumulative weight per element
  2. cluster_id = floor(k(cum/W))    → monotone, so equal ids are contiguous
  3. head flag where id changes      → segment starts
  4. segmented reduce (Σw, Σ w·mean) → per-cluster totals
  5. compact segment ends            → new (mean = Σw·mean/Σw, weight = Σw)

The whole thing runs in a single block in shared memory. Arbitrary-length
batches are handled by the host wrapper, which sorts and tiles the input,
compresses each tile, and reduces the partial centroids hierarchically until the
survivors (plus the digest's existing centroids) fit in one block for a final
compress.

Everything here is Float32 (the GPU dtype). The scale function is reused on
device via `kdigest.scale` (Stage 5c) — one source of truth for `k`.
"""

from std.math import ceildiv, floor

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major, stack_allocation

from kdigest.scale import ScaleFunction
from kdigest.gpu.primitives import BLOCK_CAP, cpu_sort_pairs


comptime _DT = DType.float32
comptime _FLAG = DType.int32
comptime _cap_f32 = row_major[BLOCK_CAP]()


def _compress_block_kernel[
    S: ScaleFunction
](
    means: TileTensor[_DT, type_of(_cap_f32), MutAnyOrigin],
    weights: TileTensor[_DT, type_of(_cap_f32), MutAnyOrigin],
    out_means: TileTensor[_DT, type_of(_cap_f32), MutAnyOrigin],
    out_weights: TileTensor[_DT, type_of(_cap_f32), MutAnyOrigin],
    out_count: TileTensor[_FLAG, type_of(_cap_f32), MutAnyOrigin],
    compression: Float32,
    n: Int32,
) where (S.DT == _DT):
    """Compress up to BLOCK_CAP *sorted* (mean, weight) pairs into cluster
    centroids via parallel k-bucketing. Writes the compacted centroids to
    out_* and the survivor count to out_count[0]."""
    comptime assert means.flat_rank == 1 and weights.flat_rank == 1
    var w = stack_allocation[_DT, address_space=AddressSpace.SHARED](_cap_f32)
    var cum = stack_allocation[_DT, address_space=AddressSpace.SHARED](_cap_f32)
    var tmp = stack_allocation[_DT, address_space=AddressSpace.SHARED](_cap_f32)
    var cid = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var head = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var tid = thread_idx.x

    w[tid] = weights[tid] if Int32(tid) < n else Scalar[_DT](0)
    barrier()

    # 1. Inclusive prefix sum of weights (Hillis-Steele).
    cum[tid] = w[tid]
    barrier()
    var offset = 1
    while offset < BLOCK_CAP:
        if tid >= offset:
            tmp[tid] = rebind[Scalar[_DT]](cum[tid]) + rebind[Scalar[_DT]](
                cum[tid - offset]
            )
        else:
            tmp[tid] = cum[tid]
        barrier()
        cum[tid] = tmp[tid]
        barrier()
        offset *= 2

    var total = rebind[Scalar[_DT]](cum[BLOCK_CAP - 1])
    if total <= 0.0:
        total = Scalar[_DT](1)

    # 2. cluster id = floor(k(q_right)), q_right = cum/total.
    var s = S()
    var c = rebind[Scalar[S.DT]](compression)
    var tot = rebind[Scalar[S.DT]](total)
    var nrm = s.normalizer(c, tot)
    if Int32(tid) < n:
        var q_right = rebind[Scalar[S.DT]](cum[tid]) / tot
        var kv = s.k(q_right, nrm)
        cid[tid] = Int32(floor(rebind[Scalar[_DT]](kv)))
    else:
        cid[tid] = Int32(2147483647)  # pad into its own trailing bucket
    barrier()

    # 3. Head flag where the cluster id changes (element 0 is always a head).
    if tid == 0:
        head[tid] = Scalar[_FLAG](1)
    else:
        head[tid] = Scalar[_FLAG](1) if rebind[Scalar[_FLAG]](
            cid[tid]
        ) != rebind[Scalar[_FLAG]](cid[tid - 1]) else Scalar[_FLAG](0)
    barrier()

    # 4. Segmented inclusive scans of weight and weight*mean. Reuse `cum` for
    # the segmented weight sum and `tmp` for the segmented weight*mean sum.
    var sw = cum  # alias
    var smw = tmp
    sw[tid] = w[tid]
    smw[tid] = w[tid] * (
        rebind[Scalar[_DT]](means[tid]) if Int32(tid) < n else Scalar[_DT](0)
    )
    # Segment flags that evolve during the scan (copy of head).
    var fl = head
    barrier()

    var voff = 1
    while voff < BLOCK_CAP:
        var add_w = Scalar[_DT](0)
        var add_mw = Scalar[_DT](0)
        var merge_flag: Scalar[_FLAG]
        var cur_flag = rebind[Scalar[_FLAG]](fl[tid])
        if tid >= voff and cur_flag == 0:
            add_w = rebind[Scalar[_DT]](sw[tid - voff])
            add_mw = rebind[Scalar[_DT]](smw[tid - voff])
        if tid >= voff:
            var prev_flag = rebind[Scalar[_FLAG]](fl[tid - voff])
            merge_flag = Scalar[_FLAG](1) if (
                cur_flag != 0 or prev_flag != 0
            ) else Scalar[_FLAG](0)
        else:
            merge_flag = cur_flag
        barrier()
        sw[tid] = rebind[Scalar[_DT]](sw[tid]) + add_w
        smw[tid] = rebind[Scalar[_DT]](smw[tid]) + add_mw
        fl[tid] = merge_flag
        barrier()
        voff *= 2

    # 5. A segment's totals live at its last element: the position right before
    # the next head (or the last valid element). Emit there, compacting by the
    # exclusive scan of head flags recomputed from `head` ... but head was
    # consumed. Instead: an element is a segment END if it is the last valid
    # element, or the next element is a head. Recompute next-head from cid.
    var is_end = Scalar[_FLAG](0)
    if Int32(tid) < n:
        if Int32(tid) == n - 1:
            is_end = Scalar[_FLAG](1)
        else:
            var same = rebind[Scalar[_FLAG]](cid[tid]) == rebind[Scalar[_FLAG]](
                cid[tid + 1]
            )
            is_end = Scalar[_FLAG](0) if same else Scalar[_FLAG](1)
    barrier()

    # Exclusive scan of is_end over a fresh buffer to get output slots.
    var pos = head  # reuse (fl == head alias already consumed)
    pos[tid] = is_end
    barrier()
    var poff = 1
    while poff < BLOCK_CAP:
        if tid >= poff:
            cid[tid] = rebind[Scalar[_FLAG]](pos[tid]) + rebind[Scalar[_FLAG]](
                pos[tid - poff]
            )
        else:
            cid[tid] = pos[tid]
        barrier()
        pos[tid] = cid[tid]
        barrier()
        poff *= 2

    if is_end != 0:
        var slot = Int(rebind[Scalar[_FLAG]](pos[tid])) - 1
        var seg_w = rebind[Scalar[_DT]](sw[tid])
        var seg_mw = rebind[Scalar[_DT]](smw[tid])
        out_means[slot] = seg_mw / seg_w if seg_w > 0.0 else Scalar[_DT](0)
        out_weights[slot] = seg_w
    if Int32(tid) == n - 1:
        out_count[0] = pos[tid]
    if n == 0 and tid == 0:
        out_count[0] = Scalar[_FLAG](0)


# ======================================================================
# Host pipeline
# ======================================================================


struct _CompressBuffers(Movable):
    """Device scratch for one single-block compress, allocated once per
    `gpu_compress_batch` call and reused by every tile (previously each tile
    allocated five fresh device buffers)."""

    var means: DeviceBuffer[_DT]
    var weights: DeviceBuffer[_DT]
    var out_means: DeviceBuffer[_DT]
    var out_weights: DeviceBuffer[_DT]
    var count: DeviceBuffer[_FLAG]

    def __init__(out self, ctx: DeviceContext) raises:
        self.means = ctx.enqueue_create_buffer[_DT](BLOCK_CAP)
        self.weights = ctx.enqueue_create_buffer[_DT](BLOCK_CAP)
        self.out_means = ctx.enqueue_create_buffer[_DT](BLOCK_CAP)
        self.out_weights = ctx.enqueue_create_buffer[_DT](BLOCK_CAP)
        self.count = ctx.enqueue_create_buffer[_FLAG](BLOCK_CAP)


def _compress_once[
    S: ScaleFunction
](
    ctx: DeviceContext,
    bufs: _CompressBuffers,
    means: List[Float32],
    weights: List[Float32],
    lo: Int,
    hi: Int,
    compression: Float32,
    mut out_m: List[Float32],
    mut out_w: List[Float32],
) raises where (S.DT == _DT):
    """Compress the sorted slice `[lo, hi)` (<= BLOCK_CAP pairs) in one block,
    reusing `bufs`, and *append* the surviving centroids to `out_m` / `out_w`.
    """
    var n = hi - lo
    if n <= 0:
        return

    with bufs.means.map_to_host() as hm, bufs.weights.map_to_host() as hw:
        var tm = TileTensor(hm, _cap_f32)
        var tw = TileTensor(hw, _cap_f32)
        for i in range(n):
            tm[i] = means[lo + i]
            tw[i] = weights[lo + i]
    var tm = TileTensor(bufs.means, _cap_f32)
    var tw = TileTensor(bufs.weights, _cap_f32)
    var tom = TileTensor(bufs.out_means, _cap_f32)
    var tow = TileTensor(bufs.out_weights, _cap_f32)
    var tc = TileTensor(bufs.count, _cap_f32)

    comptime kernel = _compress_block_kernel[S]
    ctx.enqueue_function[kernel](
        tm,
        tw,
        tom,
        tow,
        tc,
        compression,
        Int32(n),
        grid_dim=1,
        block_dim=BLOCK_CAP,
    )

    var kept: Int
    with bufs.count.map_to_host() as hc:
        var t = TileTensor(hc, _cap_f32)
        kept = Int(rebind[Scalar[_FLAG]](t[0]))
    with bufs.out_means.map_to_host() as hm, bufs.out_weights.map_to_host() as hw:
        var rm = TileTensor(hm, _cap_f32)
        var rw = TileTensor(hw, _cap_f32)
        for i in range(kept):
            out_m.append(rebind[Scalar[_DT]](rm[i]))
            out_w.append(rebind[Scalar[_DT]](rw[i]))


def gpu_compress_batch[
    S: ScaleFunction
](
    means: List[Float32], weights: List[Float32], compression: Float32
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Convenience overload: creates a `DeviceContext` for this call. Callers
    that compress repeatedly should create one context and use the
    `gpu_compress_batch[S](ctx, ...)` overload instead."""
    var ctx = DeviceContext()
    return gpu_compress_batch[S](ctx, means, weights, compression)


def gpu_compress_batch[
    S: ScaleFunction
](
    ctx: DeviceContext,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Compress an arbitrary-length (mean, weight) set into t-digest centroids
    on the GPU, using scale function `S` and the caller's `ctx`.

    The input need not be sorted. It is sorted once on the host (O(n log n)),
    split into BLOCK_CAP tiles, each compressed on-device, and the partial
    centroids reduced hierarchically until a single final block yields the
    result. One set of device buffers is allocated per call and reused by every
    tile.
    """
    var n = len(means)
    if n == 0:
        return (List[Float32](), List[Float32]())

    var bufs = _CompressBuffers(ctx)

    var cur_m = means.copy()
    var cur_w = weights.copy()

    # Compress in BLOCK_CAP tiles; repeat while the survivors exceed one block.
    # Each pass strictly shrinks the set (a tile compresses to <= its size), so
    # this terminates. Re-sorting between passes keeps every tile mean-sorted.
    while True:
        cpu_sort_pairs(cur_m, cur_w)
        var m = len(cur_m)
        var next_m = List[Float32](capacity=m)
        var next_w = List[Float32](capacity=m)
        var p = 0
        while p < m:
            var hi = min(p + BLOCK_CAP, m)
            _compress_once[S](
                ctx, bufs, cur_m, cur_w, p, hi, compression, next_m, next_w
            )
            p = hi
        cur_m = next_m^
        cur_w = next_w^
        if m <= BLOCK_CAP:
            # That pass was the single final block.
            return (cur_m^, cur_w^)
