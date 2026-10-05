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
batches are handled by `gpu_compress_batch`, which keeps everything on the
device: a stable GPU sort (`kdigest.gpu.sort`), then hierarchical passes where
each pass compresses *every* tile in one multi-block launch and a device-side
gather packs the survivors for the next pass. Only the per-pass survivor counts
and the final <= BLOCK_CAP centroids are read back to the host.

Everything here is Float32 (the GPU dtype). The scale function is reused on
device via `kdigest.scale` (Stage 5c) — one source of truth for `k`.
"""

from std.math import ceildiv, floor

from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from kdigest.scale import ScaleFunction
from kdigest.gpu.primitives import BLOCK_CAP, cpu_sort_pairs
from kdigest.gpu.sort import gpu_sort_by_mean


comptime _DT = DType.float32
comptime _FLAG = DType.int32
comptime _cap_f32 = row_major[BLOCK_CAP]()


def _compress_block_kernel[
    S: ScaleFunction, IL: TensorLayout, CL: TensorLayout
](
    means: TileTensor[_DT, IL, MutAnyOrigin],
    weights: TileTensor[_DT, IL, MutAnyOrigin],
    out_means: TileTensor[_DT, IL, MutAnyOrigin],
    out_weights: TileTensor[_DT, IL, MutAnyOrigin],
    out_count: TileTensor[_FLAG, CL, MutAnyOrigin],
    compression: Float32,
    total_n: Int32,
) where (S.DT == _DT):
    """Tiled compress: block `b` compresses the *sorted* tile
    `[b*BLOCK_CAP, min((b+1)*BLOCK_CAP, total_n))` into cluster centroids via
    parallel k-bucketing, writing them to `out_*[b*BLOCK_CAP + slot]` and the
    survivor count to `out_count[b]`. Launched with one block it is the original
    single-block compress; launched with `ceildiv(total_n, BLOCK_CAP)` blocks it
    compresses every tile of a pass in a single launch."""
    comptime assert means.flat_rank == 1 and weights.flat_rank == 1
    comptime assert out_means.flat_rank == 1 and out_weights.flat_rank == 1
    comptime assert out_count.flat_rank == 1
    var blk = Int(block_idx.x)
    var base = blk * BLOCK_CAP
    var n = Int32(min(BLOCK_CAP, Int(total_n) - base))
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

    w[tid] = weights[base + tid] if Int32(tid) < n else Scalar[_DT](0)
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
        rebind[Scalar[_DT]](means[base + tid]) if Int32(tid)
        < n else Scalar[_DT](0)
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
    var is_head = False
    if Int32(tid) < n:
        if Int32(tid) == n - 1:
            is_end = Scalar[_FLAG](1)
        else:
            var same = rebind[Scalar[_FLAG]](cid[tid]) == rebind[Scalar[_FLAG]](
                cid[tid + 1]
            )
            is_end = Scalar[_FLAG](0) if same else Scalar[_FLAG](1)
        is_head = tid == 0 or rebind[Scalar[_FLAG]](cid[tid]) != rebind[
            Scalar[_FLAG]
        ](cid[tid - 1])
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

    # Each segment's head records its (smallest) mean, indexed by segment, in
    # the now-free `w` scratch: segment index = number of ends before the head.
    var seg_lo = w
    if is_head:
        var seg = 0 if tid == 0 else Int(rebind[Scalar[_FLAG]](pos[tid - 1]))
        seg_lo[seg] = means[base + tid]
    barrier()

    if is_end != 0:
        var slot = Int(rebind[Scalar[_FLAG]](pos[tid])) - 1
        var seg_w = rebind[Scalar[_DT]](sw[tid])
        var seg_mw = rebind[Scalar[_DT]](smw[tid])
        var mean = seg_mw / seg_w if seg_w > 0.0 else Scalar[_DT](0)
        # Clamp to the segment's [first, last] value. The fp32 weighted mean can
        # round a hair outside that range; clamping guarantees centroids come
        # out in non-decreasing order across segments *and* tiles, which is what
        # lets the device pipeline skip re-sorting between passes.
        var lo_v = rebind[Scalar[_DT]](seg_lo[slot])
        var hi_v = rebind[Scalar[_DT]](means[base + tid])
        mean = min(max(mean, lo_v), hi_v)
        out_means[base + slot] = mean
        out_weights[base + slot] = seg_w
    if Int32(tid) == n - 1:
        out_count[blk] = pos[tid]
    if n <= 0 and tid == 0:
        out_count[blk] = Scalar[_FLAG](0)


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

    comptime kernel = _compress_block_kernel[
        S, type_of(_cap_f32), type_of(_cap_f32)
    ]
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


def gpu_compress_batch_host_tiled[
    S: ScaleFunction
](
    ctx: DeviceContext,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Reference path (the pre-step-3 implementation), kept for parity tests
    and the crossover benchmark: host sort, then one upload -> single-block
    launch -> readback round trip *per tile*. Prefer `gpu_compress_batch`."""
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


# ----------------------------------------------------------------------
# Device-resident pipeline
# ----------------------------------------------------------------------


struct CompressWorkspace(Movable):
    """Reusable device buffers + pinned host staging for `gpu_compress_batch`.

    Allocated once and grown (geometrically) to the largest batch seen, so
    repeated calls pay no per-call allocation. Uploads/readbacks go through the
    pinned host buffers with async copies, so a call synchronizes with the GPU
    once at the end plus once per intermediate reduction pass (to learn the
    survivor count) — instead of a blocking `map_to_host` per transfer."""

    var capacity: Int  # max points
    var tile_capacity: Int  # max tiles = ceildiv(capacity, BLOCK_CAP)
    var keys_a: DeviceBuffer[_DT]
    var keys_b: DeviceBuffer[_DT]
    var w_in: DeviceBuffer[_DT]
    var w_sorted: DeviceBuffer[_DT]
    var idx_a: DeviceBuffer[_FLAG]
    var idx_b: DeviceBuffer[_FLAG]
    var counts: DeviceBuffer[_FLAG]  # tile_capacity + 1
    var offsets: DeviceBuffer[_FLAG]  # tile_capacity + 1; [tiles] = total
    var h_m: HostBuffer[_DT]  # capacity (upload staging)
    var h_w: HostBuffer[_DT]
    var h_out_m: HostBuffer[_DT]  # BLOCK_CAP (final centroids)
    var h_out_w: HostBuffer[_DT]
    var h_int: HostBuffer[_FLAG]  # 1 (survivor total / final count)

    def __init__(out self, ctx: DeviceContext, capacity: Int = 0) raises:
        var cap = max(capacity, BLOCK_CAP)
        var tcap = ceildiv(cap, BLOCK_CAP)
        self.capacity = cap
        self.tile_capacity = tcap
        self.keys_a = ctx.enqueue_create_buffer[_DT](cap)
        self.keys_b = ctx.enqueue_create_buffer[_DT](cap)
        self.w_in = ctx.enqueue_create_buffer[_DT](cap)
        self.w_sorted = ctx.enqueue_create_buffer[_DT](cap)
        self.idx_a = ctx.enqueue_create_buffer[_FLAG](cap)
        self.idx_b = ctx.enqueue_create_buffer[_FLAG](cap)
        self.counts = ctx.enqueue_create_buffer[_FLAG](tcap + 1)
        self.offsets = ctx.enqueue_create_buffer[_FLAG](tcap + 1)
        self.h_m = ctx.enqueue_create_host_buffer[_DT](cap)
        self.h_w = ctx.enqueue_create_host_buffer[_DT](cap)
        self.h_out_m = ctx.enqueue_create_host_buffer[_DT](BLOCK_CAP)
        self.h_out_w = ctx.enqueue_create_host_buffer[_DT](BLOCK_CAP)
        self.h_int = ctx.enqueue_create_host_buffer[_FLAG](1)
        ctx.synchronize()  # host buffers must be ready before host writes

    def reserve(mut self, ctx: DeviceContext, n: Int) raises:
        """Ensure capacity for `n` points (grows to max(n, 2 * capacity))."""
        if n <= self.capacity:
            return
        self = CompressWorkspace(ctx, max(n, 2 * self.capacity))


def _scan_counts_kernel[
    CL: TensorLayout
](
    counts: TileTensor[_FLAG, CL, MutAnyOrigin],
    offsets: TileTensor[_FLAG, CL, MutAnyOrigin],
    tiles: Int32,
):
    """Single block: `offsets[b] = sum(counts[0:b])` for b < tiles, and
    `offsets[tiles] = total`. Each thread scans a contiguous chunk serially,
    then the per-thread sums are scanned across the block (Hillis-Steele), so
    any number of tiles fits in one launch."""
    comptime assert counts.flat_rank == 1 and offsets.flat_rank == 1
    var sums = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var tmp = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var tid = Int(thread_idx.x)
    var t = Int(tiles)
    var chunk = ceildiv(t, BLOCK_CAP)
    var lo = min(tid * chunk, t)
    var hi = min(lo + chunk, t)
    var local = Int32(0)
    for i in range(lo, hi):
        local += rebind[Scalar[_FLAG]](counts[i])
    sums[tid] = local
    barrier()
    var off = 1
    while off < BLOCK_CAP:
        if tid >= off:
            tmp[tid] = rebind[Scalar[_FLAG]](sums[tid]) + rebind[Scalar[_FLAG]](
                sums[tid - off]
            )
        else:
            tmp[tid] = sums[tid]
        barrier()
        sums[tid] = tmp[tid]
        barrier()
        off *= 2
    var run = rebind[Scalar[_FLAG]](sums[tid]) - local
    for i in range(lo, hi):
        offsets[i] = run
        run += rebind[Scalar[_FLAG]](counts[i])
    if tid == BLOCK_CAP - 1:
        offsets[t] = sums[tid]


def _gather_tiles_kernel[
    IL: TensorLayout, CL: TensorLayout
](
    tiled_m: TileTensor[_DT, IL, MutAnyOrigin],
    tiled_w: TileTensor[_DT, IL, MutAnyOrigin],
    dense_m: TileTensor[_DT, IL, MutAnyOrigin],
    dense_w: TileTensor[_DT, IL, MutAnyOrigin],
    counts: TileTensor[_FLAG, CL, MutAnyOrigin],
    offsets: TileTensor[_FLAG, CL, MutAnyOrigin],
):
    """Pack each tile's survivors (`tiled_*[b*BLOCK_CAP : +counts[b]]`) densely
    to `dense_*[offsets[b] : ...]`. Tiles are in global mean order and each
    tile's centroids are mean-sorted, so the packed result is the sorted input
    of the next pass — no re-sort needed."""
    comptime assert tiled_m.flat_rank == 1 and tiled_w.flat_rank == 1
    comptime assert dense_m.flat_rank == 1 and dense_w.flat_rank == 1
    comptime assert counts.flat_rank == 1 and offsets.flat_rank == 1
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var c = Int(rebind[Scalar[_FLAG]](counts[b]))
    if tid < c:
        var dst = Int(rebind[Scalar[_FLAG]](offsets[b])) + tid
        dense_m[dst] = tiled_m[b * BLOCK_CAP + tid]
        dense_w[dst] = tiled_w[b * BLOCK_CAP + tid]


def _device_compress_passes[
    S: ScaleFunction
](
    ctx: DeviceContext,
    ws: CompressWorkspace,
    src_m: DeviceBuffer[_DT],
    src_w: DeviceBuffer[_DT],
    scratch_m: DeviceBuffer[_DT],
    scratch_w: DeviceBuffer[_DT],
    n: Int,
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Hierarchical compress of `n` mean-sorted pairs already on the device in
    `src_*`. Each pass: ONE multi-block compress launch into `scratch_*`, a
    one-block scan of the per-tile survivor counts, and a gather that packs the
    survivors back into `src_*` — all on the device. The host reads a single
    integer per intermediate pass (the survivor total, to size the next launch)
    and the final <= BLOCK_CAP centroids once at the end."""
    var layout = row_major(n)
    var clayout = row_major(ws.tile_capacity + 1)
    var t_src_m = TileTensor(src_m, layout)
    var t_src_w = TileTensor(src_w, layout)
    var t_scr_m = TileTensor(scratch_m, layout)
    var t_scr_w = TileTensor(scratch_w, layout)
    var t_cnt = TileTensor(ws.counts, clayout)
    var t_off = TileTensor(ws.offsets, clayout)

    comptime compress = _compress_block_kernel[
        S, type_of(layout), type_of(clayout)
    ]
    comptime scan = _scan_counts_kernel[type_of(clayout)]
    comptime gather = _gather_tiles_kernel[type_of(layout), type_of(clayout)]

    var m = n
    while True:
        var tiles = ceildiv(m, BLOCK_CAP)
        ctx.enqueue_function[compress](
            t_src_m,
            t_src_w,
            t_scr_m,
            t_scr_w,
            t_cnt,
            compression,
            Int32(m),
            grid_dim=tiles,
            block_dim=BLOCK_CAP,
        )
        if tiles == 1:
            # Final block: one async readback of count + centroids, one sync.
            ctx.enqueue_copy(ws.h_int, ws.counts.create_sub_buffer[_FLAG](0, 1))
            ctx.enqueue_copy(
                ws.h_out_m.create_sub_buffer[_DT](0, m),
                scratch_m.create_sub_buffer[_DT](0, m),
            )
            ctx.enqueue_copy(
                ws.h_out_w.create_sub_buffer[_DT](0, m),
                scratch_w.create_sub_buffer[_DT](0, m),
            )
            ctx.synchronize()
            var kept = Int(ws.h_int[0])
            var out_m = List[Float32](capacity=kept)
            var out_w = List[Float32](capacity=kept)
            for i in range(kept):
                out_m.append(ws.h_out_m[i])
                out_w.append(ws.h_out_w[i])
            return (out_m^, out_w^)

        ctx.enqueue_function[scan](
            t_cnt, t_off, Int32(tiles), grid_dim=1, block_dim=BLOCK_CAP
        )
        ctx.enqueue_function[gather](
            t_scr_m,
            t_scr_w,
            t_src_m,
            t_src_w,
            t_cnt,
            t_off,
            grid_dim=tiles,
            block_dim=BLOCK_CAP,
        )
        # The only per-pass host sync: learn the survivor total.
        ctx.enqueue_copy(
            ws.h_int, ws.offsets.create_sub_buffer[_FLAG](tiles, 1)
        )
        ctx.synchronize()
        m = Int(ws.h_int[0])


def _stage_input(
    ctx: DeviceContext,
    h_m: HostBuffer[_DT],
    h_w: HostBuffer[_DT],
    means: List[Float32],
    weights: List[Float32],
    dst_m: DeviceBuffer[_DT],
    dst_w: DeviceBuffer[_DT],
) raises:
    """Copy the batch into pinned host staging and enqueue async uploads."""
    var n = len(means)
    var pm = h_m.unsafe_ptr()
    var pw = h_w.unsafe_ptr()
    for i in range(n):
        pm[i] = means[i]
        pw[i] = weights[i]
    ctx.enqueue_copy(
        dst_m.create_sub_buffer[_DT](0, n), h_m.create_sub_buffer[_DT](0, n)
    )
    ctx.enqueue_copy(
        dst_w.create_sub_buffer[_DT](0, n), h_w.create_sub_buffer[_DT](0, n)
    )


def gpu_compress_batch_host_sort[
    S: ScaleFunction
](
    ctx: DeviceContext,
    mut ws: CompressWorkspace,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Step 3, part A: host sort (O(n log n)) + device-resident tiling. Kept so
    the benchmark and parity tests can isolate the GPU sort."""
    var n = len(means)
    if n == 0:
        return (List[Float32](), List[Float32]())
    var sm = means.copy()
    var sw = weights.copy()
    cpu_sort_pairs(sm, sw)
    ws.reserve(ctx, n)
    _stage_input(ctx, ws.h_m, ws.h_w, sm, sw, ws.keys_a, ws.w_sorted)
    return _device_compress_passes[S](
        ctx, ws, ws.keys_a, ws.w_sorted, ws.keys_b, ws.w_in, n, compression
    )


def gpu_compress_batch_host_sort[
    S: ScaleFunction
](
    ctx: DeviceContext,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    var ws = CompressWorkspace(ctx, len(means))
    return gpu_compress_batch_host_sort[S](ctx, ws, means, weights, compression)


def gpu_compress_batch[
    S: ScaleFunction
](
    ctx: DeviceContext,
    mut ws: CompressWorkspace,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Compress an arbitrary-length (mean, weight) set into t-digest centroids
    on the GPU, using scale function `S`, the caller's `ctx`, and a reusable
    workspace `ws` (the fast path for repeated calls).

    Fully device-resident: one async upload, a stable GPU sort
    (`kdigest.gpu.sort`), then the hierarchical compress passes (compress,
    scan, gather — all on the device). The host synchronizes once per
    intermediate pass (one integer) and once for the final <= BLOCK_CAP
    centroids. Output is identical to `gpu_compress_batch_host_tiled`."""
    var n = len(means)
    if n == 0:
        return (List[Float32](), List[Float32]())
    ws.reserve(ctx, n)
    _stage_input(ctx, ws.h_m, ws.h_w, means, weights, ws.keys_a, ws.w_in)
    var sorted_in_a = gpu_sort_by_mean(
        ctx, ws.keys_a, ws.keys_b, ws.idx_a, ws.idx_b, ws.w_in, ws.w_sorted, n
    )
    # Sorted (mean, weight) now live in (keys_?, w_sorted); the other key
    # buffer and w_in serve as the passes' scratch.
    if sorted_in_a:
        return _device_compress_passes[S](
            ctx, ws, ws.keys_a, ws.w_sorted, ws.keys_b, ws.w_in, n, compression
        )
    return _device_compress_passes[S](
        ctx, ws, ws.keys_b, ws.w_sorted, ws.keys_a, ws.w_in, n, compression
    )


def gpu_compress_batch[
    S: ScaleFunction
](
    ctx: DeviceContext,
    means: List[Float32],
    weights: List[Float32],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """One-shot overload: allocates a workspace for this call. Repeated callers
    should hold a `CompressWorkspace` (or a `GpuSession`) and reuse it."""
    var ws = CompressWorkspace(ctx, len(means))
    return gpu_compress_batch[S](ctx, ws, means, weights, compression)


def gpu_compress_batch[
    S: ScaleFunction
](
    means: List[Float32], weights: List[Float32], compression: Float32
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Convenience overload: creates a `DeviceContext` and workspace."""
    var ctx = DeviceContext()
    return gpu_compress_batch[S](ctx, means, weights, compression)
