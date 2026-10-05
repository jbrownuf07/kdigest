"""Grouped multi-digest GPU backend (Stage 5e) — the headline win.

One t-digest *per key* (metric / customer / sensor / window). Thousands of
independent small digests are compressed in a single kernel launch with **one
GPU block per key**, so the parallelism is *across* digests — this sidesteps the
single-digest sequential bottleneck entirely and mirrors how RAPIDS/libcudf
powers Spark's `approx_percentile`.

Pipeline (host):
  1. Two linear passes: count samples per key, then scatter values into pinned
     staging so each key's slice is contiguous (no host sort).
  2. Launch `K` blocks; block `b` bitonic-sorts its slice in shared memory and
     compresses it with the Stage 5d parallel k-bucketing merge, writing its
     centroids compactly at its own input offset.
  3. Keys whose slice exceeds BLOCK_CAP fall back to the single-digest
     `gpu_compress_batch` (device-resident sort + tiling), so correctness is
     independent of per-key volume.

Device buffers and pinned staging live in a reusable `GroupedWorkspace`.

Everything is Float32 (the GPU dtype); the scale function is reused on-device
from `kdigest.scale` (one source of truth).
"""

from std.math import floor

from max.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from kdigest.scale import ScaleFunction
from kdigest.gpu.primitives import BLOCK_CAP
from kdigest.gpu.add_batch import CompressWorkspace, gpu_compress_batch


comptime _DT = DType.float32
comptime _FLAG = DType.int32


def _grouped_compress_kernel[
    S: ScaleFunction,
    VL: TensorLayout,
    OL: TensorLayout,
    CL: TensorLayout,
    NL: TensorLayout,
](
    values: TileTensor[_DT, VL, MutAnyOrigin],  # flattened, key-contiguous
    offsets: TileTensor[_FLAG, OL, MutAnyOrigin],  # length K+1
    out_means: TileTensor[_DT, CL, MutAnyOrigin],  # same size as values
    out_weights: TileTensor[_DT, CL, MutAnyOrigin],  # same size as values
    out_counts: TileTensor[_FLAG, NL, MutAnyOrigin],  # length K
    compression: Float32,
) where (S.DT == _DT):
    """One block per key. Block `b` sorts and compresses the key's unit-weight
    values `values[offsets[b] : offsets[b+1]]` into centroids via parallel
    k-bucketing, writing them to `out_*[offsets[b] : ...]`. Assumes each key's
    slice length <= BLOCK_CAP (the host routes larger keys elsewhere)."""
    comptime assert values.flat_rank == 1 and offsets.flat_rank == 1
    comptime assert out_means.flat_rank == 1 and out_counts.flat_rank == 1

    var b = block_idx.x
    var start = Int(rebind[Scalar[_FLAG]](offsets[b]))
    var end = Int(rebind[Scalar[_FLAG]](offsets[b + 1]))
    var n = end - start

    var m = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var w = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var cum = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var tmp = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var cid = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var fl = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var tid = thread_idx.x

    # Load this key's slice (unsorted) and sort it in shared memory with a
    # bitonic network; padding (+MAX) sorts to the end. Unit weights, so no
    # payload needs to travel with the values.
    if tid < n:
        m[tid] = values[start + tid]
        w[tid] = Scalar[_DT](1)
    else:
        m[tid] = Scalar[_DT].MAX_FINITE
        w[tid] = Scalar[_DT](0)
    barrier()
    var bk = 2
    while bk <= BLOCK_CAP:
        var bj = bk // 2
        while bj > 0:
            var ixj = tid ^ bj
            if ixj > tid:
                var a = rebind[Scalar[_DT]](m[tid])
                var c = rebind[Scalar[_DT]](m[ixj])
                var ascending = (tid & bk) == 0
                if (ascending and a > c) or (not ascending and a < c):
                    m[tid] = c
                    m[ixj] = a
            barrier()
            bj //= 2
        bk *= 2
    if tid >= n:
        m[tid] = Scalar[_DT](0)  # restore zero padding for the w*m scan
    barrier()

    # 1. Inclusive prefix sum of weights.
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

    # 2. cluster id = floor(k(cum/total)).
    var s = S()
    var cc = rebind[Scalar[S.DT]](compression)
    var tot = rebind[Scalar[S.DT]](total)
    var nrm = s.normalizer(cc, tot)
    if tid < n:
        var q_right = rebind[Scalar[S.DT]](cum[tid]) / tot
        cid[tid] = Int32(floor(rebind[Scalar[_DT]](s.k(q_right, nrm))))
    else:
        cid[tid] = Int32(2147483647)
    barrier()

    # 3. Head flags.
    if tid == 0:
        fl[tid] = Scalar[_FLAG](1)
    else:
        fl[tid] = Scalar[_FLAG](1) if rebind[Scalar[_FLAG]](cid[tid]) != rebind[
            Scalar[_FLAG]
        ](cid[tid - 1]) else Scalar[_FLAG](0)
    barrier()

    # 4. Segmented inclusive scan of weight and weight*mean.
    var sw = cum
    var smw = tmp
    sw[tid] = w[tid]
    smw[tid] = rebind[Scalar[_DT]](w[tid]) * rebind[Scalar[_DT]](m[tid])
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

    # 5. Segment ends → emit compacted centroids. is_end = last valid element or
    # next element has a different cluster id.
    var is_end = Scalar[_FLAG](0)
    if tid < n:
        if tid == n - 1:
            is_end = Scalar[_FLAG](1)
        else:
            var same = rebind[Scalar[_FLAG]](cid[tid]) == rebind[Scalar[_FLAG]](
                cid[tid + 1]
            )
            is_end = Scalar[_FLAG](0) if same else Scalar[_FLAG](1)
    barrier()

    # Exclusive-scan is_end to get output slots (reuse cid as scratch, fl→pos).
    var pos = fl
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

    # Compact output: a key yields at most as many centroids as samples, so
    # its centroids go at its *input* offset (output size == input size, not
    # num_keys * BLOCK_CAP).
    var base = start
    if is_end != 0:
        var slot = Int(rebind[Scalar[_FLAG]](pos[tid])) - 1
        var seg_w = rebind[Scalar[_DT]](sw[tid])
        var seg_mw = rebind[Scalar[_DT]](smw[tid])
        out_means[base + slot] = seg_mw / seg_w if seg_w > 0.0 else Scalar[_DT](
            0
        )
        out_weights[base + slot] = seg_w
    if tid == n - 1:
        out_counts[b] = pos[tid]
    if n == 0 and tid == 0:
        out_counts[b] = Scalar[_FLAG](0)


struct GroupedResult(Movable):
    """Per-key compressed centroids returned by `gpu_grouped_compress`.

    `means[k]` / `weights[k]` are the centroid arrays for key `k`; `mins[k]` and
    `maxs[k]` are that key's observed value range (for tail interpolation)."""

    var means: List[List[Float32]]
    var weights: List[List[Float32]]
    var mins: List[Float32]
    var maxs: List[Float32]

    def __init__(out self):
        self.means = List[List[Float32]]()
        self.weights = List[List[Float32]]()
        self.mins = List[Float32]()
        self.maxs = List[Float32]()


struct GroupedWorkspace(Movable):
    """Reusable device buffers + pinned host staging for `gpu_grouped_compress`
    (grown geometrically to the largest batch / key count seen), plus the
    `CompressWorkspace` used by the oversized-key fallback."""

    var val_capacity: Int
    var key_capacity: Int
    var values: DeviceBuffer[_DT]
    var offsets: DeviceBuffer[_FLAG]
    var out_m: DeviceBuffer[_DT]
    var out_w: DeviceBuffer[_DT]
    var counts: DeviceBuffer[_FLAG]
    var h_values: HostBuffer[_DT]
    var h_offsets: HostBuffer[_FLAG]
    var h_out_m: HostBuffer[_DT]
    var h_out_w: HostBuffer[_DT]
    var h_counts: HostBuffer[_FLAG]
    var batch: CompressWorkspace

    def __init__(
        out self, ctx: DeviceContext, values: Int = 0, keys: Int = 0
    ) raises:
        var vc = max(values, BLOCK_CAP)
        var kc = max(keys, 64)
        self.val_capacity = vc
        self.key_capacity = kc
        self.values = ctx.enqueue_create_buffer[_DT](vc)
        self.offsets = ctx.enqueue_create_buffer[_FLAG](kc + 1)
        self.out_m = ctx.enqueue_create_buffer[_DT](vc)
        self.out_w = ctx.enqueue_create_buffer[_DT](vc)
        self.counts = ctx.enqueue_create_buffer[_FLAG](kc)
        self.h_values = ctx.enqueue_create_host_buffer[_DT](vc)
        self.h_offsets = ctx.enqueue_create_host_buffer[_FLAG](kc + 1)
        self.h_out_m = ctx.enqueue_create_host_buffer[_DT](vc)
        self.h_out_w = ctx.enqueue_create_host_buffer[_DT](vc)
        self.h_counts = ctx.enqueue_create_host_buffer[_FLAG](kc)
        self.batch = CompressWorkspace(ctx)
        ctx.synchronize()

    def reserve(mut self, ctx: DeviceContext, values: Int, keys: Int) raises:
        if values <= self.val_capacity and keys <= self.key_capacity:
            return
        var fresh = GroupedWorkspace(
            ctx,
            max(values, 2 * self.val_capacity),
            max(keys, 2 * self.key_capacity),
        )
        # Keep the (already grown) batch workspace for the fallback path.
        swap(fresh.batch, self.batch)
        self = fresh^


def gpu_grouped_compress[
    S: ScaleFunction
](
    keys: List[Int32],
    values: List[Float64],
    num_keys: Int,
    compression: Float32,
) raises -> GroupedResult where (S.DT == _DT):
    """Convenience overload: creates a `DeviceContext` and workspace."""
    var ctx = DeviceContext()
    return gpu_grouped_compress[S](ctx, keys, values, num_keys, compression)


def gpu_grouped_compress[
    S: ScaleFunction
](
    ctx: DeviceContext,
    keys: List[Int32],
    values: List[Float64],
    num_keys: Int,
    compression: Float32,
) raises -> GroupedResult where (S.DT == _DT):
    """One-shot overload on the caller's `ctx` (allocates a workspace)."""
    var ws = GroupedWorkspace(ctx, len(values), num_keys)
    return gpu_grouped_compress[S](ctx, ws, keys, values, num_keys, compression)


def gpu_grouped_compress[
    S: ScaleFunction
](
    ctx: DeviceContext,
    mut ws: GroupedWorkspace,
    keys: List[Int32],
    values: List[Float64],
    num_keys: Int,
    compression: Float32,
) raises -> GroupedResult where (S.DT == _DT):
    """Compress a (key, value) batch into one centroid set per key on the GPU.

    `keys[i]` is a dense key id in `[0, num_keys)` for `values[i]`. Keys with no
    samples yield empty centroid lists.

    Host work is two linear passes: count per key, then scatter values straight
    into pinned staging in small-key-contiguous order (no host sort — each
    block sorts its own key in shared memory). Then one async upload, one
    launch, one async readback and a single synchronize. Keys with more than
    BLOCK_CAP samples go through `gpu_compress_batch` with the reused
    `ws.batch` workspace.
    """
    var result = GroupedResult()
    var n = len(values)
    result.means = List[List[Float32]](length=num_keys, fill=List[Float32]())
    result.weights = List[List[Float32]](length=num_keys, fill=List[Float32]())
    result.mins = List[Float32](length=num_keys, fill=Float32(0))
    result.maxs = List[Float32](length=num_keys, fill=Float32(0))
    if n == 0:
        return result^

    # Pass 1: per-key counts. Small keys (<= BLOCK_CAP) get a launch block;
    # `block_of[k]` is that block id, or -1 for empty / oversized keys.
    var counts = List[Int](length=num_keys, fill=0)
    for i in range(n):
        counts[Int(keys[i])] += 1
    var block_of = List[Int](length=num_keys, fill=-1)
    var small_keys = List[Int]()
    var big_keys = List[Int]()
    var small_total = 0
    for k in range(num_keys):
        var c = counts[k]
        if c == 0:
            continue
        if c <= BLOCK_CAP:
            block_of[k] = len(small_keys)
            small_keys.append(k)
            small_total += c
        else:
            big_keys.append(k)
    var num_small = len(small_keys)
    ws.reserve(ctx, small_total, num_small)

    # Block offsets (exclusive prefix sum over small keys), written straight
    # into pinned staging; `cursor[b]` tracks the next write slot per block.
    var h_off = ws.h_offsets.unsafe_ptr()
    var cursor = List[Int](length=num_small, fill=0)
    var run = 0
    for b in range(num_small):
        h_off[b] = Int32(run)
        cursor[b] = run
        run += counts[small_keys[b]]
    h_off[num_small] = Int32(run)

    # Big keys collect their values separately for the fallback.
    var big_slot = List[Int](length=num_keys, fill=-1)
    var big_vals = List[List[Float32]]()
    for j in range(len(big_keys)):
        big_slot[big_keys[j]] = j
        big_vals.append(List[Float32](capacity=counts[big_keys[j]]))

    # Pass 2: scatter values + per-key min/max.
    var seen = List[Bool](length=num_keys, fill=False)
    var h_val = ws.h_values.unsafe_ptr()
    for i in range(n):
        var k = Int(keys[i])
        var v = Float32(values[i])
        if not seen[k]:
            seen[k] = True
            result.mins[k] = v
            result.maxs[k] = v
        else:
            if v < result.mins[k]:
                result.mins[k] = v
            if v > result.maxs[k]:
                result.maxs[k] = v
        var b = block_of[k]
        if b >= 0:
            h_val[cursor[b]] = v
            cursor[b] += 1
        else:
            big_vals[big_slot[k]].append(v)

    if num_small > 0:
        ctx.enqueue_copy(
            ws.values.create_sub_buffer[_DT](0, small_total),
            ws.h_values.create_sub_buffer[_DT](0, small_total),
        )
        ctx.enqueue_copy(
            ws.offsets.create_sub_buffer[_FLAG](0, num_small + 1),
            ws.h_offsets.create_sub_buffer[_FLAG](0, num_small + 1),
        )
        var v_layout = row_major(ws.val_capacity)
        var o_layout = row_major(ws.key_capacity + 1)
        var oc_layout = row_major(ws.key_capacity)
        comptime kernel = _grouped_compress_kernel[
            S,
            type_of(v_layout),
            type_of(o_layout),
            type_of(v_layout),
            type_of(oc_layout),
        ]
        ctx.enqueue_function[kernel](
            TileTensor(ws.values, v_layout),
            TileTensor(ws.offsets, o_layout),
            TileTensor(ws.out_m, v_layout),
            TileTensor(ws.out_w, v_layout),
            TileTensor(ws.counts, oc_layout),
            compression,
            grid_dim=num_small,
            block_dim=BLOCK_CAP,
        )
        # Readback is compact: output occupies the same index range as input.
        ctx.enqueue_copy(
            ws.h_out_m.create_sub_buffer[_DT](0, small_total),
            ws.out_m.create_sub_buffer[_DT](0, small_total),
        )
        ctx.enqueue_copy(
            ws.h_out_w.create_sub_buffer[_DT](0, small_total),
            ws.out_w.create_sub_buffer[_DT](0, small_total),
        )
        ctx.enqueue_copy(
            ws.h_counts.create_sub_buffer[_FLAG](0, num_small),
            ws.counts.create_sub_buffer[_FLAG](0, num_small),
        )

    # Oversized keys: run while the grouped launch is in flight (the CPU-side
    # staging overlaps the GPU work; the GPU executes them in order).
    for j in range(len(big_keys)):
        var kw = List[Float32](length=len(big_vals[j]), fill=Float32(1))
        var part = gpu_compress_batch[S](
            ctx, ws.batch, big_vals[j], kw, compression
        )
        result.means[big_keys[j]] = part[0].copy()
        result.weights[big_keys[j]] = part[1].copy()

    if num_small > 0:
        ctx.synchronize()
        var pm = ws.h_out_m.unsafe_ptr()
        var pw = ws.h_out_w.unsafe_ptr()
        var pc = ws.h_counts.unsafe_ptr()
        for b in range(num_small):
            var k = small_keys[b]
            var base = Int(h_off[b])
            var cnt = Int(pc[b])
            var km = List[Float32](capacity=cnt)
            var kw = List[Float32](capacity=cnt)
            for j in range(cnt):
                km.append(pm[base + j])
                kw.append(pw[base + j])
            result.means[k] = km^
            result.weights[k] = kw^

    return result^
