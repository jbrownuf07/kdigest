"""Grouped multi-digest GPU backend (Stage 5e) — the headline win.

One t-digest *per key* (metric / customer / sensor / window). Thousands of
independent small digests are compressed in a single kernel launch with **one
GPU block per key**, so the parallelism is *across* digests — this sidesteps the
single-digest sequential bottleneck entirely and mirrors how RAPIDS/libcudf
powers Spark's `approx_percentile`.

Pipeline (host):
  1. Counting-sort the (key, value) batch into per-key contiguous slices and
     build an offsets array (slice `b` is `[offsets[b], offsets[b+1])`).
  2. Launch `K` blocks; block `b` compresses its slice with the exact Stage 5d
     parallel k-bucketing merge in block-local shared memory, writing up to
     BLOCK_CAP centroids to its own output region.
  3. Keys whose slice exceeds BLOCK_CAP fall back to the Stage 5d single-digest
     `gpu_compress_batch` (hierarchical tiling), so correctness is independent
     of per-key volume.

Everything is Float32 (the GPU dtype); the scale function is reused on-device
from `kdigest.scale` (one source of truth).
"""

from std.math import floor

from max.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from kdigest.scale import ScaleFunction
from kdigest.gpu.primitives import BLOCK_CAP, cpu_sort_pairs
from kdigest.gpu.add_batch import gpu_compress_batch


comptime _DT = DType.float32
comptime _FLAG = DType.int32


def _grouped_compress_kernel[
    S: ScaleFunction,
    VL: TensorLayout,
    OL: TensorLayout,
    CL: TensorLayout,
    NL: TensorLayout,
](
    values: TileTensor[_DT, VL, MutAnyOrigin],  # flattened, key-sorted
    offsets: TileTensor[_FLAG, OL, MutAnyOrigin],  # length K+1
    out_means: TileTensor[_DT, CL, MutAnyOrigin],  # K * BLOCK_CAP
    out_weights: TileTensor[_DT, CL, MutAnyOrigin],  # K * BLOCK_CAP
    out_counts: TileTensor[_FLAG, NL, MutAnyOrigin],  # length K
    compression: Float32,
) where (S.DT == _DT):
    """One block per key. Block `b` compresses the key's unit-weight values
    `values[offsets[b] : offsets[b+1]]` into centroids via parallel
    k-bucketing. Assumes each key's slice length <= BLOCK_CAP (the host routes
    larger keys elsewhere)."""
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

    # Load this key's slice; values arrive pre-sorted per key (host sorts).
    if tid < n:
        m[tid] = values[start + tid]
        w[tid] = Scalar[_DT](1)
    else:
        m[tid] = Scalar[_DT](0)
        w[tid] = Scalar[_DT](0)
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

    var base = Int(b) * BLOCK_CAP
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


def gpu_grouped_compress[
    S: ScaleFunction
](
    keys: List[Int32],
    values: List[Float64],
    num_keys: Int,
    compression: Float32,
) raises -> GroupedResult where (S.DT == _DT):
    """Compress a (key, value) batch into one centroid set per key on the GPU.

    `keys[i]` is a dense key id in `[0, num_keys)` for `values[i]`. Keys with no
    samples yield empty centroid lists.
    """
    var result = GroupedResult()
    for _ in range(num_keys):
        result.means.append(List[Float32]())
        result.weights.append(List[Float32]())
        result.mins.append(Float32(0))
        result.maxs.append(Float32(0))

    var n = len(values)
    if n == 0:
        return result^

    # Counting sort by key → per-key contiguous, per-key value-sorted slices.
    var counts = List[Int](length=num_keys, fill=0)
    for i in range(n):
        counts[Int(keys[i])] += 1
    var offsets = List[Int32](length=num_keys + 1, fill=Int32(0))
    var running = 0
    for k in range(num_keys):
        offsets[k] = Int32(running)
        running += counts[k]
    offsets[num_keys] = Int32(running)

    # Scatter values into key-contiguous order; also track per-key min/max.
    var sorted_vals = List[Float32](length=n, fill=Float32(0))
    var cursor = List[Int](length=num_keys, fill=0)
    for i in range(n):
        var k = Int(keys[i])
        var dst = Int(offsets[k]) + cursor[k]
        sorted_vals[dst] = Float32(values[i])
        cursor[k] += 1
        var v = Float32(values[i])
        if cursor[k] == 1:
            result.mins[k] = v
            result.maxs[k] = v
        else:
            if v < result.mins[k]:
                result.mins[k] = v
            if v > result.maxs[k]:
                result.maxs[k] = v

    # Sort each key's slice by value (the kernel assumes a sorted slice).
    for k in range(num_keys):
        var lo = Int(offsets[k])
        var hi = Int(offsets[k + 1])
        var slice_m = List[Float32]()
        var slice_w = List[Float32]()
        for j in range(lo, hi):
            slice_m.append(sorted_vals[j])
            slice_w.append(Float32(1))
        cpu_sort_pairs(slice_m, slice_w)
        for j in range(lo, hi):
            sorted_vals[j] = slice_m[j - lo]

    # Split keys into those that fit one block and those that need the Stage 5d
    # hierarchical fallback.
    var small_keys = List[Int]()
    for k in range(num_keys):
        var cnt = counts[k]
        if cnt == 0:
            continue
        if cnt <= BLOCK_CAP:
            small_keys.append(k)
        else:
            # Large key: reuse the single-digest batch path on its slice.
            var lo = Int(offsets[k])
            var hi = Int(offsets[k + 1])
            var km = List[Float32]()
            var kw = List[Float32]()
            for j in range(lo, hi):
                km.append(sorted_vals[j])
                kw.append(Float32(1))
            var part = gpu_compress_batch[S](km, kw, compression)
            result.means[k] = part[0].copy()
            result.weights[k] = part[1].copy()

    var num_small = len(small_keys)
    if num_small == 0:
        return result^

    # Build a compact launch: one block per small key. Remap to dense block ids
    # so block b handles small_keys[b]. The kernel reads a per-block offset pair.
    var ctx = DeviceContext()
    var block_offsets = List[Int32](length=num_small + 1, fill=Int32(0))
    var flat = List[Float32]()
    var run = 0
    for b in range(num_small):
        block_offsets[b] = Int32(run)
        var k = small_keys[b]
        var lo = Int(offsets[k])
        var hi = Int(offsets[k + 1])
        for j in range(lo, hi):
            flat.append(sorted_vals[j])
        run += hi - lo
    block_offsets[num_small] = Int32(run)

    var total_vals = run
    var vbuf = ctx.enqueue_create_buffer[_DT](max(total_vals, 1))
    var obuf = ctx.enqueue_create_buffer[_FLAG](num_small + 1)
    var ombuf = ctx.enqueue_create_buffer[_DT](num_small * BLOCK_CAP)
    var owbuf = ctx.enqueue_create_buffer[_DT](num_small * BLOCK_CAP)
    var ocbuf = ctx.enqueue_create_buffer[_FLAG](num_small)

    with vbuf.map_to_host() as hv:
        var t = TileTensor(hv, row_major(max(total_vals, 1)))
        for i in range(total_vals):
            t[i] = flat[i]
    with obuf.map_to_host() as ho:
        var t = TileTensor(ho, row_major(num_small + 1))
        for i in range(num_small + 1):
            t[i] = block_offsets[i]

    var v_layout = row_major(max(total_vals, 1))
    var o_layout = row_major(num_small + 1)
    var c_layout = row_major(num_small * BLOCK_CAP)
    var oc_layout = row_major(num_small)
    var tv = TileTensor(vbuf, v_layout)
    var to = TileTensor(obuf, o_layout)
    var tom = TileTensor(ombuf, c_layout)
    var tow = TileTensor(owbuf, c_layout)
    var toc = TileTensor(ocbuf, oc_layout)

    comptime kernel = _grouped_compress_kernel[
        S,
        type_of(v_layout),
        type_of(o_layout),
        type_of(c_layout),
        type_of(oc_layout),
    ]
    ctx.enqueue_function[kernel](
        tv,
        to,
        tom,
        tow,
        toc,
        compression,
        grid_dim=num_small,
        block_dim=BLOCK_CAP,
    )

    var counts_out = List[Int]()
    with ocbuf.map_to_host() as hc:
        var t = TileTensor(hc, oc_layout)
        for b in range(num_small):
            counts_out.append(Int(rebind[Scalar[_FLAG]](t[b])))
    with ombuf.map_to_host() as hm, owbuf.map_to_host() as hw:
        var tm = TileTensor(hm, c_layout)
        var tw = TileTensor(hw, c_layout)
        for b in range(num_small):
            var k = small_keys[b]
            var base = b * BLOCK_CAP
            var cnt = counts_out[b]
            var km = List[Float32]()
            var kw = List[Float32]()
            for j in range(cnt):
                km.append(rebind[Scalar[_DT]](tm[base + j]))
                kw.append(rebind[Scalar[_DT]](tw[base + j]))
            result.means[k] = km^
            result.weights[k] = kw^

    return result^
