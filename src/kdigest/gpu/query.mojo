"""Vectorized query + merge-tree reduction on the GPU (Stage 5f).

Two map/reduce companions to the Stage 5d/5e compress kernels:

  * `gpu_quantile` / `gpu_cdf` — a **parallel map**: one GPU thread per query.
    Given one compressed digest (centroid means + weights, plus total weight and
    the observed min/max), evaluate an arbitrary array of `q` (or `x`) values at
    once. This is the finalization step when you have one digest and many
    quantiles to read (e.g. a full p50/p90/p99/p999 panel per refresh, or an SLO
    sweep). The per-thread interpolation mirrors `TDigest.quantile` / `.cdf`
    exactly, so results agree with the CPU to fp32 rounding.

  * `gpu_merge_digests` — a **parallel reduction** over many partial digests.
    t-digests are *mergeable*: a digest of the union is well approximated by
    compressing the concatenation of the parts' centroids. The merge-tree is
    therefore "gather all centroids, compress once" via the Stage 5d
    `gpu_compress_batch` — the associative reduce that map-reduce finalization
    needs (each mapper emits a partial digest; the reducer folds them).

Everything is Float32 (the GPU dtype); digests here are decay-free (as produced
by `add_batch` / `GroupedTDigest`), so a centroid is a singleton iff its weight
is <= 1.
"""

from std.math import ceildiv

from max.gpu import global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from kdigest.scale import ScaleFunction
from kdigest.gpu.primitives import BLOCK_CAP
from kdigest.gpu.add_batch import gpu_compress_batch


comptime _DT = DType.float32
comptime _QBLOCK = 256


# --- Vectorized quantile -----------------------------------------------------


def _quantile_kernel[
    CL: TensorLayout, QL: TensorLayout
](
    means: TileTensor[_DT, CL, MutAnyOrigin],
    weights: TileTensor[_DT, CL, MutAnyOrigin],
    qs: TileTensor[_DT, QL, MutAnyOrigin],
    res: TileTensor[_DT, QL, MutAnyOrigin],
    ncent: Int32,
    total: Float32,
    min_v: Float32,
    max_v: Float32,
    nq: Int32,
):
    """One thread per query `q`. Interpolates the digest value at `q`, mirroring
    `TDigest.quantile`. Centroids (<= BLOCK_CAP) are cooperatively staged into
    shared memory so every thread in the block reuses them."""
    comptime assert means.flat_rank == 1 and weights.flat_rank == 1
    comptime assert qs.flat_rank == 1 and res.flat_rank == 1

    var sm = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var sw = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var nc = Int(ncent)
    var tid = thread_idx.x
    var i = Int(tid)
    while i < nc:
        sm[i] = means[i]
        sw[i] = weights[i]
        i += _QBLOCK
    barrier()

    var gid = global_idx.x
    if Int32(gid) >= nq:
        return

    var q = rebind[Scalar[_DT]](qs[gid])
    var n = total

    if nc == 1:
        res[gid] = rebind[Scalar[_DT]](sm[0])
        return
    if q <= 0.0:
        res[gid] = min_v
        return
    if q >= 1.0:
        res[gid] = max_v
        return

    var target = q * n
    var m0 = rebind[Scalar[_DT]](sm[0])
    var w0 = rebind[Scalar[_DT]](sw[0])

    # Below the first centroid's center of mass: interpolate from min.
    var first_half = w0 / 2.0
    if target < first_half:
        if w0 <= 1.0 + 1e-6:
            res[gid] = min_v
        else:
            res[gid] = min_v + (m0 - min_v) * (target / first_half)
        return

    var mL = rebind[Scalar[_DT]](sm[nc - 1])
    var wL = rebind[Scalar[_DT]](sw[nc - 1])
    var last_half = wL / 2.0
    if target > n - last_half:
        if wL <= 1.0 + 1e-6:
            res[gid] = max_v
        else:
            var frac = (target - (n - last_half)) / last_half
            res[gid] = mL + (max_v - mL) * frac
        return

    # Interior: walk cumulative weight between adjacent centers of mass.
    var weight_so_far = w0 / 2.0
    var j = 0
    while j < nc - 1:
        var lm = rebind[Scalar[_DT]](sm[j])
        var lw = rebind[Scalar[_DT]](sw[j])
        var rm = rebind[Scalar[_DT]](sm[j + 1])
        var rw = rebind[Scalar[_DT]](sw[j + 1])
        var dw = (lw + rw) / 2.0
        if target <= weight_so_far + dw:
            var frac = (target - weight_so_far) / dw
            res[gid] = lm + (rm - lm) * frac
            return
        weight_so_far += dw
        j += 1
    res[gid] = mL


def _cdf_kernel[
    CL: TensorLayout, QL: TensorLayout
](
    means: TileTensor[_DT, CL, MutAnyOrigin],
    weights: TileTensor[_DT, CL, MutAnyOrigin],
    xs: TileTensor[_DT, QL, MutAnyOrigin],
    res: TileTensor[_DT, QL, MutAnyOrigin],
    ncent: Int32,
    total: Float32,
    min_v: Float32,
    max_v: Float32,
    nq: Int32,
):
    """One thread per query `x`. Estimates P(X <= x), mirroring `TDigest.cdf`.
    """
    comptime assert means.flat_rank == 1 and xs.flat_rank == 1
    comptime assert res.flat_rank == 1

    var sm = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var sw = stack_allocation[_DT, address_space=AddressSpace.SHARED](
        row_major[BLOCK_CAP]()
    )
    var nc = Int(ncent)
    var tid = thread_idx.x
    var i = Int(tid)
    while i < nc:
        sm[i] = means[i]
        sw[i] = weights[i]
        i += _QBLOCK
    barrier()

    var gid = global_idx.x
    if Int32(gid) >= nq:
        return

    var x = rebind[Scalar[_DT]](xs[gid])
    var n = total

    if x < min_v:
        res[gid] = Scalar[_DT](0)
        return
    if x > max_v:
        res[gid] = Scalar[_DT](1)
        return

    var m0 = rebind[Scalar[_DT]](sm[0])
    var w0 = rebind[Scalar[_DT]](sw[0])
    if nc == 1:
        if x < m0:
            res[gid] = Scalar[_DT](0)
        elif x > m0:
            res[gid] = Scalar[_DT](1)
        else:
            res[gid] = Scalar[_DT](0.5)
        return

    if x < m0:
        if min_v == m0:
            res[gid] = Scalar[_DT](0)
        else:
            var frac = (x - min_v) / (m0 - min_v)
            res[gid] = (w0 / 2.0) * frac / n
        return

    var mL = rebind[Scalar[_DT]](sm[nc - 1])
    var wL = rebind[Scalar[_DT]](sw[nc - 1])
    if x > mL:
        if max_v == mL:
            res[gid] = Scalar[_DT](1)
        else:
            var frac = (x - mL) / (max_v - mL)
            res[gid] = (n - wL / 2.0 + (wL / 2.0) * frac) / n
        return

    var weight_so_far = w0 / 2.0
    var j = 0
    while j < nc - 1:
        var lm = rebind[Scalar[_DT]](sm[j])
        var lw = rebind[Scalar[_DT]](sw[j])
        var rm = rebind[Scalar[_DT]](sm[j + 1])
        var rw = rebind[Scalar[_DT]](sw[j + 1])
        if x <= rm:
            var dw = (lw + rw) / 2.0
            if rm == lm:
                res[gid] = (weight_so_far + dw / 2.0) / n
            else:
                var frac = (x - lm) / (rm - lm)
                res[gid] = (weight_so_far + dw * frac) / n
            return
        weight_so_far += (lw + rw) / 2.0
        j += 1
    res[gid] = Scalar[_DT](1)


def _launch_query[
    kernel_is_cdf: Bool
](
    means: List[Float32],
    weights: List[Float32],
    queries: List[Float32],
    min_v: Float32,
    max_v: Float32,
) raises -> List[Float32]:
    var nc = len(means)
    var nq = len(queries)
    var out = List[Float32]()
    if nq == 0:
        return out^
    if nc == 0:
        raise Error("cannot query an empty digest")
    if nc > BLOCK_CAP:
        raise Error("digest exceeds one block; merge/compress it first")

    var total = Float32(0)
    for i in range(nc):
        total += weights[i]
    if total <= 0.0:
        total = Float32(1)

    var ctx = DeviceContext()
    var c_layout = row_major(nc)
    var q_layout = row_major(nq)
    var mbuf = ctx.enqueue_create_buffer[_DT](nc)
    var wbuf = ctx.enqueue_create_buffer[_DT](nc)
    var qbuf = ctx.enqueue_create_buffer[_DT](nq)
    var obuf = ctx.enqueue_create_buffer[_DT](nq)
    with mbuf.map_to_host() as hm, wbuf.map_to_host() as hw:
        var tm = TileTensor(hm, c_layout)
        var tw = TileTensor(hw, c_layout)
        for i in range(nc):
            tm[i] = means[i]
            tw[i] = weights[i]
    with qbuf.map_to_host() as hq:
        var tq = TileTensor(hq, q_layout)
        for i in range(nq):
            tq[i] = queries[i]

    var tm = TileTensor(mbuf, c_layout)
    var tw = TileTensor(wbuf, c_layout)
    var tq = TileTensor(qbuf, q_layout)
    var to = TileTensor(obuf, q_layout)

    comptime if kernel_is_cdf:
        comptime kernel = _cdf_kernel[type_of(c_layout), type_of(q_layout)]
        ctx.enqueue_function[kernel](
            tm,
            tw,
            tq,
            to,
            Int32(nc),
            total,
            min_v,
            max_v,
            Int32(nq),
            grid_dim=ceildiv(nq, _QBLOCK),
            block_dim=_QBLOCK,
        )
    else:
        comptime kernel = _quantile_kernel[type_of(c_layout), type_of(q_layout)]
        ctx.enqueue_function[kernel](
            tm,
            tw,
            tq,
            to,
            Int32(nc),
            total,
            min_v,
            max_v,
            Int32(nq),
            grid_dim=ceildiv(nq, _QBLOCK),
            block_dim=_QBLOCK,
        )

    with obuf.map_to_host() as ho:
        var t = TileTensor(ho, q_layout)
        for i in range(nq):
            out.append(rebind[Scalar[_DT]](t[i]))
    return out^


def gpu_quantile(
    means: List[Float32],
    weights: List[Float32],
    qs: List[Float32],
    min_v: Float32,
    max_v: Float32,
) raises -> List[Float32]:
    """Vectorized `quantile`: evaluate every `q` in `qs` against one compressed
    digest, one GPU thread per query. The digest must fit a block (<= BLOCK_CAP
    centroids), which is always true after a compress."""
    return _launch_query[False](means, weights, qs, min_v, max_v)


def gpu_cdf(
    means: List[Float32],
    weights: List[Float32],
    xs: List[Float32],
    min_v: Float32,
    max_v: Float32,
) raises -> List[Float32]:
    """Vectorized `cdf`: evaluate P(X <= x) for every `x` in `xs` against one
    compressed digest, one GPU thread per query."""
    return _launch_query[True](means, weights, xs, min_v, max_v)


# --- Merge-tree reduction ----------------------------------------------------


def gpu_merge_digests[
    S: ScaleFunction
](
    part_means: List[List[Float32]],
    part_weights: List[List[Float32]],
    compression: Float32,
) raises -> Tuple[List[Float32], List[Float32]] where (S.DT == _DT):
    """Reduce many partial digests into one via the mergeable property.

    Gathers all parts' centroids and compresses the union once with the Stage 5d
    `gpu_compress_batch` — the associative reduce behind map-reduce finalization
    (each mapper emits a partial digest; this folds them into the global digest).
    Weights are preserved, so the merge is weighted correctly regardless of how
    the stream was partitioned across mappers.
    """
    var all_m = List[Float32]()
    var all_w = List[Float32]()
    for p in range(len(part_means)):
        for i in range(len(part_means[p])):
            all_m.append(part_means[p][i])
            all_w.append(part_weights[p][i])
    return gpu_compress_batch[S](all_m, all_w, compression)
