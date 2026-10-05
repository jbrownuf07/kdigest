"""Scale functions evaluated on-device (Stage 5c).

The t-digest merge pass needs `k(q)`, its inverse `q(k)`, and `normalizer(...)`
evaluated per-thread on the GPU. Rather than re-derive the formulas in device
code, these kernels instantiate the *same* `ScaleFunction` structs from
`kdigest.scale` (at `DType.float32`) and call their methods directly — the trait
math compiles to device code unchanged. That keeps a single source of truth: the
Phase 1 formulas, called from both the CPU digest and the GPU backend.

Only the Float32 instantiations are meaningful on-device (the GPU backend is
fp32; Apple silicon has no native fp64). Host wrappers evaluate a batch of `q`
values and return `k(q)` and the round-tripped `q(k(q))`, plus the device
`normalizer`, so tests can check both the forward map and the inverse against the
CPU trait.
"""

from std.math import ceildiv

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from kdigest.scale import ScaleFunction


comptime _DT = DType.float32


def _scale_eval_kernel[
    S: ScaleFunction, Layout: TensorLayout
](
    qs: TileTensor[_DT, Layout, MutAnyOrigin],
    out_k: TileTensor[_DT, Layout, MutAnyOrigin],
    out_q: TileTensor[_DT, Layout, MutAnyOrigin],
    compression: Float32,
    count: Float32,
    n: Int32,
) where (S.DT == _DT):
    comptime assert qs.flat_rank == 1
    var tid = global_idx.x
    if Int32(tid) < n:
        var s = S()
        var c = rebind[Scalar[S.DT]](compression)
        var cnt = rebind[Scalar[S.DT]](count)
        var nrm = s.normalizer(c, cnt)
        var q = rebind[Scalar[S.DT]](qs[tid])
        var kq = s.k(q, nrm)
        out_k[tid] = rebind[Scalar[_DT]](kq)
        # Round-trip the inverse: q(k(q)) must recover q.
        out_q[tid] = rebind[Scalar[_DT]](s.q(kq, nrm))


def gpu_scale_eval[
    S: ScaleFunction
](qs: List[Float32], compression: Float32, count: Float32) raises -> Tuple[
    List[Float32], List[Float32]
] where (S.DT == _DT):
    """Evaluate `k(q)` and the round-trip `q(k(q))` on the GPU for every `q` in
    `qs`, using the on-device `ScaleFunction` `S`. Returns `(k_values,
    q_roundtrip)`."""
    var n = len(qs)
    var out_k = List[Float32]()
    var out_q = List[Float32]()
    if n == 0:
        return (out_k^, out_q^)

    var ctx = DeviceContext()
    var layout = row_major(n)
    var qbuf = ctx.enqueue_create_buffer[_DT](n)
    var kbuf = ctx.enqueue_create_buffer[_DT](n)
    var rbuf = ctx.enqueue_create_buffer[_DT](n)
    with qbuf.map_to_host() as h:
        var t = TileTensor(h, layout)
        for i in range(n):
            t[i] = qs[i]
    var tq = TileTensor(qbuf, layout)
    var tk = TileTensor(kbuf, layout)
    var tr = TileTensor(rbuf, layout)

    comptime kernel = _scale_eval_kernel[S, type_of(layout)]
    ctx.enqueue_function[kernel](
        tq,
        tk,
        tr,
        compression,
        count,
        Int32(n),
        grid_dim=ceildiv(n, 256),
        block_dim=256,
    )

    with kbuf.map_to_host() as hk, rbuf.map_to_host() as hr:
        var rk = TileTensor(hk, layout)
        var rr = TileTensor(hr, layout)
        for i in range(n):
            out_k.append(rebind[Scalar[_DT]](rk[i]))
            out_q.append(rebind[Scalar[_DT]](rr[i]))
    return (out_k^, out_q^)
