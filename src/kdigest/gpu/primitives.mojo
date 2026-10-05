"""Single-block GPU primitives for the t-digest merge pass (Stage 5b).

These are the reusable device building blocks the GPU t-digest backend composes
in later stages:

  * `gpu_sort_pairs`      — sort (key, value) pairs ascending by key.
  * `gpu_inclusive_scan`  — inclusive prefix sum.
  * `gpu_exclusive_scan`  — exclusive prefix sum.
  * `gpu_segmented_scan`  — inclusive prefix sum that resets at segment heads.
  * `gpu_compact`         — keep elements whose flag is 1, preserving order.

Granularity. Each primitive runs in a *single thread block* and therefore
handles up to `BLOCK_CAP` elements. That is exactly the shape the digest
pipeline needs: one block processes one digest's `buffer ∪ centroids`
(Stage 5d), and the grouped backend assigns one block per key (Stage 5e). Larger
single arrays are out of scope here and are handled by tiling in the backend.

Algorithms are chosen for obvious correctness over peak speed:
  * Bitonic sort over a power-of-two padded shared array (pad keys with +inf).
  * Hillis-Steele scans in shared memory.
These are correctness-first known-good parts; the backend stages tune on top.

Each GPU entry point has a matching `cpu_*` reference with identical semantics,
so tests can assert equality on tricky lengths without a second implementation
drifting. The CPU references are also the fallback when no accelerator exists.
"""

from std.bit import next_power_of_two
from std.sys import has_accelerator

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major, stack_allocation


# Maximum elements a single-block primitive can process. Power of two so the
# bitonic network and the Hillis-Steele scans tile cleanly.
comptime BLOCK_CAP = 1024

comptime _KEY = DType.float32
comptime _VAL = DType.float32
comptime _FLAG = DType.int32

comptime _cap_f32 = row_major[BLOCK_CAP]()
comptime _cap_i32 = row_major[BLOCK_CAP]()


# ======================================================================
# Device kernels
# ======================================================================


def _sort_pairs_kernel(
    keys: TileTensor[_KEY, type_of(_cap_f32), MutAnyOrigin],
    vals: TileTensor[_VAL, type_of(_cap_f32), MutAnyOrigin],
    n: Int32,
):
    """Bitonic sort of (keys, vals) ascending by key, in one block.

    The tail [n, BLOCK_CAP) is padded with +inf keys so padding sinks to the
    end and never displaces real data.
    """
    comptime assert keys.flat_rank == 1 and vals.flat_rank == 1
    var sk = stack_allocation[_KEY, address_space=AddressSpace.SHARED](_cap_f32)
    var sv = stack_allocation[_VAL, address_space=AddressSpace.SHARED](_cap_f32)
    var tid = thread_idx.x

    if Int32(tid) < n:
        sk[tid] = keys[tid]
        sv[tid] = vals[tid]
    else:
        sk[tid] = Scalar[_KEY].MAX_FINITE
        sv[tid] = Scalar[_VAL](0)
    barrier()

    var k = 2
    while k <= BLOCK_CAP:
        var j = k // 2
        while j > 0:
            var ixj = tid ^ j
            if ixj > tid:
                var ascending = (tid & k) == 0
                var a = rebind[Scalar[_KEY]](sk[tid])
                var b = rebind[Scalar[_KEY]](sk[ixj])
                var swap = (ascending and a > b) or (not ascending and a < b)
                if swap:
                    sk[tid] = b
                    sk[ixj] = a
                    var va = rebind[Scalar[_VAL]](sv[tid])
                    var vb = rebind[Scalar[_VAL]](sv[ixj])
                    sv[tid] = vb
                    sv[ixj] = va
            barrier()
            j //= 2
        k *= 2

    if Int32(tid) < n:
        keys[tid] = sk[tid]
        vals[tid] = sv[tid]


def _scan_kernel(
    data: TileTensor[_VAL, type_of(_cap_f32), MutAnyOrigin],
    n: Int32,
    exclusive: Int32,
):
    """Hillis-Steele prefix sum in one block. `exclusive != 0` shifts the
    inclusive result right by one (first element becomes 0)."""
    comptime assert data.flat_rank == 1
    var a = stack_allocation[_VAL, address_space=AddressSpace.SHARED](_cap_f32)
    var b = stack_allocation[_VAL, address_space=AddressSpace.SHARED](_cap_f32)
    var tid = thread_idx.x

    a[tid] = data[tid] if Int32(tid) < n else Scalar[_VAL](0)
    barrier()

    var offset = 1
    while offset < BLOCK_CAP:
        if tid >= offset:
            b[tid] = rebind[Scalar[_VAL]](a[tid]) + rebind[Scalar[_VAL]](
                a[tid - offset]
            )
        else:
            b[tid] = a[tid]
        barrier()
        a[tid] = b[tid]
        barrier()
        offset *= 2

    if Int32(tid) < n:
        if exclusive != 0:
            var incl = rebind[Scalar[_VAL]](a[tid])
            var self_val = data[tid]
            data[tid] = incl - self_val
        else:
            data[tid] = a[tid]


def _segscan_kernel(
    data: TileTensor[_VAL, type_of(_cap_f32), MutAnyOrigin],
    flags: TileTensor[_FLAG, type_of(_cap_i32), MutAnyOrigin],
    n: Int32,
):
    """Inclusive segmented prefix sum. `flags[i] != 0` marks the first element
    of a new segment; the running sum resets there. Flags are consumed (a copy
    is scanned in shared memory, so the caller's flags are preserved)."""
    comptime assert data.flat_rank == 1 and flags.flat_rank == 1
    var val = stack_allocation[_VAL, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var fl = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_i32
    )
    var vtmp = stack_allocation[_VAL, address_space=AddressSpace.SHARED](
        _cap_f32
    )
    var ftmp = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_i32
    )
    var tid = thread_idx.x

    if Int32(tid) < n:
        val[tid] = data[tid]
        fl[tid] = flags[tid]
    else:
        val[tid] = Scalar[_VAL](0)
        fl[tid] = Scalar[_FLAG](1)  # pad is its own segment: never merges back
    barrier()

    var offset = 1
    while offset < BLOCK_CAP:
        if tid >= offset:
            var cur_flag = rebind[Scalar[_FLAG]](fl[tid])
            # Accumulate only if no segment boundary lies in (tid-offset, tid].
            if cur_flag == 0:
                vtmp[tid] = rebind[Scalar[_VAL]](val[tid]) + rebind[
                    Scalar[_VAL]
                ](val[tid - offset])
            else:
                vtmp[tid] = val[tid]
            var prev_flag = rebind[Scalar[_FLAG]](fl[tid - offset])
            ftmp[tid] = Scalar[_FLAG](1) if (
                cur_flag != 0 or prev_flag != 0
            ) else Scalar[_FLAG](0)
        else:
            vtmp[tid] = val[tid]
            ftmp[tid] = fl[tid]
        barrier()
        val[tid] = vtmp[tid]
        fl[tid] = ftmp[tid]
        barrier()
        offset *= 2

    if Int32(tid) < n:
        data[tid] = val[tid]


def _compact_kernel(
    keys_in: TileTensor[_KEY, type_of(_cap_f32), MutAnyOrigin],
    vals_in: TileTensor[_VAL, type_of(_cap_f32), MutAnyOrigin],
    flags: TileTensor[_FLAG, type_of(_cap_i32), MutAnyOrigin],
    keys_out: TileTensor[_KEY, type_of(_cap_f32), MutAnyOrigin],
    vals_out: TileTensor[_VAL, type_of(_cap_f32), MutAnyOrigin],
    count_out: TileTensor[_FLAG, type_of(_cap_i32), MutAnyOrigin],
    n: Int32,
):
    """Stable stream compaction: copy (key, val) pairs whose flag != 0 to the
    front of the output, preserving order. `count_out[0]` receives the number
    kept. Uses an exclusive scan of the flags to compute scatter positions."""
    comptime assert keys_in.flat_rank == 1 and flags.flat_rank == 1
    comptime assert keys_out.flat_rank == 1 and count_out.flat_rank == 1
    var pos = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_i32
    )
    var tmp = stack_allocation[_FLAG, address_space=AddressSpace.SHARED](
        _cap_i32
    )
    var tid = thread_idx.x

    var keep = Scalar[_FLAG](0)
    if Int32(tid) < n:
        keep = rebind[Scalar[_FLAG]](flags[tid])
    pos[tid] = keep
    barrier()

    # Inclusive scan of the keep flags.
    var offset = 1
    while offset < BLOCK_CAP:
        if tid >= offset:
            tmp[tid] = rebind[Scalar[_FLAG]](pos[tid]) + rebind[Scalar[_FLAG]](
                pos[tid - offset]
            )
        else:
            tmp[tid] = pos[tid]
        barrier()
        pos[tid] = tmp[tid]
        barrier()
        offset *= 2

    # Exclusive position = inclusive - keep. Scatter the kept element.
    if Int32(tid) < n and keep != 0:
        var dst = Int(rebind[Scalar[_FLAG]](pos[tid])) - 1
        keys_out[dst] = keys_in[tid]
        vals_out[dst] = vals_in[tid]

    # Total kept = inclusive scan value at the last valid element.
    if Int32(tid) == n - 1:
        count_out[0] = pos[tid]
    if n == 0 and tid == 0:
        count_out[0] = Scalar[_FLAG](0)


# ======================================================================
# Host wrappers
# ======================================================================


def _check_cap(n: Int) raises:
    if n > BLOCK_CAP:
        raise Error(
            "kdigest.gpu primitive received n=",
            n,
            " > BLOCK_CAP=",
            BLOCK_CAP,
            "; tile larger inputs in the backend.",
        )


def gpu_sort_pairs(mut keys: List[Float32], mut vals: List[Float32]) raises:
    """Sort (keys, vals) ascending by key on the GPU, in place."""
    var n = len(keys)
    _check_cap(n)
    if n <= 1:
        return
    var ctx = DeviceContext()
    var kbuf = ctx.enqueue_create_buffer[_KEY](BLOCK_CAP)
    var vbuf = ctx.enqueue_create_buffer[_VAL](BLOCK_CAP)
    with kbuf.map_to_host() as hk, vbuf.map_to_host() as hv:
        var tk = TileTensor(hk, _cap_f32)
        var tv = TileTensor(hv, _cap_f32)
        for i in range(n):
            tk[i] = keys[i]
            tv[i] = vals[i]
    var tk = TileTensor(kbuf, _cap_f32)
    var tv = TileTensor(vbuf, _cap_f32)
    ctx.enqueue_function[_sort_pairs_kernel](
        tk, tv, Int32(n), grid_dim=1, block_dim=BLOCK_CAP
    )
    with kbuf.map_to_host() as hk, vbuf.map_to_host() as hv:
        var rk = TileTensor(hk, _cap_f32)
        var rv = TileTensor(hv, _cap_f32)
        for i in range(n):
            keys[i] = rebind[Scalar[_KEY]](rk[i])
            vals[i] = rebind[Scalar[_VAL]](rv[i])


def _run_scan(mut data: List[Float32], exclusive: Int) raises:
    var n = len(data)
    _check_cap(n)
    if n == 0:
        return
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[_VAL](BLOCK_CAP)
    with buf.map_to_host() as h:
        var t = TileTensor(h, _cap_f32)
        for i in range(n):
            t[i] = data[i]
    var t = TileTensor(buf, _cap_f32)
    ctx.enqueue_function[_scan_kernel](
        t, Int32(n), Int32(exclusive), grid_dim=1, block_dim=BLOCK_CAP
    )
    with buf.map_to_host() as h:
        var r = TileTensor(h, _cap_f32)
        for i in range(n):
            data[i] = rebind[Scalar[_VAL]](r[i])


def gpu_inclusive_scan(mut data: List[Float32]) raises:
    """Inclusive prefix sum on the GPU, in place."""
    _run_scan(data, 0)


def gpu_exclusive_scan(mut data: List[Float32]) raises:
    """Exclusive prefix sum on the GPU, in place."""
    _run_scan(data, 1)


def gpu_segmented_scan(mut data: List[Float32], flags: List[Int32]) raises:
    """Inclusive segmented prefix sum on the GPU, in place. `flags[i] != 0`
    marks the first element of a new segment."""
    var n = len(data)
    _check_cap(n)
    if n == 0:
        return
    var ctx = DeviceContext()
    var dbuf = ctx.enqueue_create_buffer[_VAL](BLOCK_CAP)
    var fbuf = ctx.enqueue_create_buffer[_FLAG](BLOCK_CAP)
    with dbuf.map_to_host() as hd, fbuf.map_to_host() as hf:
        var td = TileTensor(hd, _cap_f32)
        var tf = TileTensor(hf, _cap_i32)
        for i in range(n):
            td[i] = data[i]
            tf[i] = flags[i]
    var td = TileTensor(dbuf, _cap_f32)
    var tf = TileTensor(fbuf, _cap_i32)
    ctx.enqueue_function[_segscan_kernel](
        td, tf, Int32(n), grid_dim=1, block_dim=BLOCK_CAP
    )
    with dbuf.map_to_host() as hd:
        var r = TileTensor(hd, _cap_f32)
        for i in range(n):
            data[i] = rebind[Scalar[_VAL]](r[i])


def gpu_compact(
    keys: List[Float32], vals: List[Float32], flags: List[Int32]
) raises -> Tuple[List[Float32], List[Float32]]:
    """Keep (key, val) pairs whose flag != 0, preserving order. Returns the
    compacted keys and vals."""
    var n = len(keys)
    _check_cap(n)
    var out_keys = List[Float32]()
    var out_vals = List[Float32]()
    if n == 0:
        return (out_keys^, out_vals^)
    var ctx = DeviceContext()
    var kin = ctx.enqueue_create_buffer[_KEY](BLOCK_CAP)
    var vin = ctx.enqueue_create_buffer[_VAL](BLOCK_CAP)
    var fin = ctx.enqueue_create_buffer[_FLAG](BLOCK_CAP)
    var kout = ctx.enqueue_create_buffer[_KEY](BLOCK_CAP)
    var vout = ctx.enqueue_create_buffer[_VAL](BLOCK_CAP)
    var cout = ctx.enqueue_create_buffer[_FLAG](BLOCK_CAP)
    with kin.map_to_host() as hk, vin.map_to_host() as hv, fin.map_to_host() as hf:
        var tk = TileTensor(hk, _cap_f32)
        var tv = TileTensor(hv, _cap_f32)
        var tf = TileTensor(hf, _cap_i32)
        for i in range(n):
            tk[i] = keys[i]
            tv[i] = vals[i]
            tf[i] = flags[i]
    var tkin = TileTensor(kin, _cap_f32)
    var tvin = TileTensor(vin, _cap_f32)
    var tfin = TileTensor(fin, _cap_i32)
    var tkout = TileTensor(kout, _cap_f32)
    var tvout = TileTensor(vout, _cap_f32)
    var tcout = TileTensor(cout, _cap_i32)
    ctx.enqueue_function[_compact_kernel](
        tkin,
        tvin,
        tfin,
        tkout,
        tvout,
        tcout,
        Int32(n),
        grid_dim=1,
        block_dim=BLOCK_CAP,
    )
    var kept: Int
    with cout.map_to_host() as hc:
        var tc = TileTensor(hc, _cap_i32)
        kept = Int(rebind[Scalar[_FLAG]](tc[0]))
    with kout.map_to_host() as hk, vout.map_to_host() as hv:
        var rk = TileTensor(hk, _cap_f32)
        var rv = TileTensor(hv, _cap_f32)
        for i in range(kept):
            out_keys.append(rebind[Scalar[_KEY]](rk[i]))
            out_vals.append(rebind[Scalar[_VAL]](rv[i]))
    return (out_keys^, out_vals^)


# ======================================================================
# CPU references (semantics shared with the GPU paths; also the fallback)
# ======================================================================


comptime _INSERTION_RUN = 32


def cpu_sort_pairs(mut keys: List[Float32], mut vals: List[Float32]):
    """Stable sort of (keys, vals) ascending by key, in place. O(n log n).

    Bottom-up merge sort: insertion-sort fixed runs of `_INSERTION_RUN`
    elements (fast on tiny ranges), then merge runs of doubling width through a
    scratch buffer, ping-ponging between the two. Stable (ties keep input
    order), which the GPU tests rely on when comparing against the bitonic sort.
    This is on the host hot path of `gpu_compress_batch` / grouped compress, so
    it must not be quadratic.
    """
    var n = len(keys)
    if n < 2:
        return

    # 1. Insertion-sort each small run.
    var start = 0
    while start < n:
        var end = min(start + _INSERTION_RUN, n)
        for i in range(start + 1, end):
            var k = keys[i]
            var v = vals[i]
            var j = i - 1
            while j >= start and keys[j] > k:
                keys[j + 1] = keys[j]
                vals[j + 1] = vals[j]
                j -= 1
            keys[j + 1] = k
            vals[j + 1] = v
        start = end
    if n <= _INSERTION_RUN:
        return

    # 2. Merge runs of doubling width, ping-ponging src <-> dst.
    var src_k = keys.copy()
    var src_v = vals.copy()
    var dst_k = List[Float32](length=n, fill=Float32(0))
    var dst_v = List[Float32](length=n, fill=Float32(0))
    var width = _INSERTION_RUN
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var a = lo
            var b = mid
            var o = lo
            while a < mid and b < hi:
                # `<=` takes from the left run on ties -> stable.
                if src_k[a] <= src_k[b]:
                    dst_k[o] = src_k[a]
                    dst_v[o] = src_v[a]
                    a += 1
                else:
                    dst_k[o] = src_k[b]
                    dst_v[o] = src_v[b]
                    b += 1
                o += 1
            while a < mid:
                dst_k[o] = src_k[a]
                dst_v[o] = src_v[a]
                a += 1
                o += 1
            while b < hi:
                dst_k[o] = src_k[b]
                dst_v[o] = src_v[b]
                b += 1
                o += 1
            lo = hi
        var tk = src_k^
        src_k = dst_k^
        dst_k = tk^
        var tv = src_v^
        src_v = dst_v^
        dst_v = tv^
        width *= 2

    keys = src_k^
    vals = src_v^


def cpu_inclusive_scan(mut data: List[Float32]):
    var acc = Float32(0)
    for i in range(len(data)):
        acc += data[i]
        data[i] = acc


def cpu_exclusive_scan(mut data: List[Float32]):
    var acc = Float32(0)
    for i in range(len(data)):
        var cur = data[i]
        data[i] = acc
        acc += cur


def cpu_segmented_scan(mut data: List[Float32], flags: List[Int32]):
    var acc = Float32(0)
    for i in range(len(data)):
        if flags[i] != 0:
            acc = data[i]
        else:
            acc += data[i]
        data[i] = acc


def cpu_compact(
    keys: List[Float32], vals: List[Float32], flags: List[Int32]
) -> Tuple[List[Float32], List[Float32]]:
    var out_keys = List[Float32]()
    var out_vals = List[Float32]()
    for i in range(len(keys)):
        if flags[i] != 0:
            out_keys.append(keys[i])
            out_vals.append(vals[i])
    return (out_keys^, out_vals^)
