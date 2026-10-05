"""Stage 5c: on-device scale functions vs. the CPU `ScaleFunction` trait.

The GPU backend evaluates `k(q)`, `q(k)`, and `normalizer(...)` by instantiating
the same `ScaleFunction` structs on-device (there is no second implementation).
These tests confirm that the device evaluation agrees with the host trait across
a dense grid of `q` values for all four scale functions, and that the inverse
round-trips (`q(k(q)) ≈ q`) on-device.

Both sides use the Float32 scale (`ScaleKnImpl[float32]`), so the comparison is
apples-to-apples: any residual gap comes only from device vs. host transcendental
implementations, which is tiny.

Gated on `has_accelerator()`: no GPU → every test is a no-op pass.
"""

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite

from kdigest.scale import (
    ScaleFunction,
    ScaleK0Impl,
    ScaleK1Impl,
    ScaleK2Impl,
    ScaleK3Impl,
)
from kdigest.gpu import gpu_scale_eval


comptime _DT = DType.float32
comptime COMPRESSION = Float32(200.0)
comptime COUNT = Float32(100_000.0)

# Device vs. host transcendental (asin/log/exp) implementations differ in the
# last few ulps; fp32 k-values here are O(10), so an absolute 1e-3 is ample
# while still catching any real formula divergence.
comptime _K_ATOL = Float32(1e-3)
# The inverse round-trip loses a bit more precision through two transcendentals.
comptime _Q_ATOL = Float32(2e-3)


def _grid() -> List[Float32]:
    """A dense q-grid including near-tail points (but inside the K2/K3 clamp).
    """
    var qs = List[Float32]()
    var pts = [
        0.0001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        0.999,
        0.9999,
    ]
    for i in range(len(pts)):
        qs.append(Float32(pts[i]))
    # Plus a uniform sweep to fill in between.
    for i in range(1, 64):
        qs.append(Float32(i) / 64.0)
    return qs^


def _abs(x: Float32) -> Float32:
    return x if x >= 0.0 else -x


def _check[S: ScaleFunction](cpu: S) raises where S.DT == _DT:
    var qs = _grid()
    var result = gpu_scale_eval[S](qs, COMPRESSION, COUNT)
    var gpu_k = result[0].copy()
    var gpu_q = result[1].copy()

    var c = rebind[Scalar[S.DT]](COMPRESSION)
    var cnt = rebind[Scalar[S.DT]](COUNT)
    var nrm = cpu.normalizer(c, cnt)
    for i in range(len(qs)):
        var want_k = Float32(cpu.k(rebind[Scalar[S.DT]](qs[i]), nrm))
        assert_true(
            _abs(gpu_k[i] - want_k) <= _K_ATOL,
            String(
                "k mismatch @q=", qs[i], ": gpu=", gpu_k[i], " cpu=", want_k
            ),
        )
        # On-device inverse must round-trip back to the input q.
        assert_true(
            _abs(gpu_q[i] - qs[i]) <= _Q_ATOL,
            String(
                "q(k(q)) mismatch @q=",
                qs[i],
                ": got=",
                gpu_q[i],
            ),
        )


def test_scale_device_k0() raises:
    comptime if not has_accelerator():
        return
    _check(ScaleK0Impl[_DT]())


def test_scale_device_k1() raises:
    comptime if not has_accelerator():
        return
    _check(ScaleK1Impl[_DT]())


def test_scale_device_k2() raises:
    comptime if not has_accelerator():
        return
    _check(ScaleK2Impl[_DT]())


def test_scale_device_k3() raises:
    comptime if not has_accelerator():
        return
    _check(ScaleK3Impl[_DT]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
