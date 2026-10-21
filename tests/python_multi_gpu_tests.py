"""
Fields on several GPUs of one process (audit finding L4), and GPU reductions
across ranks without GPU-aware MPI (M6).

Kernels, runtime allocations and the reduction scratch act on the current
device. Field allocation selects the field's device and leaves it selected, so
with fields on two GPUs, operating on the one allocated first ran its kernels
on the other: a memory access fault on MI300A.
"""

import numpy as np
import pytest

import muGrid
from muGrid import linalg

cp = pytest.importorskip("cupy")

try:
    NB_DEVICES = cp.cuda.runtime.getDeviceCount() if muGrid.has_gpu else 0
except cp.cuda.runtime.CUDARuntimeError:
    NB_DEVICES = 0

two_gpus = pytest.mark.skipif(NB_DEVICES < 2, reason="needs two GPUs")


def _fields_on(device, n=32, value=1.0):
    fc = muGrid.GlobalFieldCollection(
        (n, n), device=muGrid.Device.gpu(device),
        nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1))
    a = fc.real_field("a")
    b = fc.real_field("b")
    with cp.cuda.Device(device):
        a.p[...] = value
        b.p[...] = 2.0
    return fc, a, b


@two_gpus
def test_linalg_runs_on_the_field_device_and_restores_the_current_one():
    fields = {d: _fields_on(d, value=1.0 + d) for d in (0, 1)}
    n2 = 32 * 32
    for current in (0, 1):
        cp.cuda.runtime.setDevice(current)
        for device, (_, a, b) in fields.items():
            assert float(linalg.vecdot(a, b)) == n2 * (1.0 + device) * 2.0
            assert float(linalg.norm_sq(b)) == n2 * 4.0
            assert cp.cuda.runtime.getDevice() == current
    _, a, b = fields[1]
    cp.cuda.runtime.setDevice(0)
    linalg.axpy(1.0, a, b)
    with cp.cuda.Device(1):
        assert float(b.p.mean()) == 4.0
    assert cp.cuda.runtime.getDevice() == 0


@two_gpus
def test_operators_and_fft_run_on_the_field_device():
    cp.cuda.runtime.setDevice(0)
    decomposition = muGrid.CartesianDecomposition(
        muGrid.Communicator(), [32, 32], nb_subdivisions=[1, 1],
        nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1),
        device=muGrid.Device.gpu(1))
    v = decomposition.real_field("v")
    w = decomposition.real_field("w")
    with cp.cuda.Device(1):
        v.p[...] = cp.sin(2 * cp.pi * cp.arange(32) / 32)[:, None]
    cp.cuda.runtime.setDevice(0)
    decomposition.communicate_ghosts(v)
    muGrid.LaplaceOperator(2).apply(v, w)
    with cp.cuda.Device(1):
        # periodic 5-point stencil: -4 sin^2(pi/32) v on this mode
        expect = -4 * np.sin(np.pi / 32) ** 2 * cp.asnumpy(v.p)
        np.testing.assert_allclose(cp.asnumpy(w.p), expect, atol=1e-12)
    assert cp.cuda.runtime.getDevice() == 0

    engine = muGrid.FFTEngine((16, 12), muGrid.Communicator(),
                              device=muGrid.Device.gpu(1))
    r = engine.real_space_field("r")
    f = engine.fourier_space_field("f")
    values = np.random.default_rng(0).standard_normal(r.p.shape)
    with cp.cuda.Device(1):
        r.p[...] = cp.asarray(values)
    cp.cuda.runtime.setDevice(0)
    engine.fft(r, f)
    engine.ifft(f, r)
    with cp.cuda.Device(1):
        np.testing.assert_allclose(cp.asnumpy(r.p) * engine.normalisation,
                                   values, atol=1e-12)
    assert cp.cuda.runtime.getDevice() == 0


def test_gpu_reductions_across_ranks(comm):
    """Run under mpirun with MUGRID_GPU_AWARE_MPI=0: the reductions must
    stage through the host instead of handing device pointers to an MPI that
    cannot read them."""
    if not muGrid.has_gpu:
        pytest.skip("needs a GPU")
    x = cp.arange(4, dtype=np.float64) + comm.rank
    total = comm.sum(x)
    expect = np.arange(4) * comm.size + sum(range(comm.size))
    np.testing.assert_array_equal(cp.asnumpy(total), expect)
    assert float(comm.reduce_max(x)) == 3 + comm.size - 1
    assert float(comm.reduce_sum(x)) == float(expect.sum())


def test_gpu_exchanges_across_ranks(comm):
    """Run under mpirun with MUGRID_GPU_AWARE_MPI=0, as the reductions above:
    allgather and sendrecv must stage device arrays through the host."""
    if not muGrid.has_gpu:
        pytest.skip("needs a GPU")
    x = cp.arange(4, dtype=np.float64) + comm.rank
    gathered = comm.allgather(x)
    assert isinstance(gathered, cp.ndarray)
    expect = np.arange(4)[None] + np.arange(comm.size)[:, None]
    np.testing.assert_array_equal(cp.asnumpy(gathered), expect)
    up, down = (comm.rank + 1) % comm.size, (comm.rank - 1) % comm.size
    received = comm.sendrecv(x, dest=up, source=down)
    assert isinstance(received, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(received), np.arange(4) + down)
