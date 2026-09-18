#!/usr/bin/env python3
"""Cost of the hybrid preconditioner's parallel tridiagonal solve.

Companion to `mg_prototype.py --hybrid`, which established that the
Fourier/tridiagonal hybrid reproduces the FFT preconditioner's CG count exactly
(see section 9 of docs/multigrid_preconditioner_plan.md). That settled
convergence; this settles cost. Three measurements, none of which the
convergence prototype can give:

``--fft``
    The 2D transform pair the hybrid issues against the 3D pair the reference
    preconditioner issues. Their ratio is what dropping the distributed axis
    saves, independent of any implementation detail of the solve.

``--sweep``
    A fused block-Thomas sweep as a HIP kernel: one thread per (qx, qy) mode,
    marching the whole z line with the 3x3 blocks in registers. This is the
    arithmetic the hybrid adds, and it has to be a real kernel rather than a
    cupy loop -- a Python loop over nz issues thousands of tiny launches and
    measures launch latency instead of the sweep.

    Note the (z, mode, component) layout. It is not incidental: it is what lets
    a wavefront read contiguous bytes at each z step. The natural
    (mode, z, component) layout strides by nz*3 between neighbouring threads and
    gives up most of the bandwidth.

``--comm``
    Run under mpirun. The all-to-all the reference pays per apply against the
    interface exchange a partitioned-Thomas solve pays, the latter by recursive
    doubling rather than a gather to one rank, which is an O(P) bottleneck and
    not what anyone would implement.

    These are shared-memory numbers on one node, and section 7 showed that such
    numbers understate inter-device penalties -- so treat the advantage shown
    here as a lower bound.
"""

import argparse
import time

import numpy as np

BLOCK_THOMAS = r'''
typedef double2 c64;
__device__ __forceinline__ c64 cmul(c64 a, c64 b){
    return make_double2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ c64 csub(c64 a, c64 b){
    return make_double2(a.x-b.x, a.y-b.y);
}
extern "C" __global__ void block_thomas(
    const c64* __restrict__ r, const c64* __restrict__ Mfwd,
    const c64* __restrict__ Dinv, const c64* __restrict__ Pbak,
    c64* __restrict__ y, c64* __restrict__ z, int nz, int nmodes)
{
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= nmodes) return;
    c64 M[9], D[9], P[9], prev[3];
    for (int i=0;i<9;i++){ M[i]=Mfwd[m*9+i]; D[i]=Dinv[m*9+i]; P[i]=Pbak[m*9+i]; }
    for (int i=0;i<3;i++) prev[i]=make_double2(0.0,0.0);
    for (int k=0;k<nz;k++){                       /* forward: y = r - M y_{k-1} */
        size_t o = (size_t)k*nmodes*3 + (size_t)m*3;
        c64 acc[3];
        for (int i=0;i<3;i++){
            c64 s = r[o+i];
            for (int j=0;j<3;j++) s = csub(s, cmul(M[i*3+j], prev[j]));
            acc[i]=s;
        }
        for (int i=0;i<3;i++){ y[o+i]=acc[i]; prev[i]=acc[i]; }
    }
    for (int i=0;i<3;i++) prev[i]=make_double2(0.0,0.0);
    for (int k=nz-1;k>=0;k--){                    /* backward: z = D y - P z_{k+1} */
        size_t o = (size_t)k*nmodes*3 + (size_t)m*3;
        c64 acc[3];
        for (int i=0;i<3;i++){
            c64 s = make_double2(0.0,0.0);
            for (int j=0;j<3;j++){
                s.x += cmul(D[i*3+j], y[o+j]).x - cmul(P[i*3+j], prev[j]).x;
                s.y += cmul(D[i*3+j], y[o+j]).y - cmul(P[i*3+j], prev[j]).y;
            }
            acc[i]=s;
        }
        for (int i=0;i<3;i++){ z[o+i]=acc[i]; prev[i]=acc[i]; }
    }
}
'''


def _bench(fn, sync, repeats=10):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - start) / repeats


def cmd_fft(sizes):
    import cupy as cp
    sync = cp.cuda.runtime.deviceSynchronize
    rng = np.random.default_rng(0)
    print("Transform pair: what dropping the distributed axis saves\n")
    print(f"{'grid':>8} {'3D pair':>10} {'2D pair':>10} {'2D/3D':>8} {'saved':>10}")
    for n in sizes:
        u = cp.asarray(rng.standard_normal((3, n, n, n)))
        t3 = _bench(lambda: cp.fft.irfftn(cp.fft.rfftn(u, axes=(1, 2, 3)),
                                          axes=(1, 2, 3), s=(n, n, n)), sync)
        t2 = _bench(lambda: cp.fft.irfftn(cp.fft.rfftn(u, axes=(1, 2)),
                                          axes=(1, 2), s=(n, n)), sync)
        print(f"{n:>8} {t3*1e6:>9.0f}us {t2*1e6:>9.0f}us {t2/t3:>8.2f} "
              f"{(t3-t2)*1e6:>9.0f}us")
        del u
        cp.get_default_memory_pool().free_all_blocks()


def cmd_sweep(sizes):
    import cupy as cp
    sync = cp.cuda.runtime.deviceSynchronize
    kernel = cp.RawKernel(BLOCK_THOMAS, "block_thomas", backend="hiprtc")
    rng = cp.random.default_rng(0)
    print("Fused block-Thomas sweep along the distributed axis\n")
    print(f"{'grid':>8} {'modes':>9} {'nz':>6} {'sweep':>10} {'GB moved':>10} "
          f"{'GB/s':>8}")
    for n in sizes:
        nmodes, nz = n * (n // 2 + 1), n
        def rand(shape):
            return (rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(cp.complex128)
        r = rand((nz, nmodes, 3))
        y, z = cp.empty_like(r), cp.empty_like(r)
        blocks = [rand((nmodes, 3, 3)) * 0.1 for _ in range(3)]
        threads = 256
        grid = ((nmodes + threads - 1) // threads,)
        args = (r, blocks[0], blocks[1], blocks[2], y, z,
                np.int32(nz), np.int32(nmodes))
        dt = _bench(lambda: kernel(grid, (threads,), args), sync)
        moved = 4 * r.nbytes / 1e9        # read r, write y, read y, write z
        print(f"{n:>8} {nmodes:>9} {nz:>6} {dt*1e6:>9.0f}us {moved:>10.2f} "
              f"{moved/dt:>8.0f}")
        del r, y, z, blocks
        cp.get_default_memory_pool().free_all_blocks()


def cmd_comm(n):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nb_ranks, rank = comm.size, comm.rank
    nmodes = n * (n // 2 + 1)
    per_rank = nmodes * n * 3 // nb_ranks
    iface = nmodes * 3 * 2            # two z-planes, every mode, every component

    def bench(fn, repeats=20):
        fn()
        comm.Barrier()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        comm.Barrier()
        return (time.perf_counter() - start) / repeats

    send = np.ones(per_rank, dtype=np.complex128)
    recv = np.empty_like(send)
    buf = np.ones(iface, dtype=np.complex128)
    rbuf = np.empty_like(buf)

    def alltoall():
        comm.Alltoall([send, MPI.DOUBLE_COMPLEX], [recv, MPI.DOUBLE_COMPLEX])

    def recursive_doubling():
        step = 1
        while step < nb_ranks:
            partner = rank ^ step
            if partner < nb_ranks:
                comm.Sendrecv([buf, MPI.DOUBLE_COMPLEX], dest=partner,
                              recvbuf=[rbuf, MPI.DOUBLE_COMPLEX],
                              source=partner)
            step <<= 1

    # Four all-to-alls per reference apply; two interface solves per hybrid apply.
    ta, tr = bench(alltoall), bench(recursive_doubling)
    if rank == 0:
        print(f"{nb_ranks:>6} {4*ta*1e6:>15.0f} {2*tr*1e6:>16.0f} "
              f"{4*send.nbytes*nb_ranks/1e6:>12.0f} "
              f"{2*buf.nbytes*nb_ranks/1e6:>13.0f} {(4*ta)/(2*tr):>9.1f}x")


def main():
    parser = argparse.ArgumentParser(
        prog="mg_tridiag_bench",
        description="Cost of the hybrid preconditioner's tridiagonal solve.")
    parser.add_argument("--fft", action="store_true",
                        help="2D versus 3D transform pair (GPU)")
    parser.add_argument("--sweep", action="store_true",
                        help="fused block-Thomas sweep (GPU)")
    parser.add_argument("--comm", action="store_true",
                        help="all-to-all versus interface exchange (run under "
                             "mpirun)")
    parser.add_argument("-n", "--sizes", type=int, nargs="+",
                        default=[128, 192, 256], help="grid sizes")
    args = parser.parse_args()
    if not any((args.fft, args.sweep, args.comm)):
        parser.error("pick at least one of --fft / --sweep / --comm")
    if args.fft:
        cmd_fft(args.sizes)
    if args.sweep:
        cmd_sweep(args.sizes)
    if args.comm:
        cmd_comm(args.sizes[-1])


if __name__ == "__main__":
    main()
