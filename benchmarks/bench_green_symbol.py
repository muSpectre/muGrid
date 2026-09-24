"""Evaluated vs stored reference symbol on a device: apply time and memory.

    python bench_green_symbol.py N dtype mode      # mode: evaluated | stored

Prints one CSV line: N,dtype,mode,setup_s,mem_MB,apply_ms,fft_ms,symbol_ms
"""
import sys
import time

import cupy as cp
import numpy as np

import muGrid
from muGrid.Preconditioners import make_reference_stiffness_preconditioner

N, dtype, mode = int(sys.argv[1]), np.dtype(sys.argv[2]), sys.argv[3]
dim = 3
comm = muGrid.Communicator()
engine = muGrid.FFTEngine((N,) * dim, comm, device=muGrid.Device.gpu())

r = engine.real_space_field("r", components=(dim,), dtype=dtype)
z = engine.real_space_field("z", components=(dim,), dtype=dtype)
r.p[...] = cp.asarray(
    np.random.default_rng(0).standard_normal(r.p.shape).astype(dtype))

pool = cp.get_default_memory_pool()
pool.free_all_blocks()
cp.cuda.runtime.deviceSynchronize()
free0, total = cp.cuda.runtime.memGetInfo()

t0 = time.perf_counter()
prec = make_reference_stiffness_preconditioner(
    engine, nb_components=dim, dtype=dtype, element=muGrid.FEMElement.q1,
    grid_spacing=(1.0 / N,) * dim, lambda_ref=1.3, mu_ref=0.7,
    evaluate_symbol=(mode == "evaluated"), name="bench")
cp.cuda.runtime.deviceSynchronize()
setup = time.perf_counter() - t0


def timed(fn, reps):
    for _ in range(3):
        fn()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / reps


reps = 20
apply_ms = timed(lambda: prec(r, z), reps)
work = prec._work
fft_ms = timed(lambda: (engine.fft(r, work), engine.ifft(work, z)), reps)
cp.cuda.runtime.deviceSynchronize()
free1, _ = cp.cuda.runtime.memGetInfo()
# Held after setup and applies: muGrid fields plus the cupy pool's reserve,
# which does not shrink, so it is the pool's high-water mark.
mem_MB = (free0 - free1) / 1e6
print(f"{N},{dtype.name},{mode},{setup:.2f},{mem_MB:.1f},{apply_ms:.3f},"
      f"{fft_ms:.3f},{apply_ms - fft_ms:.3f}", flush=True)
