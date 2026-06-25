# Benchmark: homogenization

Wall time of the FEM elasticity [homogenization example](examples.md)
(`examples/homogenization.py`, fused stiffness kernel), across log-spaced **3D**
grid sizes. Lower is better.

!!! info "Test machine & code version"
    - **CPU:** AMD Instinct MI300A Accelerator (192 logical cores)
    - **GPU:** 4x AMD Instinct MI300A
    - **muGrid:** `0.109.0-25-g6f430f83-dirty` — run 2026-06-25T13:07:28

Run configuration: 3D single spherical inclusion, fused stiffness kernel,
6 load cases, fixed `100` CG iterations per load case — i.e. a **fixed work
budget** so every configuration performs identical arithmetic. Times are the
solver wall time (`total_time_seconds`, excluding setup).

## Time vs. grid size

The plot below merges three ways of running the *same* solve on this machine:

- **CPU (1 core)** — a single core, MPI disabled. muGrid's compute kernels carry
  no OpenMP, so a non-MPI CPU run uses exactly one core.
- **CPU (92 cores, MPI)** — the whole CPU via MPI domain decomposition
  (`mpiexec -n 92`), the grid split into per-rank subdomains that exchange
  ghost layers each iteration.
- **GPU (1 device)** — the whole GPU.
- **GPU (N devices, MPI)** — all GPUs, one rank per device.

| Configuration | 16³ (4k) | 24³ (14k) | 32³ (33k) | 48³ (111k) | 64³ (262k) | 96³ (885k) | 128³ (2.1M) |
|---|---|---|---|---|---|---|---|
| CPU (1 core) | 1.16 | 3.82 | 9.17 | 30.5 | 74.6 | 251 | 602 |
| CPU (92 cores, MPI) | 0.127 | 0.165 | 0.236 | 0.594 | 1.05 | 3.54 | 7.69 |
| GPU (1 device) | 0.525 | 0.453 | 0.51 | 0.486 | 0.634 | 1.34 | 2.78 |
| GPU (4 devices, MPI) | 0.739 | 0.654 | 0.661 | 0.739 | 0.639 | 1.05 | 1.43 |

(values are **solve time in seconds**)

![Homogenization solve time vs. number of grid points](benchmark_homogenization.png)

Three regimes are visible. At tiny grids everything is overhead-bound and the
three are within a factor of two. In the **mid-range** (here ~32³–64³) the
GPU dominates: the heavy per-point FEM stiffness kernel keeps it busy, the
working set fits in device memory, and it beats even the full CPU. At the **high
end** the picture flips — a single CPU core is hopeless, but the **whole CPU**
(all 92 cores via MPI) overtakes the GPU once the problem outgrows GPU
memory (see below). The fair comparison is full-CPU-vs-GPU, not one-core-vs-GPU:
against all 92 cores the GPU's advantage is modest where it leads and
reverses where it does not.

!!! warning "GPU memory wall at large grids"
    A 128³ run needs about 5.85 GB of field storage, which nearly fills this
    GPU's 6 GB. As the working set approaches that limit the allocator
    oversubscribes to host memory and effective throughput collapses — already
    visible at 96³ (throughput drops below the 48³–64³ peak, and the GPU loses to
    the full CPU) and severe at 128³. On this 6 GB card the GPU's fast path tops
    out around 64³; larger 3D grids need a bigger-memory GPU or MPI domain
    decomposition across several GPUs (see the multi-GPU note below).

!!! note "Multi-GPU"
    `homogenization.py` binds each MPI rank to a distinct GPU (round-robin over
    the visible devices), so `mpiexec -n <#GPUs> python homogenization.py -d gpu`
    runs one rank per GPU. This benchmark adds a *GPU (N devices, MPI)* curve
    automatically when more than one GPU is present. **Runs with more than one GPU show the multi-GPU curve.**

## MPI strong scaling

Strong scaling of the same 3D fused solve (fixed problem size, increasing MPI
ranks), with `E_eff` identical across all rank counts. The grid is split into
per-rank subdomains that exchange ghost layers each iteration. Two decompositions
are measured: across the 92-core CPU (one rank per core), and — on a
multi-GPU host — across the GPUs (one rank per device, round-robin).

### Strong scaling on the CPU

**64³ (262,144 points)**

| Cores | Time (s) | Speedup | Parallel eff. | Agg. GB/s |
|---|---|---|---|---|
| 1 | 74.44 | 1.00× | 100% | 1.6 |
| 2 | 37.61 | 1.98× | 99% | 3.1 |
| 4 | 18.83 | 3.95× | 99% | 6.3 |
| 8 | 9.77 | 7.62× | 95% | 12.1 |
| 16 | 4.89 | 15.23× | 95% | 24.2 |
| 32 | 2.71 | 27.46× | 86% | 43.6 |
| 64 | 1.34 | 55.54× | 87% | 88.2 |
| 92 | 1.07 | 69.55× | 76% | 110.4 |

**96³ (884,736 points)**

| Cores | Time (s) | Speedup | Parallel eff. | Agg. GB/s |
|---|---|---|---|---|
| 1 | 258.01 | 1.00× | 100% | 1.5 |
| 2 | 126.13 | 2.05× | 102% | 3.2 |
| 4 | 64.36 | 4.01× | 100% | 6.2 |
| 8 | 34.60 | 7.46× | 93% | 11.5 |
| 16 | 16.87 | 15.30× | 96% | 23.6 |
| 32 | 8.65 | 29.84× | 93% | 46.1 |
| 64 | 4.36 | 59.24× | 93% | 91.6 |
| 92 | 3.57 | 72.25× | 79% | 111.7 |

### Strong scaling on the GPU(s)

**64³ (262,144 points)**

| GPUs | Time (s) | Speedup | Parallel eff. | Agg. GB/s |
|---|---|---|---|---|
| 1 | 0.65 | 1.00× | 100% | 182.7 |
| 2 | 0.61 | 1.06× | 53% | 193.7 |
| 4 | 0.78 | 0.83× | 21% | 152.1 |

**96³ (884,736 points)**

| GPUs | Time (s) | Speedup | Parallel eff. | Agg. GB/s |
|---|---|---|---|---|
| 1 | 1.34 | 1.00× | 100% | 298.7 |
| 2 | 0.92 | 1.46× | 73% | 435.4 |
| 4 | 0.89 | 1.50× | 38% | 449.0 |

![Homogenization MPI strong scaling](benchmark_homogenization_mpi.png)

The solve is memory-bandwidth-bound, so aggregate throughput keeps climbing as
ranks are added but parallel efficiency falls once per-rank subdomains get small
and ghost exchange plus CG dot-product reductions start to dominate. On the GPU
side, the per-device working set must stay large enough to hide the
inter-device communication, so the larger grids scale best.

All data points live in the shared benchmark database `benchmarks/results.csv`
(date, code version, machine, parameters, results). This page is generated by
`examples/benchmark_homogenization.py`; re-render it from the database (no
recompute) with `--render-only`, or run a fresh measurement that appends a new
dated row set:

```bash
python examples/benchmark_homogenization.py \
    --doc-out docs/benchmark_homogenization.md \
    --plot-out docs/benchmark_homogenization.png
```
