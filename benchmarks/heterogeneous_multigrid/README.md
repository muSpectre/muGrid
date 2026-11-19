# Heterogeneous multigrid: cross-checks

Scripts behind the performance tables of
[`docs/heterogeneous_multigrid.md`](../../docs/heterogeneous_multigrid.md). Both compare
muGrid's `MultigridPreconditioner` with the multigrid of muFFTTO
(`muFFTTO/multigrid.py`, re-discretised coarse levels), which they import from a muFFTTO checkout (`--muffto` or `$MUFFTTO_PATH`).

- `compare_muffto.py` runs the production case: the Cahn-Hilliard phase field of muFFTTO's
  multigrid study (`--data` or `$MUFFTTO_CH3D_DATA`), contrast 10⁴, fused 3D operator, PCG.
  - `--method` selects the preconditioner: `mugrid`, `muffto`, `green` or `green_jacobi`.
  - Each run is one process and prints one JSON line: iterations, true residual,
    homogenised stress, setup, V-cycle and CG time, GPU memory per voxel from `nvidia-smi`.
- `cross_check_operators.py` hands muFFTTO's own stencil operators to muGrid's multigrid as a
  black box and compares the iteration counts with muFFTTO's hierarchy on the same operator. The
  elements are linear triangles, Q1, hexahedra, and one-point hexahedra with hourglass control.

```bash
export MUFFTTO_PATH=/path/to/muFFTTO MUFFTTO_CH3D_DATA=$MUFFTTO_PATH/experiments/Multigrid_preconditioning/data/CH3D
python compare_muffto.py --method mugrid --grid 128 --precision single --device gpu
python cross_check_operators.py --element linear_triangles --grid 128
```

The material arrays are cached in `cache/`, which is git-ignored.
