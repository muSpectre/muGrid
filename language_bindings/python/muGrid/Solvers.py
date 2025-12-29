"""
Collection of simple parallel solvers
"""

import numpy as np


def conjugate_gradients(
    comm,
    fc,
    hessp: callable,
    b,
    x,
    tol: float = 1e-6,
    maxiter: int = 1000,
    callback: callable = None,
    timer=None,
    check_positive_definite: bool = False,
    profile_gpu: bool = False,
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution x
    until the residual ||Ax - b|| is less than tol or until maxiter iterations
    are reached.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing.
    fc : muGrid.GlobalFieldCollection, muGrid.LocalFieldCollection, or
         muGrid.CartesianDecomposition
        Collection for creating temporary fields used by the CG algorithm.
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
        Signature: hessp(input_field, output_field) where both are muGrid.Field.
    b : muGrid.Field
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution (modified in place).
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with signature:
        callback(iteration, x_array, residual_array, search_direction_array)
    timer : muGrid.Timer, optional
        Timer object for performance profiling. If provided, the solver will
        record timing for internal operations (dot products, vector updates).
    check_positive_definite : bool, optional
        If True, check that the Hessian is positive definite each iteration.
        This adds GPU synchronization overhead. Default is False.
    profile_gpu : bool, optional
        If True, add explicit GPU synchronization around each operation for
        accurate profiling. This adds overhead but shows true GPU time.
        Default is False.

    Returns
    -------
    x : muGrid.Field
        Solution to the system Ax = b (same as input field x).

    Raises
    ------
    RuntimeError
        If the algorithm does not converge within maxiter iterations,
        or if the Hessian is not positive definite (when check_positive_definite=True),
        or if the residual becomes NaN (indicating numerical issues).
    """
    tol_sq = tol * tol

    # Timer context manager (no-op if timer is None)
    from contextlib import nullcontext

    def timed(name):
        return timer(name) if timer is not None else nullcontext()

    # GPU sync function for profiling (no-op if not profiling or not on GPU)
    def gpu_sync():
        if profile_gpu:
            # Try to sync GPU if available
            try:
                import cupy

                # Use device synchronize for full GPU sync (more reliable on ROCm/HIP)
                cupy.cuda.runtime.deviceSynchronize()
            except (ImportError, Exception):
                pass

    # Get spatial dimension from the field collection
    spatial_dim = len(fc.nb_grid_pts)

    # Extract component shape from b: b.s.shape = (*components, nb_sub_pts, *spatial)
    # The +1 accounts for the nb_sub_pts dimension
    components_shape = b.s.shape[: -(spatial_dim + 1)]

    # Create temporary fields with matching component shape
    p = fc.real_field("cg-search-direction", components_shape)
    Ap = fc.real_field("cg-hessian-product", components_shape)

    # Initial residual: r = b - A*x
    gpu_sync()
    hessp(x, Ap)
    gpu_sync()

    with timed("cg_init_residual"):
        gpu_sync()
        p.s[...] = b.s - Ap.s
        gpu_sync()

    with timed("cg_copy"):
        gpu_sync()
        r = p.s.copy()
        gpu_sync()

    if callback:
        callback(0, x.s, r, p.s)

    # Get array module (numpy or cupy) from the residual array
    xp = type(r).__module__.split(".")[0]
    if xp == "cupy":
        import cupy

        xp = cupy
    else:
        xp = np

    with timed("cg_dot_rr"):
        gpu_sync()
        rr_local = xp.sum(r * r)
        gpu_sync()

    with timed("cg_allreduce"):
        gpu_sync()
        rr = comm.sum(rr_local)
        gpu_sync()

    with timed("cg_sync_for_check"):
        rr_val = float(rr)

    if rr_val < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product: Ap = A * p
        with timed("cg_hessp_wrapper"):
            gpu_sync()
            hessp(p, Ap)
            gpu_sync()

        # Compute pAp for step size
        with timed("cg_dot_pAp"):
            gpu_sync()
            pAp_local = xp.sum(p.s * Ap.s)
            gpu_sync()

        with timed("cg_allreduce"):
            gpu_sync()
            pAp = comm.sum(pAp_local)
            gpu_sync()

        # Optional positive-definiteness check (adds sync overhead)
        if check_positive_definite:
            with timed("cg_sync_for_check"):
                if float(pAp) <= 0:
                    raise RuntimeError("Hessian is not positive definite")

        # Compute alpha (stays on GPU if pAp is GPU array)
        with timed("cg_scalar_div_alpha"):
            gpu_sync()
            alpha = rr / pAp
            gpu_sync()

        # Update solution: x += alpha * p
        with timed("cg_update_x"):
            gpu_sync()
            x.s[...] += alpha * p.s
            gpu_sync()

        # Update residual: r -= alpha * Ap
        with timed("cg_update_r"):
            gpu_sync()
            r -= alpha * Ap.s
            gpu_sync()

        if callback:
            callback(iteration + 1, x.s, r, p.s)

        # Compute next residual norm
        with timed("cg_dot_rr"):
            gpu_sync()
            next_rr_local = xp.sum(r * r)
            gpu_sync()

        with timed("cg_allreduce"):
            gpu_sync()
            next_rr = comm.sum(next_rr_local)
            gpu_sync()

        # Sync to check convergence
        with timed("cg_sync_for_check"):
            next_rr_val = float(next_rr)

        # Check for numerical issues (NaN indicates non-positive-definite H)
        if next_rr_val != next_rr_val:  # NaN check
            raise RuntimeError(
                "Residual became NaN - Hessian may not be positive definite"
            )

        if next_rr_val < tol_sq:
            return x

        # Compute beta
        with timed("cg_scalar_div_beta"):
            gpu_sync()
            beta = next_rr / rr
            gpu_sync()

        # Update rr for next iteration
        rr = next_rr

        # Update search direction: p = r + beta * p
        with timed("cg_update_p_scale"):
            gpu_sync()
            p.s[...] *= beta
            gpu_sync()

        with timed("cg_update_p_add"):
            gpu_sync()
            p.s[...] += r
            gpu_sync()

    raise RuntimeError("Conjugate gradient algorithm did not converge")
