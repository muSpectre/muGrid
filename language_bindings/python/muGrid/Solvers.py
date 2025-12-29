"""
Collection of simple parallel solvers
"""


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

    # Get spatial dimension from the field collection
    spatial_dim = len(fc.nb_grid_pts)

    # Extract component shape from b: b.s.shape = (*components, nb_sub_pts, *spatial)
    # The +1 accounts for the nb_sub_pts dimension
    components_shape = b.s.shape[: -(spatial_dim + 1)]

    # Create temporary fields with matching component shape
    p = fc.real_field("cg-search-direction", components_shape)
    Ap = fc.real_field("cg-hessian-product", components_shape)

    # Initial residual: r = b - A*x
    hessp(x, Ap)
    with timed("cg_vector_ops"):
        p.s[...] = b.s - Ap.s
        r = p.s.copy()

    if callback:
        callback(0, x.s, r, p.s)

    with timed("cg_dot_products"):
        rr = comm.sum(r.ravel().dot(r.ravel()))

    # Initial convergence check (single sync point)
    with timed("cg_convergence_check"):
        rr_val = float(rr)
    if rr_val < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product: Ap = A * p
        hessp(p, Ap)

        # Compute step size (stays on GPU, no sync)
        with timed("cg_dot_products"):
            pAp = comm.sum(p.s.ravel().dot(Ap.s.ravel()))

        # Optional positive-definiteness check (adds sync overhead)
        if check_positive_definite:
            with timed("cg_convergence_check"):
                if float(pAp) <= 0:
                    raise RuntimeError("Hessian is not positive definite")

        # All arithmetic stays on GPU (no sync)
        alpha = rr / pAp

        # Update solution and residual
        with timed("cg_vector_ops"):
            x.s[...] += alpha * p.s
            r -= alpha * Ap.s

        if callback:
            callback(iteration + 1, x.s, r, p.s)

        # Compute next residual norm (stays on GPU)
        with timed("cg_dot_products"):
            next_rr = comm.sum(r.ravel().dot(r.ravel()))

        # Single sync point per iteration: convergence check
        with timed("cg_convergence_check"):
            next_rr_val = float(next_rr)

        # Check for numerical issues (NaN indicates non-positive-definite H)
        if next_rr_val != next_rr_val:  # NaN check
            raise RuntimeError(
                "Residual became NaN - Hessian may not be positive definite"
            )

        if next_rr_val < tol_sq:
            return x

        # Update search direction (stays on GPU, no sync)
        with timed("cg_vector_ops"):
            beta = next_rr / rr
            rr = next_rr
            p.s[...] *= beta
            p.s[...] += r

    raise RuntimeError("Conjugate gradient algorithm did not converge")
