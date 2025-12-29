"""
Collection of simple parallel solvers
"""

from . import linalg


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
        record timing for the hessp (Hessian-vector product) operations.

    Returns
    -------
    x : muGrid.Field
        Solution to the system Ax = b (same as input field x).

    Raises
    ------
    RuntimeError
        If the algorithm does not converge within maxiter iterations,
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
    # r: residual field
    # p: search direction field
    # Ap: Hessian product field
    r = fc.real_field("cg-residual", components_shape)
    p = fc.real_field("cg-search-direction", components_shape)
    Ap = fc.real_field("cg-hessian-product", components_shape)

    # Get underlying C++ field objects for efficient linalg operations
    # (avoids Python wrapper overhead in the hot loop)
    b_cpp = b._cpp
    x_cpp = x._cpp
    r_cpp = r._cpp
    p_cpp = p._cpp
    Ap_cpp = Ap._cpp

    # Initial residual: r = b - A*x
    hessp(x, Ap)
    # r = b (copy)
    linalg.copy(b_cpp, r_cpp)
    # r = r - Ap = b - A*x (axpy with alpha=-1)
    linalg.axpy(-1.0, Ap_cpp, r_cpp)

    # Initial search direction: p = r
    linalg.copy(r_cpp, p_cpp)

    if callback:
        callback(0, x.s, r.s, p.s)

    with timed("dot_rr"):
        rr = comm.sum(linalg.norm_sq(r_cpp))
    rr_val = float(rr)

    if rr_val < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product: Ap = A * p
        with timed("hessp"):
            hessp(p, Ap)

        # Compute pAp for step size
        with timed("dot_pAp"):
            pAp = comm.sum(linalg.vecdot(p_cpp, Ap_cpp))

        # Compute alpha
        alpha = rr / pAp

        # Update solution: x += alpha * p
        with timed("update_x"):
            linalg.axpy(alpha, p_cpp, x_cpp)

        # Update residual: r -= alpha * Ap
        with timed("update_r"):
            linalg.axpy(-alpha, Ap_cpp, r_cpp)

        if callback:
            callback(iteration + 1, x.s, r.s, p.s)

        # Compute next residual norm
        with timed("dot_rr"):
            next_rr = comm.sum(linalg.norm_sq(r_cpp))
        next_rr_val = float(next_rr)

        # Check for numerical issues (NaN indicates non-positive-definite H)
        if next_rr_val != next_rr_val:  # NaN check
            raise RuntimeError(
                "Residual became NaN - Hessian may not be positive definite"
            )

        if next_rr_val < tol_sq:
            return x

        # Compute beta
        beta = next_rr / rr

        # Update rr for next iteration
        rr = next_rr

        # Update search direction: p = r + beta * p
        with timed("update_p"):
            linalg.scal(beta, p_cpp)
            linalg.axpy(1.0, r_cpp, p_cpp)

    raise RuntimeError("Conjugate gradient algorithm did not converge")
