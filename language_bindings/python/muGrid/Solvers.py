"""
Collection of simple parallel solvers
"""

from . import linalg


def conjugate_gradients(
    comm,
    fc,
    b,
    x,
    hessp: callable,
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
    b : muGrid.Field
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution (modified in place).
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
        Signature: hessp(input_field, output_field) where both are muGrid.Field.
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with signature:
        callback(iteration, state_dict) where state_dict contains:
        - "x": solution field
        - "r": residual field
        - "p": search direction field
        - "rr": squared residual norm (float)
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

    with timed("startup"):
        # Create temporary fields with matching component shape
        # r: residual field
        # p: search direction field
        # Ap: Hessian product field
        r = fc.real_field("cg-residual", b.components_shape)
        p = fc.real_field("cg-search-direction", b.components_shape)
        Ap = fc.real_field("cg-hessian-product", b.components_shape)

        # Initial residual: r = b - A*x
        hessp(x, Ap)
        # r = b (copy)
        linalg.copy(b, r)
        # r = r - Ap = b - A*x (axpy with alpha=-1)
        linalg.axpy(-1.0, Ap, r)

        # Initial search direction: p = r
        linalg.copy(r, p)

        if callback:
            rr = comm.sum(linalg.norm_sq(r))
            callback(0, {"x": x, "r": r, "p": p, "rr": rr})

        with timed("dot_rr"):
            rr = comm.sum(linalg.norm_sq(r))
        rr_val = float(rr)

        if rr_val < tol_sq:
            return x

    with timed("iteration"):
        for iteration in range(maxiter):
            # Compute Hessian product: Ap = A * p
            with timed("hessp"):
                hessp(p, Ap)

            # Compute pAp for step size
            with timed("dot_pAp"):
                pAp = comm.sum(linalg.vecdot(p, Ap))

            # Compute alpha
            alpha = rr / pAp

            # Update solution: x += alpha * p
            with timed("update_x"):
                linalg.axpy(alpha, p, x)

            # Update residual and compute norm in one pass: r -= alpha * Ap
            with timed("update_r"):
                next_rr = comm.sum(linalg.axpy_norm_sq(-alpha, Ap, r))
            next_rr_val = float(next_rr)

            if callback:
                with timed("callback"):
                    callback(iteration + 1, {"x": x, "r": r, "p": p, "rr": next_rr})

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
                linalg.axpby(1.0, r, beta, p)

    raise RuntimeError("Conjugate gradient algorithm did not converge")
