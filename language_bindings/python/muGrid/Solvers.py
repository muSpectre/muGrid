"""
Collection of simple parallel solvers
"""

from muGrid import real_field

# Import the C++ extension module
# Try relative import first (for installed wheels),
# fall back to absolute (for development)
try:
    from ._muGrid import Communicator, FieldCollection
except ImportError:
    from _muGrid import Communicator, FieldCollection

from .Field import wrap_field


def conjugate_gradients(
    comm: Communicator,
    fc: FieldCollection,
    hessp: callable,
    b,
    x,
    tol: float = 1e-6,
    maxiter: int = 1000,
    callback: callable = None,
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
    fc : muGrid.FieldCollection
        Collection holding temporary fields of the CG algorithm.
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
    b : muGrid.Field
        Right-hand side vector (C++ field or wrapped Field).
    x : muGrid.Field
        Initial guess for the solution (C++ field or wrapped Field).
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with the current solution, residual,
        and search direction.

    Returns
    -------
    x : array_like
        Approximate solution to the system Ax = b. (Same as input field x.)
    """
    tol_sq = tol * tol

    # Wrap fields for array access if they are not already wrapped
    b = wrap_field(b)
    x = wrap_field(x)

    p = real_field(fc, "cg-search-direction")
    Ap = real_field(fc, "cg-hessian-product")

    hessp(x, Ap)
    p.s[...] = b.s - Ap.s
    r = p.s.copy()  # residual

    if callback:
        callback(0, x.s, r, p.s)

    rr = comm.sum(r.ravel().dot(r.ravel()))  # initial residual dot product
    if rr < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(p.s.ravel().dot(Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rr / pAp
        x.s[...] += alpha * p.s
        r -= alpha * Ap.s

        if callback:
            callback(iteration + 1, x.s, r, p.s)

        # Check convergence
        next_rr = comm.sum(r.ravel().dot(r.ravel()))
        if next_rr < tol_sq:
            return x

        # Update search direction
        beta = next_rr / rr
        rr = next_rr
        p.s[...] *= beta
        p.s[...] += r

    raise RuntimeError("Conjugate gradient algorithm did not converge")
