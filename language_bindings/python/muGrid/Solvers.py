"""
Collection of simple parallel solvers
"""

import numpy as np
from _muGrid import Communicator, Field, FieldCollection


def conjugate_gradients(
    comm: Communicator,
    fc: FieldCollection,
    hessp: callable,
    b: Field,
    x: Field,
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
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution.
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
    p = fc.real_field("cg-search-direction")
    Ap = fc.real_field("cg-hessian-product")
    hessp(x, Ap)
    p.s = b.s - Ap.s
    r = np.copy(p.s)  # residual

    if callback:
        callback(0, x.s, r, p.s)

    rr = comm.sum(np.dot(r.ravel(), r.ravel()))  # initial residual dot product
    if rr < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rr / pAp
        x.s += alpha * p.s
        r -= alpha * Ap.s

        if callback:
            callback(iteration + 1, x.s, r, p.s)

        # Check convergence
        next_rr = comm.sum(np.dot(r.ravel(), r.ravel()))
        if next_rr < tol_sq:
            return x

        # Update search direction
        beta = next_rr / rr
        rr = next_rr
        p.s *= beta
        p.s += r

    raise RuntimeError("Conjugate gradient algorithm did not converge")
