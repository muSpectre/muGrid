"""
Collection of simple parallel solvers
"""

import numpy as np
from _muGrid import Communicator, FieldCollection

from .Field import Field as FieldWrapper, wrap_field


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
    b_wrap = b if isinstance(b, FieldWrapper) else wrap_field(b)
    x_wrap = x if isinstance(x, FieldWrapper) else wrap_field(x)

    p_cpp = fc.real_field("cg-search-direction")
    Ap_cpp = fc.real_field("cg-hessian-product")
    p_wrap = wrap_field(p_cpp)
    Ap_wrap = wrap_field(Ap_cpp)

    hessp(x, Ap_cpp)
    p_wrap.s[...] = b_wrap.s - Ap_wrap.s
    r = np.copy(p_wrap.s)  # residual

    if callback:
        callback(0, x_wrap.s, r, p_wrap.s)

    rr = comm.sum(np.dot(r.ravel(), r.ravel()))  # initial residual dot product
    if rr < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p_cpp, Ap_cpp)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p_wrap.s.ravel(), Ap_wrap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rr / pAp
        x_wrap.s[...] += alpha * p_wrap.s
        r -= alpha * Ap_wrap.s

        if callback:
            callback(iteration + 1, x_wrap.s, r, p_wrap.s)

        # Check convergence
        next_rr = comm.sum(np.dot(r.ravel(), r.ravel()))
        if next_rr < tol_sq:
            return x

        # Update search direction
        beta = next_rr / rr
        rr = next_rr
        p_wrap.s[...] *= beta
        p_wrap.s[...] += r

    raise RuntimeError("Conjugate gradient algorithm did not converge")
