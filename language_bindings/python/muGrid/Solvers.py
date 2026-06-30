"""
Collection of simple parallel solvers
"""

import warnings

import numpy as np

from . import linalg


class ConvergenceError(RuntimeError):
    """
    Raised when an iterative solver fails to converge. Subclasses
    ``RuntimeError`` for backwards compatibility; catching it specifically
    avoids masking unrelated runtime errors (e.g. out-of-memory) as
    non-convergence.
    """


def conjugate_gradients(
    comm,
    fc,
    b,
    x,
    hessp: callable,
    prec: callable = None,
    tol: float = None,
    maxiter: int = 1000,
    callback: callable = None,
    timer=None,
    rtol: float = None,
    atol: float = 0.0,
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution
    x until the residual satisfies ``||b - Ax|| <= max(rtol * ||b||, atol)``
    or until maxiter iterations are reached.

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
    prec : callable, optional
        Function that applies the preconditioner P to a vector r: z = P*r.
        Signature: prec(input_field, output_field) where both are muGrid.Field.
        If None, no preconditioning is applied (P is identity).
    tol : float, optional
        Deprecated alias for `atol`. Passing it restores the historic,
        purely absolute criterion ``||b - Ax|| < tol`` (it also sets
        ``rtol = 0`` unless `rtol` is given explicitly). Use `rtol`/`atol`
        instead.
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
        record timing for the individual solver operations (Hessian-vector
        products, preconditioner applications, dot products and vector
        updates).
    rtol : float, optional
        Relative tolerance: convergence when ``||b - Ax|| <= rtol * ||b||``.
        The default is 1e-6. A relative criterion is robust against the
        scale of the right-hand side and the grid size; an absolute one is
        unreachable in double precision when ``||b||`` is large.
    atol : float, optional
        Absolute tolerance: convergence when ``||b - Ax|| <= atol``,
        whichever of the two criteria is weaker. The default is 0 (purely
        relative convergence).

    Returns
    -------
    x : muGrid.Field
        Solution to the system Ax = b (same as input field x).

    Raises
    ------
    ConvergenceError
        If the algorithm does not converge within maxiter iterations,
        or if the residual becomes NaN (indicating numerical issues).
    """
    if tol is not None:
        warnings.warn(
            "`tol` is deprecated; use `atol` for an absolute or `rtol` for "
            "a relative convergence criterion",
            DeprecationWarning,
            stacklevel=2,
        )
        atol = tol
        if rtol is None:
            rtol = 0.0
    if rtol is None:
        rtol = 1e-6

    # Timer context manager (no-op if timer is None)
    from contextlib import nullcontext

    def timed(name):
        return timer(name) if timer is not None else nullcontext()

    # Without a preconditioner, the preconditioned residual z is identically r,
    # so alias z onto r and skip the apply instead of allocating (and copying
    # into) a fourth full work vector. This removes dim doubles/voxel -- the
    # largest remaining term once the strain/stress fields are gone.
    unpreconditioned = prec is None
    if unpreconditioned:
        def prec(src, dst):  # z is aliased to r; nothing to apply
            return None

    # Match the precision of the work fields to the right-hand side, so a
    # single-precision (float32) solve stays in single precision throughout
    # (and a default double-precision solve is unchanged).
    dtype = getattr(b, "dtype", np.float64)

    with timed("startup"):
        # Create temporary fields with matching component shape
        # r: residual field
        # p: search direction field
        # z: preconditioned search direction field (aliased to r if no prec)
        # Ap: Hessian product field
        r = fc.real_field("cg-residual", b.components_shape, dtype=dtype)
        p = fc.real_field("cg-search-direction", b.components_shape, dtype=dtype)
        if unpreconditioned:
            z = r
        else:
            z = fc.real_field(
                "cg-preconditioned-residual", b.components_shape, dtype=dtype
            )
        Ap = fc.real_field("cg-hessian-product", b.components_shape, dtype=dtype)

        # Initial residual: r = b - A*x
        hessp(x, Ap)
        # r = b (copy)
        linalg.copy(b, r)
        # r = r - Ap = b - A*x (axpy with alpha=-1)
        linalg.axpy(-1.0, Ap, r)

        # preconditioner
        with timed("prec"):
            prec(r, z)

        # Initial search direction: p = z
        linalg.copy(z, p)

        with timed("dot_rr"):
            bb = comm.sum(linalg.norm_sq(b))
            rr = comm.sum(linalg.norm_sq(r))
            rz = comm.sum(linalg.vecdot(r, z))

        # Convergence threshold on the squared residual norm:
        # ||r|| <= max(rtol * ||b||, atol)
        tol_sq = max(rtol * rtol * float(bb), atol * atol)

        if callback:
            callback(0, {"x": x, "r": r, "p": p, "z": z, "rr": rr, "rz": rz})

        rr_val = float(rr)

        if rr_val <= tol_sq:
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
            alpha = rz / pAp

            # Update solution: x += alpha * p
            with timed("update_x"):
                linalg.axpy(alpha, p, x)

            # Update residual: r -= alpha * Ap
            with timed("update_r"):
                next_rr = comm.sum(linalg.axpy_norm_sq(-alpha, Ap, r))

            next_rr_val = float(next_rr)

            # apply preconditioner z=P*r
            with timed("prec"):
                prec(r, z)

            # Compute next_rz after applying preconditioner
            with timed("dot_rz"):
                next_rz = comm.sum(linalg.vecdot(r, z))

            if callback:
                with timed("callback"):
                    callback(
                        iteration + 1,
                        {
                            "x": x,
                            "r": r,
                            "p": p,
                            "z": z,
                            "rr": next_rr,
                            "rz": next_rz,
                        },
                    )

            # Check for numerical issues (NaN indicates non-positive-definite H)
            if next_rr_val != next_rr_val:  # NaN check
                raise ConvergenceError(
                    "Residual became NaN - Hessian may not be positive definite"
                )

            if next_rr_val <= tol_sq:
                return x

            # Compute beta
            beta = next_rz / rz
            # Update rz for next iteration
            rz = next_rz

            # Update search direction: p = z + beta * p
            with timed("update_p"):
                linalg.axpby(1.0, z, beta, p)

    raise ConvergenceError(
        "Preconditioned conjugate gradient algorithm did not converge"
    )


def conjugate_gradients_pipelined(
    comm,
    fc,
    b,
    x,
    hessp: callable,
    prec: callable = None,
    tol: float = None,
    maxiter: int = 1000,
    callback: callable = None,
    timer=None,
    rtol: float = None,
    atol: float = 0.0,
):
    """
    Pipelined preconditioned conjugate gradients (Ghysels & Vanroose, 2014).

    PROTOTYPE. Drop-in replacement for :func:`conjugate_gradients` with the same
    signature and semantics, but reorganised so that the per-iteration inner
    products are merged into a **single global reduction**, instead of the three
    (``dot_pAp``, ``dot_rr``, ``dot_rz``) that standard PCG performs.

    Standard PCG has the dot products on the critical path: you must finish one
    reduction to form a scalar before the next vector/reduction can be computed.
    The pipelined formulation introduces auxiliary recurrences (``z, q, s, p``)
    so that the two coupling inner products ``(r, u)`` and ``(w, u)`` — plus the
    residual norm used for convergence — can be computed together, and the
    matrix-vector product ``n = A m`` and preconditioner apply ``m = M⁻¹ w`` of
    the *next* step no longer depend on that reduction's result.

    Cost per iteration vs. standard PCG:
      - reductions:        1 (combined)        vs. 3
      - operator applies:  1 (``A m``)         vs. 1
      - preconditioner:    1 (``M⁻¹ w``)       vs. 1
      - vector updates:    8 axpy/axpby        vs. ~4

    i.e. it trades a few extra cheap vector updates for two fewer
    synchronisations — a win when reductions are latency-bound (MPI allreduce at
    high rank counts, or the blocking device->host copy on the GPU once the
    inner products are fused into one kernel).

    Reference: P. Ghysels and W. Vanroose, *Hiding global synchronization
    latency in the preconditioned Conjugate Gradient algorithm*, Parallel
    Computing 40 (2014) 224-238.

    .. note::
        The residual, preconditioned residual and operator image are advanced by
        recurrence rather than recomputed, so this variant is slightly more
        susceptible to rounding than standard CG. For ill-conditioned systems a
        residual-replacement strategy may be needed; this prototype omits it.
    """
    if tol is not None:
        warnings.warn(
            "`tol` is deprecated; use `atol` for an absolute or `rtol` for "
            "a relative convergence criterion",
            DeprecationWarning,
            stacklevel=2,
        )
        atol = tol
        if rtol is None:
            rtol = 0.0
    if rtol is None:
        rtol = 1e-6

    from contextlib import nullcontext

    def timed(name):
        return timer(name) if timer is not None else nullcontext()

    if prec is None:
        prec = linalg.copy

    # Match the precision of the work fields to the right-hand side (float32
    # rhs -> single-precision solve throughout; double rhs is unchanged).
    dtype = getattr(b, "dtype", np.float64)

    with timed("startup"):
        # Work fields (zero-initialised by the collection)
        r = fc.real_field("pcg-residual", b.components_shape, dtype=dtype)
        u = fc.real_field("pcg-prec-residual", b.components_shape, dtype=dtype)
        w = fc.real_field("pcg-w", b.components_shape, dtype=dtype)
        m = fc.real_field("pcg-m", b.components_shape, dtype=dtype)
        n = fc.real_field("pcg-n", b.components_shape, dtype=dtype)
        p = fc.real_field("pcg-p", b.components_shape, dtype=dtype)
        s = fc.real_field("pcg-s", b.components_shape, dtype=dtype)
        q = fc.real_field("pcg-q", b.components_shape, dtype=dtype)
        z = fc.real_field("pcg-z", b.components_shape, dtype=dtype)

        # r = b - A x
        hessp(x, r)
        linalg.axpby(1.0, b, -1.0, r)
        # u = M^{-1} r ; w = A u
        with timed("prec"):
            prec(r, u)
        hessp(u, w)
        # Recurrence accumulators start at zero
        for f in (p, s, q, z):
            linalg.scal(0.0, f)

        bb = comm.sum(linalg.norm_sq(b))
        tol_sq = max(rtol * rtol * float(bb), atol * atol)

        gamma_prev = None
        alpha_prev = None

    with timed("iteration"):
        for iteration in range(maxiter):
            # Single combined reduction: (r,u), (w,u), (r,r). The three inner
            # products are computed by one fused kernel (one device->host copy)
            # and reduced across ranks in a *non-blocking* allreduce. The
            # preconditioner apply m = M^{-1} w and operator apply n = A m below
            # do not depend on gamma/delta/rr, so they run while the reduction
            # is in flight, hiding the global synchronisation latency (the
            # multi-node payoff of the pipelined formulation).
            with timed("reduce-begin"):
                local = np.array(linalg.pipelined_cg_dots(r, u, w))
                reduce_handle = comm.isum(local)

            # Apply preconditioner and operator for this step (overlaps the
            # in-flight reduction).
            with timed("prec"):
                prec(w, m)
            with timed("hessp"):
                hessp(m, n)

            # Complete the overlapped reduction.
            with timed("reduce-wait"):
                gamma, delta, rr = (
                    float(v) for v in comm.isum_wait(reduce_handle).ravel()
                )

            if callback:
                with timed("callback"):
                    callback(
                        iteration,
                        {"x": x, "r": r, "p": p, "rr": rr, "rz": gamma},
                    )

            if rr != rr:  # NaN
                raise ConvergenceError(
                    "Residual became NaN - Hessian may not be positive definite"
                )
            if rr <= tol_sq:
                return x

            if iteration == 0:
                beta = 0.0
                alpha = gamma / delta
            else:
                beta = gamma / gamma_prev
                alpha = gamma / (delta - beta * gamma / alpha_prev)
            gamma_prev = gamma
            alpha_prev = alpha

            with timed("update"):
                # z = n + beta z ; q = m + beta q ; s = w + beta s ; p = u + beta p
                linalg.axpby(1.0, n, beta, z)
                linalg.axpby(1.0, m, beta, q)
                linalg.axpby(1.0, w, beta, s)
                linalg.axpby(1.0, u, beta, p)
                # x += alpha p ; r -= alpha s ; u -= alpha q ; w -= alpha z
                linalg.axpy(alpha, p, x)
                linalg.axpy(-alpha, s, r)
                linalg.axpy(-alpha, q, u)
                linalg.axpy(-alpha, z, w)

    raise ConvergenceError(
        "Pipelined preconditioned conjugate gradient algorithm did not converge"
    )
