"""
Collection of simple parallel solvers
"""

import itertools
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


#: Source of the per-call suffix of work-field names (see _WorkFields).
_work_field_ids = itertools.count()


class _WorkFields:
    """The work fields of one solver call, removed again when it ends.

    Each field is named ``<solver>#<call>-<role>``, so no two calls share one:
    a solve nested inside another's ``hessp``, ``prec`` or ``callback`` on the
    same collection gets its own vectors instead of overwriting the caller's,
    and a float32 solve after a float64 one is not handed the float64 fields.
    The fields are popped from the collection on exit, however the solve ends
    (return or exception), so repeated solves do not accumulate them. A
    callback may keep a field it was given: popping only drops the
    collection's reference.
    """

    def __init__(self, fc, solver):
        self._fc = fc
        self._prefix = f"{solver}#{next(_work_field_ids)}-"
        self._names = []

    def __enter__(self):
        return self

    def field(self, role, like, dtype):
        """A zero-initialised field shaped like `like`, in `dtype`."""
        name = self._prefix + role
        field = self._fc.real_field(name, like.components_shape,
                                    sub_pt=like.sub_division, dtype=dtype)
        self._names.append(name)
        return field

    def __exit__(self, *exc):
        collection = getattr(self._fc, "_get_field_collection", None)
        collection = collection() if collection is not None else self._fc
        for name in self._names:
            collection.pop_field(name)
        return False


#: Relative tolerance below which a float32 solve cannot certify convergence.
#: float32 eps is 1.19e-7, so the *true* residual ``b - Ax`` stagnates around
#: here even though the recursively updated CG residual keeps shrinking.
#: Asking for less is asking for something arithmetic cannot deliver.
FLOAT32_RTOL_FLOOR = 1e-6


def _warn_unreachable_rtol(dtype, rtol, solver):
    """Warn once if a single-precision solve is given a tolerance below the
    float32 accuracy floor."""
    if (np.dtype(dtype) == np.dtype(np.float32)
            and 0.0 < rtol < FLOAT32_RTOL_FLOOR):
        warnings.warn(
            f"{solver}: rtol={rtol:.1e} is below the float32 solve accuracy "
            f"floor (~{FLOAT32_RTOL_FLOOR:.0e}). The true residual b - Ax "
            "cannot reach it -- only the recursive residual can -- so the "
            "solve will run to maxiter. Use a looser rtol or a float64 "
            "right-hand side.",
            RuntimeWarning,
            stacklevel=3,
        )


def _reject_non_finite(value, what, iteration, dtype, solver):
    """Raise ConvergenceError if ``value`` is NaN or infinite.

    The two causes need different diagnoses, so they get different messages.
    An infinity here is almost always dynamic range rather than a modelling
    error: the reductions accumulate (and return) in double, so an infinite
    inner product means the *fields* themselves carry values whose squares
    leave the range of the work dtype.
    """
    if value != value:  # NaN
        raise ConvergenceError(
            f"{solver}: {what} became NaN at iteration {iteration}. Either the "
            "operator is not positive definite, or the input contains NaN, or "
            f"an intermediate field overflowed the {np.dtype(dtype).name} range "
            "and a subsequent inf - inf produced NaN. If the operator has a "
            "large norm, rescale it or use a float64 right-hand side."
        )
    if value in (float("inf"), float("-inf")):
        raise ConvergenceError(
            f"{solver}: {what} overflowed to {value} at iteration "
            f"{iteration}. The iterate has grown beyond the dynamic range of "
            f"the {np.dtype(dtype).name} work fields, which means the solve is "
            "diverging. Rescale the operator and right-hand side, improve the "
            "preconditioner, or use a float64 right-hand side."
        )


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
        residual=None,
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
    residual : muGrid.Field, optional
        If given, the final residual ``r = b - Ax`` is copied into this field
        on (converged) return -- CG maintains it anyway, so this costs one
        field copy instead of an extra operator apply. Used e.g. for
        adjoint-corrected (Lagrangian) objective evaluation with truncated
        solves.

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
    _warn_unreachable_rtol(dtype, rtol, "conjugate_gradients")

    with _WorkFields(fc, "cg") as work:
        with timed("startup"):
            # Create temporary fields with matching component shape
            # r: residual field
            # p: search direction field
            # z: preconditioned search direction field (aliased to r if no prec)
            # Ap: Hessian product field
            r = work.field("residual", b, dtype)
            p = work.field("search-direction", b, dtype)
            if unpreconditioned:
                z = r
            else:
                z = work.field("preconditioned-residual", b, dtype)
            Ap = work.field("hessian-product", b, dtype)

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
                if residual is not None:
                    linalg.copy(r, residual)
                return x

        with timed("iteration"):
            for iteration in range(maxiter):
                # Compute Hessian product: Ap = A * p
                with timed("hessp"):
                    hessp(p, Ap)

                # Compute pAp for step size
                with timed("dot_pAp"):
                    pAp = comm.sum(linalg.vecdot(p, Ap))

                # Guard the curvature term *before* dividing by it. A non-finite
                # pAp silently gives alpha = 0, which leaves x untouched and rr
                # unchanged for every remaining iteration -- a stall that looks
                # exactly like slow convergence. A zero pAp means p lies in the
                # operator's null space, so it is not positive definite.
                pAp_val = float(pAp)
                _reject_non_finite(pAp_val, "the curvature p^T A p", iteration,
                                   dtype, "conjugate_gradients")
                if pAp_val == 0.0:
                    raise ConvergenceError(
                        "conjugate_gradients: the curvature p^T A p is zero at "
                        f"iteration {iteration}; the search direction lies in the "
                        "null space of the operator, which is therefore not "
                        "positive definite."
                    )

                # Compute alpha
                alpha = rz / pAp

                # A zero step on a residual that has not converged cannot make
                # progress; with pAp already checked this means rz = 0, i.e. an
                # indefinite preconditioner.
                if alpha == 0.0:
                    raise ConvergenceError(
                        "conjugate_gradients: the step length is zero at "
                        f"iteration {iteration} while |r|^2 = {rr_val:.6e} is "
                        f"above the tolerance {tol_sq:.6e}; the preconditioned "
                        "residual is orthogonal to the residual, which means the "
                        "preconditioner is not positive definite."
                    )

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

                # NaN means a non-positive-definite operator; an infinity means
                # the iterate has outgrown the work fields' dynamic range.
                _reject_non_finite(next_rr_val, "the squared residual |r|^2",
                                   iteration, dtype, "conjugate_gradients")

                if next_rr_val <= tol_sq:
                    if residual is not None:
                        linalg.copy(r, residual)
                    return x

                # Compute beta
                beta = next_rz / rz
                # Carry the residual and its preconditioned inner product into the
                # next iteration (rr_val is what the guards above report).
                rz = next_rz
                rr_val = next_rr_val

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
        replacement_interval: int = None,
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

    The residual, preconditioned residual and operator images are advanced by
    recurrence rather than recomputed, and in this formulation the rounding
    errors of those recurrences are *amplified* rather than merely accumulated
    (Cools et al., SIAM J. Matrix Anal. Appl. 39 (2018) 426). Left alone, the
    recursive residual keeps shrinking while the true residual ``b - Ax``
    stalls orders of magnitude above it -- at condition number 1e6 and
    ``rtol=1e-8`` it stopped at a true relative residual of 5e-5 -- and in
    float32 it diverges outright. Two safeguards keep the result honest:

    - **Residual replacement** (Cools et al., 2018): every
      `replacement_interval` iterations the recurrence vectors are recomputed
      from their definitions (``r = b - Ax``, ``u = M⁻¹r``, ``w = Au``,
      ``s = Ap``, ``q = M⁻¹s``, ``z = Aq``), which resets the gap between the
      recursive and the true residual, at the cost of four operator and two
      preconditioner applies. If the gap had grown beyond a factor of two in
      the residual norm, the search direction is discarded as well and CG
      restarts from the true residual: keeping a direction built from badly
      drifted recurrences was seen to stall the solve.
    - **True-residual check**: convergence is only declared once ``b - Ax``
      itself meets the tolerance (one extra operator apply). If it does not,
      CG restarts from the true residual; if the true residual has not
      decreased since the previous such check, the solve has reached its
      attainable accuracy and a :class:`ConvergenceError` is raised rather
      than returning an unconverged solution.

    On diagonal test operators with condition numbers 1e4-1e7 (float64,
    ``rtol`` 1e-8-1e-10) this needs about 10% more operator applies than
    :func:`conjugate_gradients` and reaches the same true residual.

    Parameters are those of :func:`conjugate_gradients`, plus:

    replacement_interval : int, optional
        Iterations between residual replacements. The default depends on the
        precision: 50 for float64, 20 for float32, where the recurrences drift
        apart much faster (at 50, float32 solves at condition number 1e4
        failed to converge). ``0`` disables periodic replacement; the
        true-residual check on convergence is always performed.

    Raises
    ------
    ConvergenceError
        If the true residual does not reach the tolerance within `maxiter`
        iterations, stagnates above it, or a breakdown (NaN, overflow, zero
        curvature) occurs.
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
    _warn_unreachable_rtol(dtype, rtol, "conjugate_gradients_pipelined")
    if replacement_interval is None:
        replacement_interval = (
            20 if np.dtype(dtype) == np.dtype(np.float32) else 50)

    with _WorkFields(fc, "pcg") as work:
        with timed("startup"):
            # Work fields (zero-initialised by the collection)
            r = work.field("residual", b, dtype)
            u = work.field("prec-residual", b, dtype)
            w = work.field("w", b, dtype)
            m = work.field("m", b, dtype)
            n = work.field("n", b, dtype)
            p = work.field("p", b, dtype)
            s = work.field("s", b, dtype)
            q = work.field("q", b, dtype)
            z = work.field("z", b, dtype)

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

        def replace_residual(restart=None):
            """Recompute the recurrence vectors from their definitions and return
            ``(true_rr, restarted)``, the true squared residual norm |b - Ax|^2
            (global) and whether CG was restarted.

            A restart discards the search direction and its images, so the next
            iteration starts CG afresh from the current iterate. `restart=None`
            decides by the size of the drift: replacing the residual while keeping
            a search direction built from recurrences that have drifted far from
            the truth was seen to stall the solve (float32), so if the true
            residual norm is off from the recursive one by more than a factor of
            two, CG restarts instead.
            """
            if restart is None:
                recursive_rr = float(comm.sum(linalg.norm_sq(r)))
            hessp(x, r)
            linalg.axpby(1.0, b, -1.0, r)
            true_rr = float(comm.sum(linalg.norm_sq(r)))
            if restart is None:
                restart = not (0.25 * recursive_rr <= true_rr <= 4.0 * recursive_rr)
            with timed("prec"):
                prec(r, u)
            hessp(u, w)
            if restart:
                for f in (p, s, q, z):
                    linalg.scal(0.0, f)
            else:
                hessp(p, s)
                with timed("prec"):
                    prec(s, q)
                hessp(q, z)
            return true_rr, restart

        # True squared residual at the previous failed convergence check, to tell
        # a solve that is still converging from one that has stagnated at its
        # attainable accuracy.
        last_true_rr = None
        rr = None
        restarted = True  # the first iteration starts from p = 0

        with timed("iteration"):
            for iteration in range(maxiter):
                if (replacement_interval and iteration > 0
                        and iteration % replacement_interval == 0):
                    with timed("replace"):
                        _, restarted = replace_residual()

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

                # NaN means a non-positive-definite operator; an infinity means
                # the iterate has outgrown the work fields' dynamic range. Both
                # used to reach the recurrence below, where a non-finite delta
                # gives alpha = 0 and the *next* iteration then divides by that
                # zero alpha_prev -- a ZeroDivisionError raised from inside the
                # recurrence, escaping this solver's own ConvergenceError contract.
                _reject_non_finite(rr, "the squared residual |r|^2", iteration,
                                   dtype, "conjugate_gradients_pipelined")
                _reject_non_finite(delta, "the curvature w^T u", iteration, dtype,
                                   "conjugate_gradients_pipelined")
                _reject_non_finite(gamma, "the inner product r^T u", iteration,
                                   dtype, "conjugate_gradients_pipelined")
                if rr <= tol_sq:
                    # The recursive residual says converged; only the true one
                    # may decide. On failure CG restarts from the true residual:
                    # the old search direction was built from the drifted
                    # recurrences, and continuing along it was seen to stall
                    # (the m, n computed above belong to the stale w and are
                    # dropped).
                    with timed("replace"):
                        true_rr, _ = replace_residual(restart=True)
                    _reject_non_finite(true_rr, "the true squared residual "
                                       "|b - Ax|^2", iteration, dtype,
                                       "conjugate_gradients_pipelined")
                    if true_rr <= tol_sq:
                        return x
                    if last_true_rr is not None and true_rr >= last_true_rr:
                        raise ConvergenceError(
                            "conjugate_gradients_pipelined: the true residual "
                            f"|b - Ax| = {np.sqrt(true_rr):.6e} has stagnated "
                            f"above the tolerance {np.sqrt(tol_sq):.6e} at "
                            f"iteration {iteration} (the recursive residual is "
                            f"{np.sqrt(rr):.6e}); the requested accuracy is not "
                            f"attainable in {np.dtype(dtype).name} for this "
                            "system. Loosen rtol, use float64, or use "
                            "conjugate_gradients."
                        )
                    last_true_rr = true_rr
                    restarted = True
                    continue

                if restarted:
                    beta = 0.0
                    denom = delta
                else:
                    beta = gamma / gamma_prev
                    denom = delta - beta * gamma / alpha_prev
                restarted = False
                if denom == 0.0:
                    raise ConvergenceError(
                        "conjugate_gradients_pipelined: the curvature term is zero "
                        f"at iteration {iteration} while |r|^2 = {rr:.6e} is above "
                        f"the tolerance {tol_sq:.6e}; the operator or the "
                        "preconditioner is not positive definite."
                    )
                alpha = gamma / denom
                if alpha == 0.0:
                    raise ConvergenceError(
                        "conjugate_gradients_pipelined: the step length is zero at "
                        f"iteration {iteration} while |r|^2 = {rr:.6e} is above the "
                        f"tolerance {tol_sq:.6e}; the solve cannot make progress."
                    )
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
            "Pipelined preconditioned conjugate gradient algorithm did not converge "
            f"in {maxiter} iterations (recursive |r| = "
            f"{np.sqrt(rr) if rr is not None else float('nan'):.6e}, tolerance "
            f"{np.sqrt(tol_sq):.6e})"
        )
