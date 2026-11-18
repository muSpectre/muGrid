"""
Minimisers for nonlinear problems on muGrid fields.

The vectors of a minimiser are muGrid fields: they live where the fields of
the problem live (host or device), are distributed under MPI like them, and
are allocated once. Only scalars reach the host: an inner product is a local
reduction, summed over the processes with the communicator.

- :func:`l_bfgs_bounded` -- limited-memory BFGS for box constraints
  ``lower <= x <= upper``, with an optional preconditioner as the initial
  inverse Hessian.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from . import linalg
from .Field import Field
from .Solvers import _WorkFields


@dataclass
class OptimizeResult:
    """
    Outcome of a minimisation.

    Attributes
    ----------
    success : bool
        True if one of the convergence tests was met (or the callback asked
        to stop).
    message : str
        The test that ended the run.
    fun : float
        Objective at the returned iterate.
    nit : int
        Number of iterations.
    nfev : int
        Number of evaluations of the objective and its gradient.
    max_grad : float
        Largest absolute entry of the projected gradient at the returned
        iterate.
    x : muGrid.Field
        The solution: the field passed to the minimiser.
    gradient : muGrid.Field
        Gradient at the solution. It is a work field of the minimiser and
        valid until the next call on the same collection.
    """

    success: bool
    message: str
    fun: float
    nit: int
    nfev: int
    max_grad: float
    x: object
    gradient: object


class _KernelOperations:
    """The vector operations of the minimiser, by muGrid's linalg kernels:
    one pass per operation, no temporaries, host or device."""

    def __init__(self, comm):
        self.comm = comm

    def dot(self, a, b):
        return float(self.comm.sum(linalg.vecdot(a, b)))

    def norm_inf(self, a):
        return float(self.comm.max(linalg.norm_inf(a)))

    def copy(self, source, destination):
        linalg.copy(source, destination)

    def scale(self, alpha, x):
        linalg.scal(alpha, x)

    def axpy(self, alpha, x, y):
        """y += alpha x"""
        linalg.axpy(alpha, x, y)

    def step(self, alpha, d, x, lower, upper, out):
        """out = clip(x + alpha d)"""
        linalg.axpy_clip(alpha, d, x, lower, upper, out)

    def free_part(self, gradient, x, lower, upper, v, out):
        """out = v on the free variables, 0 on the bound-active ones"""
        linalg.mask_bound_active(gradient, x, lower, upper, v, out)


class _NumpyOperations(_KernelOperations):
    """The same operations written as the numpy expressions of NuMPI's
    ``l_bfgs_bounded``, on the interior of the fields.

    For checks only: with these the iterates are bitwise NuMPI's on one
    process, because every sum and every update rounds as NuMPI's does (the
    kernels use fused multiply-add and sum in another order). Each operation
    makes temporaries of the size of a field.
    """

    @staticmethod
    def _values(bound):
        return bound.s if isinstance(bound, Field) else bound

    def dot(self, a, b):
        # NuMPI sums the product of the flattened vectors
        return float(self.comm.sum(float(np.sum(a.s.ravel() * b.s.ravel()))))

    def norm_inf(self, a):
        return float(self.comm.max(float(np.max(np.abs(a.s)))))

    def copy(self, source, destination):
        destination.s[...] = source.s

    def scale(self, alpha, x):
        x.s[...] *= alpha

    def axpy(self, alpha, x, y):
        y.s[...] += alpha * x.s

    def step(self, alpha, d, x, lower, upper, out):
        out.s[...] = np.clip(x.s + alpha * d.s, self._values(lower),
                             self._values(upper))

    def free_part(self, gradient, x, lower, upper, v, out):
        tolerance = 1e-12
        active = (((x.s <= self._values(lower) + tolerance)
                   & (gradient.s >= 0.0))
                  | ((x.s >= self._values(upper) - tolerance)
                     & (gradient.s <= 0.0)))
        out.s[...] = np.where(active, 0.0, v.s)


def l_bfgs_bounded(
        comm,
        fc,
        fun: callable,
        x,
        lower=0.0,
        upper=1.0,
        precond: callable = None,
        maxcor: int = 20,
        gtol: float = 1e-5,
        xtol: float = 0.0,
        ftol: float = 0.0,
        maxiter: int = 500,
        c1: float = 1e-4,
        max_halvings: int = 40,
        callback: callable = None,
        timer=None,
        _arithmetic: str = "kernels",
):
    """
    Limited-memory BFGS for the box-constrained minimisation

    .. math::

        \\min_x f(x) \\quad \\text{subject to} \\quad
        \\mathrm{lower} \\le x \\le \\mathrm{upper} .

    **Free variables.** A variable is *bound-active* when it sits on a bound
    and the gradient pushes it out of the box; all others are *free*. The
    *projected gradient* :math:`r` is the gradient on the free variables and
    zero on the bound-active ones. It vanishes exactly at a constrained
    minimiser, and the method works with it in place of the gradient.

    **One iteration.**

    1. Search direction :math:`d = -H r`, where :math:`H` approximates the
       inverse Hessian from the last `maxcor` pairs
       :math:`s = x_{k+1} - x_k`, :math:`y = r_{k+1} - r_k` (two-loop
       recursion of Nocedal). If :math:`d` is not a descent direction, the
       pairs are dropped and :math:`d` is the steepest-descent direction.
    2. Line search along the projected path
       :math:`x(\\alpha) = \\mathrm{clip}(x + \\alpha d)`: start with
       :math:`\\alpha = 1` and halve until the Armijo condition
       :math:`f(x(\\alpha)) \\le f(x) + c_1 \\alpha\\, r^T d` holds. If it
       fails, the pairs are dropped and the search is repeated once along
       the steepest-descent direction, starting with the step
       :math:`1 / \\|d\\|`.
    3. The new pair is stored if its curvature is positive,
       :math:`s^T y > 10^{-12} \\|s\\| \\|y\\|`.

    **Preconditioner.** The two-loop recursion starts from an initial
    inverse Hessian :math:`H_0`. Without `precond` it is the scaled identity
    :math:`\\gamma I`, :math:`\\gamma = s^T y / y^T y` of the newest pair.
    With `precond`, a symmetric positive definite operator :math:`M`,

    .. math::

        H_0 = \\gamma\\, P M P, \\qquad
        \\gamma = \\frac{s^T y}{y^T P M P\\, y},

    where :math:`P` sets the bound-active variables to zero. A smoother is a
    good choice of :math:`M`, e.g. the Helmholtz filter
    :math:`(I - c\\,\\Delta)^{-1}` for the density of a topology
    optimisation, where it cut the number of evaluations by a factor 2 to 3.

    The algorithm, including the order of its stopping tests, is that of
    ``NuMPI.Optimization.l_bfgs_bounded``.

    **Memory.** ``2 * maxcor`` fields for the pairs and 8 work fields
    (10 with a preconditioner), all of the size of `x`, registered in `fc`
    under names starting with ``lbfgs-work`` and kept for the next call.
    With scalar bounds no field is used for the bounds.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for the global reductions.
    fc : muGrid.GlobalFieldCollection, muGrid.LocalFieldCollection, or
         muGrid.CartesianDecomposition
        Collection of `x`, which also holds the work fields.
    fun : callable
        ``fun(x, gradient)`` writes the gradient of the objective at the
        field `x` into the field `gradient` and returns the objective. The
        returned value must be the *global* one, the same on every process.
        `x` is one of two fields the minimiser alternates between, not
        always the field passed in.
        The line search compares objectives of nearby points, so
        evaluate the objective in double precision also when the fields are
        float32.
    x : muGrid.Field
        Start (clipped to the bounds), overwritten with the solution. A
        real field, float64 or float32.
    lower, upper : float, muGrid.Field or None, optional
        The bounds: scalars, fields with one bound per entry of `x`, or
        None for no bound on that side. The defaults are 0 and 1.
    precond : callable, optional
        ``precond(v, Mv)`` writes :math:`M v` into the field `Mv`, with
        :math:`M` symmetric positive definite (the convention of `prec` in
        :func:`muGrid.Solvers.conjugate_gradients`). Default: none.
    maxcor : int, optional
        Number of pairs kept. The default is 20.
    gtol : float, optional
        Stop when the largest absolute entry of the projected gradient is
        below `gtol`. The default is 1e-5.
        The line search accepts a step by the decrease of the objective,
        which close to the minimum is of the order of the squared gradient.
        A `gtol` whose square is below the round-off of the objective
        (about 1e-16 times its value in double precision) cannot be
        reached: the iteration stalls until `maxiter`.
    xtol : float, optional
        Stop when the largest change of a variable in one step is below
        `xtol`. Not tested on a step made without pairs: such a
        steepest-descent step has the length of the gradient, which says
        nothing about the distance to the minimiser. The default 0 disables
        the test.
    ftol : float, optional
        Stop when the objective changes by less than `ftol` times
        ``max(1, |f_new|, |f_old|)`` in one step. The default 0 disables the
        test.
    maxiter : int, optional
        Maximum number of iterations. The default is 500.
    c1 : float, optional
        Constant of the Armijo condition. The default is 1e-4.
    max_halvings : int, optional
        Maximal number of step halvings in one line search. The default is
        40.
    callback : callable, optional
        ``callback(iteration, state)`` is called after every accepted step;
        returning True stops the run. `state` is a dict with

        - "x": the current iterate (a field)
        - "gradient": its gradient (a field)
        - "fun": the objective
        - "max_grad": the largest absolute entry of the projected gradient
        - "step": the accepted step length :math:`\\alpha`
    timer : muGrid.Timer, optional
        Timer for the evaluations ("fun"), the preconditioner ("precond")
        and the vector work of the minimiser ("optimizer").

    Returns
    -------
    OptimizeResult
        See :class:`OptimizeResult`. Reaching `maxiter` or a failed line
        search is reported by ``success = False``, not by an exception: the
        iterate reached is the best one found and remains in `x`.
    """
    if _arithmetic == "kernels":
        ops = _KernelOperations(comm)
    elif _arithmetic == "numpy":
        ops = _NumpyOperations(comm)
    else:
        raise ValueError(f"unknown arithmetic '{_arithmetic}'")

    def timed(name):
        return timer(name) if timer is not None else nullcontext()

    dtype = getattr(x, "dtype", np.float64)
    nfev = 0

    with _WorkFields(fc, "lbfgs") as work:

        def new_field(role):
            return work.field(role, x, dtype)

        # ---- the bounds: two scalars, or two fields ---------------------
        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper
        if isinstance(lower, Field) != isinstance(upper, Field):
            # one bound per entry on one side only: the kernels take two
            # scalars or two fields, so the scalar side becomes a field
            if isinstance(lower, Field):
                scalar, upper = upper, new_field("upper-bound")
                upper.s[...] = scalar
            else:
                scalar, lower = lower, new_field("lower-bound")
                lower.s[...] = scalar

        # ---- the vectors, allocated once --------------------------------
        # Two sets of (iterate, gradient, projected gradient): the current
        # point and the trial point of the line search. An accepted trial
        # point becomes the current one by swapping the two sets.
        trial_x = new_field("iterate")
        gradient = new_field("gradient")
        trial_gradient = new_field("trial-gradient")
        projected = new_field("projected-gradient")
        trial_projected = new_field("trial-projected-gradient")
        direction = new_field("direction")
        if precond is not None:
            masked = new_field("masked")                # P v, the input of M
            scaling = new_field("scaling")              # P M P y, for gamma
        # The pairs, newest first; (s, y) fields that dropped out are reused
        pairs = []                                      # [(s, y, 1 / s.y)]
        spare_pairs = []
        nb_pair_fields = 0

        def pair_fields():
            nonlocal nb_pair_fields
            if spare_pairs:
                return spare_pairs.pop()
            nb_pair_fields += 1
            return (new_field(f"pair{nb_pair_fields}-s"),
                    new_field(f"pair{nb_pair_fields}-y"))

        def drop_pairs():
            spare_pairs.extend((s, y) for s, y, _ in pairs)
            pairs.clear()

        candidate = pair_fields()

        # ---- the pieces of one iteration --------------------------------
        def evaluate(point, gradient_out):
            nonlocal nfev
            nfev += 1
            with timed("fun"):
                return float(fun(point, gradient_out))

        def initial_inverse_hessian(v, out):
            """out = P M P v, with P for the current iterate. `out` may be
            `v`."""
            with timed("precond"):
                ops.free_part(gradient, x, lower, upper, v, masked)
                precond(masked, out)
                ops.free_part(gradient, x, lower, upper, out, out)

        def steepest_descent():
            """direction = -r, or -P M P r with a preconditioner"""
            if precond is None:
                ops.copy(projected, direction)
            else:
                initial_inverse_hessian(projected, direction)
            ops.scale(-1.0, direction)

        def quasi_newton_direction():
            """direction = -H r by the two-loop recursion (Nocedal &
            Wright, Numerical Optimization, Algorithm 7.4)"""
            ops.copy(projected, direction)
            # first loop, newest to oldest pair
            coefficients = []
            for s, y, rho in pairs:
                a = rho * ops.dot(s, direction)
                ops.axpy(-a, y, direction)
                coefficients.append(a)
            # initial inverse Hessian, scaled with the newest pair
            if precond is not None:
                if pairs:
                    s, y, _ = pairs[0]
                    initial_inverse_hessian(y, scaling)
                    gamma = ops.dot(s, y) / ops.dot(y, scaling)
                    initial_inverse_hessian(direction, direction)
                    ops.scale(gamma, direction)
                else:
                    initial_inverse_hessian(direction, direction)
            elif pairs:
                s, y, _ = pairs[0]
                ops.scale(ops.dot(s, y) / ops.dot(y, y), direction)
            # second loop, oldest to newest pair
            for (s, y, rho), a in zip(reversed(pairs),
                                      reversed(coefficients)):
                b = rho * ops.dot(y, direction)
                ops.axpy(a - b, s, direction)
            ops.scale(-1.0, direction)

        def line_search(objective, slope, alpha):
            """Halve alpha until the trial point clip(x + alpha d) meets the
            Armijo condition. Returns (objective, alpha) of the accepted
            trial point, or None."""
            for _ in range(max_halvings):
                ops.step(alpha, direction, x, lower, upper, trial_x)
                trial_objective = evaluate(trial_x, trial_gradient)
                if trial_objective <= objective + c1 * alpha * slope:
                    return trial_objective, alpha
                alpha *= 0.5
            return None

        def result(success, message):
            # the current iterate may live in the work field: hand it back
            # in the caller's field
            if x is not start_field:
                ops.copy(x, start_field)
            return OptimizeResult(
                success=success, message=message, fun=objective,
                nit=iteration, nfev=nfev, max_grad=max_grad,
                x=start_field, gradient=gradient)

        # ---- start: a feasible point and its projected gradient ---------
        start_field = x
        iteration = 0
        ops.step(0.0, x, x, lower, upper, x)            # x = clip(x)
        objective = evaluate(x, gradient)
        ops.free_part(gradient, x, lower, upper, gradient, projected)
        max_grad = ops.norm_inf(projected)
        if max_grad < gtol:
            return result(
                True, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")

        for iteration in range(1, maxiter + 1):
            with timed("optimizer"):
                quasi_newton_direction()
                slope = ops.dot(projected, direction)
                # a step without pairs is steepest descent; see `xtol`
                used_pairs = len(pairs) > 0
                if slope >= 0.0:
                    # outdated pairs gave a direction that does not descend
                    drop_pairs()
                    steepest_descent()
                    slope = ops.dot(projected, direction)

            accepted = line_search(objective, slope, 1.0)
            if accepted is None:
                # typically the set of free variables has just changed:
                # retry once along the steepest descent, with a step of
                # unit length
                drop_pairs()
                steepest_descent()
                slope = ops.dot(projected, direction)
                length = np.sqrt(ops.dot(direction, direction))
                accepted = line_search(objective, slope,
                                       1.0 / length if length > 0 else 1.0)
            if accepted is None:
                return result(
                    False, "CONVERGENCE: line-search did not converge")
            trial_objective, alpha = accepted

            with timed("optimizer"):
                ops.free_part(trial_gradient, trial_x, lower, upper,
                              trial_gradient, trial_projected)

                # the pair of this step: s = x_new - x, y = r_new - r
                s, y = candidate
                ops.copy(trial_x, s)
                ops.axpy(-1.0, x, s)
                ops.copy(trial_projected, y)
                ops.axpy(-1.0, projected, y)
                curvature = ops.dot(s, y)
                s_norm = np.sqrt(ops.dot(s, s))
                y_norm = np.sqrt(ops.dot(y, y))
                largest_change = ops.norm_inf(s) if (
                    xtol > 0 and used_pairs) else np.inf
                # keep the pair only if its curvature is safely positive;
                # otherwise the approximation H would lose definiteness
                if curvature > 1e-12 * (s_norm * y_norm
                                        + np.finfo(float).eps):
                    pairs.insert(0, (s, y, 1.0 / curvature))
                    if len(pairs) > maxcor:
                        oldest_s, oldest_y, _ = pairs.pop()
                        spare_pairs.append((oldest_s, oldest_y))
                    candidate = pair_fields()

            # The two step-size tests return the point *before* the step,
            # as NuMPI does.
            if largest_change < xtol:
                return result(
                    True, "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL")
            if ftol > 0 and abs(trial_objective - objective) < ftol * max(
                    1, abs(trial_objective), abs(objective)):
                return result(
                    True, "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL")

            # accept the trial point: swap the two sets of vectors
            x, trial_x = trial_x, x
            gradient, trial_gradient = trial_gradient, gradient
            projected, trial_projected = trial_projected, projected
            objective = trial_objective
            max_grad = ops.norm_inf(projected)

            if callback is not None:
                with timed("callback"):
                    stop = callback(iteration, {
                        "x": x, "gradient": gradient, "fun": objective,
                        "max_grad": max_grad, "step": alpha})
                if stop is True:
                    return result(True, "CONVERGENCE: STOPPED_BY_CALLBACK")
            if max_grad < gtol:
                return result(
                    True, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")

        return result(False, "NO CONVERGENCE: MAXITERATIONS REACHED")
