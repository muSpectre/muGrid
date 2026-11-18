"""
Tests for muGrid.Optimize.l_bfgs_bounded and the linalg kernels it uses
(norm_inf, axpy_clip, mask_bound_active).
"""

import numpy as np
import pytest
from conftest import (
    create_device,
    get_array_module,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid
from muGrid import Communicator, GlobalFieldCollection, linalg
from muGrid.Optimize import l_bfgs_bounded

serial_only = pytest.mark.skipif(
    Communicator().size > 1, reason="Test only works in serial"
)


def to_host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


class KernelSetup:
    """Fields with ghost layers, filled with random numbers everywhere --
    the ghosts too, with large values that a reduction must not see."""

    def __init__(self, comm, device, dtype, nb_components=2):
        self.xp = get_array_module(device)
        self.dtype = dtype
        self.decomposition = muGrid.CartesianDecomposition(
            comm, (8, 6), nb_ghosts_left=(1, 2), nb_ghosts_right=(2, 1),
            device=create_device(device))
        self.nb_components = nb_components
        self.rng = np.random.default_rng(comm.rank)

    def field(self, name, low=-1.0, high=1.0):
        field = self.decomposition.real_field(
            name, (self.nb_components,), dtype=self.dtype)
        field.sg[...] = 1e6                       # ghosts (and interior)
        field.s[...] = self.xp.asarray(
            self.rng.uniform(low, high, field.s.shape), dtype=self.dtype)
        return field


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_norm_inf_ignores_ghosts(comm, device, dtype):
    skip_if_gpu_unavailable(device)
    setup = KernelSetup(comm, device, dtype)
    x = setup.field("x")
    expected = float(abs(to_host(x.s)).max())
    assert linalg.norm_inf(x) == expected
    # a negative entry of largest magnitude
    x.s[0, 0, 1, 1] = -7.5
    assert linalg.norm_inf(x) == 7.5


@pytest.mark.parametrize("device", get_test_devices())
def test_norm_inf_without_ghosts(device):
    skip_if_gpu_unavailable(device)
    kwargs = {} if device == "cpu" else {"device": create_device(device)}
    fc = GlobalFieldCollection((5, 7), **kwargs)
    x = fc.real_field("x", (3,))
    xp = get_array_module(device)
    values = np.random.default_rng(0).standard_normal(x.s.shape)
    x.s[...] = xp.asarray(values)
    assert linalg.norm_inf(x) == abs(values).max()
    x.set_zero()
    assert linalg.norm_inf(x) == 0.0


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("bounds", ["scalars", "fields"])
def test_axpy_clip(comm, device, dtype, bounds):
    """out = clip(x + alpha d), with scalar bounds and with bound fields"""
    skip_if_gpu_unavailable(device)
    setup = KernelSetup(comm, device, dtype)
    x, d, out = setup.field("x"), setup.field("d"), setup.field("out")
    if bounds == "scalars":
        lower, upper = -0.3, 0.4
        lower_values, upper_values = lower, upper
    else:
        lower = setup.field("lower", -0.8, -0.1)
        upper = setup.field("upper", 0.1, 0.8)
        lower_values, upper_values = to_host(lower.s), to_host(upper.s)
    alpha = 0.7
    expected = np.clip(
        to_host(x.s) + dtype(alpha) * to_host(d.s), lower_values, upper_values)
    linalg.axpy_clip(alpha, d, x, lower, upper, out)
    # the kernel may fuse the multiplication and the addition, which
    # changes the last bits of a sum of order one
    np.testing.assert_allclose(to_host(out.s), expected,
                               rtol=0, atol=4 * np.finfo(dtype).eps)
    # the bounds themselves are met exactly, in the precision of the field
    assert to_host(out.s).min() >= np.min(np.asarray(lower_values, dtype))
    assert to_host(out.s).max() <= np.max(np.asarray(upper_values, dtype))
    # in place, on x
    linalg.axpy_clip(alpha, d, x, lower, upper, x)
    np.testing.assert_array_equal(to_host(x.s), to_host(out.s))


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("bounds", ["scalars", "fields"])
def test_mask_bound_active(comm, device, dtype, bounds):
    """out = v on the free variables, 0 where x sits on a bound and the
    gradient pushes it out of the box"""
    skip_if_gpu_unavailable(device)
    setup = KernelSetup(comm, device, dtype)
    xp = setup.xp
    x, g = setup.field("x", -0.5, 1.5), setup.field("g")
    v, out = setup.field("v"), setup.field("out")
    if bounds == "scalars":
        lower, upper = 0.0, 1.0
        lower_values = np.zeros(x.s.shape, dtype)
        upper_values = np.ones(x.s.shape, dtype)
    else:
        lower = setup.field("lower", -0.2, 0.2)
        upper = setup.field("upper", 0.8, 1.2)
        lower_values, upper_values = to_host(lower.s), to_host(upper.s)
    # put about half of the variables onto their bounds
    x.s[...] = xp.asarray(
        np.clip(to_host(x.s), lower_values, upper_values))
    x_values, g_values = to_host(x.s), to_host(g.s)
    active = ((x_values <= lower_values) & (g_values >= 0)) | (
        (x_values >= upper_values) & (g_values <= 0))
    assert 0 < active.sum() < active.size
    expected = np.where(active, 0, to_host(v.s))

    linalg.mask_bound_active(g, x, lower, upper, v, out)
    np.testing.assert_array_equal(to_host(out.s), expected)
    # in place, on v
    linalg.mask_bound_active(g, x, lower, upper, v, v)
    np.testing.assert_array_equal(to_host(v.s), expected)


def test_infinite_bounds_leave_everything_free():
    fc = GlobalFieldCollection((16,))
    x, g, out = (fc.real_field(name) for name in "xgo")
    x.s[...] = np.linspace(-5, 5, 16)
    g.s[...] = 1.0
    linalg.mask_bound_active(g, x, -np.inf, np.inf, g, out)
    np.testing.assert_array_equal(out.s, g.s)
    linalg.axpy_clip(2.0, g, x, -np.inf, np.inf, out)
    np.testing.assert_array_equal(out.s, x.s + 2.0)


def test_kernels_reject_fields_of_another_collection():
    a = GlobalFieldCollection((8,)).real_field("a")
    b = GlobalFieldCollection((8,)).real_field("b")
    with pytest.raises(RuntimeError, match="same collection"):
        linalg.axpy_clip(1.0, a, b, 0.0, 1.0, b)
    with pytest.raises(RuntimeError, match="same collection"):
        linalg.mask_bound_active(a, b, 0.0, 1.0, b, b)


# ---------------------------------------------------------------------------
# Reference: NuMPI's l_bfgs_bounded with `precond`, transcribed for plain
# numpy vectors (NuMPI is not a dependency of muGrid). The cross-check
# script benchmarks/lbfgs/cross_check_numpi.py compares with NuMPI itself.
# ---------------------------------------------------------------------------


def reference_l_bfgs(fun_grad, x0, lower=None, upper=None, precond=None,
                     maxcor=20, gtol=1e-5, xtol=0.0, ftol=0.0, maxiter=500,
                     c1=1e-4, max_halvings=40):
    """Returns x, objective, iterations, evaluations, message."""
    lo = -np.inf if lower is None else lower
    hi = np.inf if upper is None else upper
    nfev = [0]

    def evaluate(x):
        nfev[0] += 1
        return fun_grad(x)

    def active(grad, x):
        return ((x <= lo + 1e-12) & (grad >= 0)) | (
            (x >= hi - 1e-12) & (grad <= 0))

    def H0(v, act):
        return np.where(act, 0.0, precond(np.where(act, 0.0, v)))

    def two_loop(r, S, Y, RHO, act):
        q = r.copy()
        a = np.empty(len(S))
        for k in range(len(S)):
            a[k] = RHO[k] * np.sum(S[k] * q)
            q -= a[k] * Y[k]
        if precond is not None:
            if S:
                gamma = np.sum(S[0] * Y[0]) / np.sum(Y[0] * H0(Y[0], act))
                q = gamma * H0(q, act)
            else:
                q = H0(q, act)
        elif S:
            q *= np.sum(S[0] * Y[0]) / np.sum(Y[0] * Y[0])
        for k in range(len(S) - 1, -1, -1):
            q += (a[k] - RHO[k] * np.sum(Y[k] * q)) * S[k]
        return -q

    def armijo(x, d, phi, gd, alpha):
        for _ in range(max_halvings):
            x_try = np.clip(x + alpha * d, lo, hi)
            phi_try, grad_try = evaluate(x_try)
            if phi_try <= phi + c1 * alpha * gd:
                return x_try, phi_try, grad_try
            alpha *= 0.5
        return None

    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    phi, grad = evaluate(x)
    r = np.where(active(grad, x), 0.0, grad)
    S, Y, RHO = [], [], []
    converged = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL"
    if np.max(np.abs(r)) < gtol:
        return x, phi, 0, nfev[0], converged
    for iteration in range(1, maxiter + 1):
        act = active(grad, x)
        d = two_loop(r, S, Y, RHO, act)
        gd = np.sum(r * d)
        used_pairs = len(S) > 0

        def steepest():
            S.clear(), Y.clear(), RHO.clear()
            return -r.copy() if precond is None else -H0(r, act)

        if gd >= 0.0:
            d = steepest()
            gd = np.sum(r * d)
        step = armijo(x, d, phi, gd, 1.0)
        if step is None:
            d = steepest()
            gd = np.sum(r * d)
            length = np.sqrt(np.sum(d * d))
            step = armijo(x, d, phi, gd, 1.0 / length if length > 0 else 1.0)
        if step is None:
            return (x, phi, iteration, nfev[0],
                    "CONVERGENCE: line-search did not converge")
        x_new, phi_new, grad_new = step
        r_new = np.where(active(grad_new, x_new), 0.0, grad_new)
        s, y = x_new - x, r_new - r
        sy = np.sum(s * y)
        if sy > 1e-12 * (np.sqrt(np.sum(s * s)) * np.sqrt(np.sum(y * y))
                         + np.finfo(float).eps):
            S.insert(0, s), Y.insert(0, y), RHO.insert(0, 1.0 / sy)
            if len(S) > maxcor:
                S.pop(), Y.pop(), RHO.pop()
        if xtol > 0 and used_pairs and np.max(np.abs(s)) < xtol:
            return (x, phi, iteration, nfev[0],
                    "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL")
        if ftol > 0 and abs(phi_new - phi) < ftol * max(
                1, abs(phi_new), abs(phi)):
            return (x, phi, iteration, nfev[0],
                    "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL")
        x, phi, grad, r = x_new, phi_new, grad_new, r_new
        if np.max(np.abs(r)) < gtol:
            return x, phi, iteration, nfev[0], converged
    return x, phi, maxiter, nfev[0], "NO CONVERGENCE: MAXITERATIONS REACHED"


# ---------------------------------------------------------------------------
# Problems on plain vectors (those of NuMPI's tests) and their field form
# ---------------------------------------------------------------------------


def quadratic(n=64, seed=0):
    y = np.random.default_rng(seed).normal(size=n)
    return lambda x: (0.5 * float(np.sum((x - y) ** 2)), x - y), y


def weighted_quadratic(n=200, seed=2):
    """0.5 sum w_i (x_i - y_i)^2 with weights over three decades"""
    rng = np.random.default_rng(seed)
    y, w = rng.normal(size=n) * 0.6 + 0.3, 10.0 ** rng.uniform(-3, 0, n)

    def fun_grad(x):
        d = x - y
        return 0.5 * float(np.sum(w * d * d)), w * d

    return fun_grad, y


def laplacian(n=256, a=1e-3, seed=10):
    """0.5 x^T (a I + K) x - b^T x, K the periodic 1D Laplacian stencil,
    and FFT Helmholtz operators (shift + K)^-1."""
    b = np.random.default_rng(seed).normal(size=n)
    K_hat = 2 - 2 * np.cos(2 * np.pi * np.fft.fftfreq(n))

    def fun_grad(x):
        Ax = a * x + 2 * x - np.roll(x, 1) - np.roll(x, -1)
        return 0.5 * float(x @ Ax) - float(b @ x), Ax - b

    def helmholtz(shift):
        return lambda v: np.real(np.fft.ifft(np.fft.fft(v) / (shift + K_hat)))

    return fun_grad, helmholtz


def many_active_bounds(n=200):
    """Laplacian-type quadratic whose minimiser in [0, 1] has most
    components on a bound, and a smoother (I + K)^-1."""
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1) + 1e-2 * np.eye(n)
    M = np.linalg.inv(np.eye(n) + A - 1e-2 * np.eye(n))
    b = np.random.default_rng(0).normal(size=n)

    def fun_grad(x):
        return 0.5 * float(x @ A @ x) - float(b @ x), A @ x - b

    return fun_grad, lambda v: M @ v


def solve_on_fields(fun_grad, x0, lower=None, upper=None, precond=None,
                    arithmetic="kernels", device="cpu", **options):
    """Solve a problem given on numpy vectors with l_bfgs_bounded: the
    vectors become scalar fields of a 1D collection."""
    xp = get_array_module(device)
    kwargs = {} if device == "cpu" else {"device": create_device(device)}
    fc = GlobalFieldCollection((x0.size,), **kwargs)
    x = fc.real_field("x")
    x.s[...] = xp.asarray(x0.reshape(x.s.shape))

    def fun(point, gradient):
        f, g = fun_grad(to_host(point.s).ravel())
        gradient.s[...] = xp.asarray(g.reshape(gradient.s.shape))
        return f

    def bound(name, value):
        if not isinstance(value, np.ndarray):
            return value
        field = fc.real_field(name)
        field.s[...] = xp.asarray(value.reshape(field.s.shape))
        return field

    def field_precond(v, out):
        out.s[...] = xp.asarray(
            precond(to_host(v.s).ravel()).reshape(out.s.shape))

    result = l_bfgs_bounded(
        Communicator(), fc, fun, x, lower=bound("lower", lower),
        upper=bound("upper", upper),
        precond=None if precond is None else field_precond,
        _arithmetic=arithmetic, **options)
    return result, to_host(x.s).ravel().copy(), fc


def problem_list():
    """name -> (fun_grad, x0, keyword arguments)"""
    problems = {}
    fun_grad, _ = quadratic()
    problems["quadratic"] = (fun_grad, np.zeros(64), dict(gtol=1e-10))
    fun_grad, _ = quadratic(128, 1)
    problems["quadratic-box"] = (
        fun_grad, np.full(128, 0.5), dict(lower=0.0, upper=1.0, gtol=1e-10))
    fun_grad, _ = weighted_quadratic()
    problems["weighted-box"] = (
        fun_grad, np.full(200, 0.5),
        dict(lower=0.0, upper=1.0, gtol=1e-8, maxiter=60))
    rng = np.random.default_rng(3)
    problems["weighted-array-bounds"] = (
        fun_grad, np.full(200, 0.3),
        dict(lower=rng.uniform(-0.5, 0.2, 200),
             upper=rng.uniform(0.4, 1.2, 200), gtol=1e-8, maxiter=60))
    problems["weighted-one-sided"] = (
        fun_grad, np.full(200, 0.3), dict(lower=0.0, gtol=1e-8, maxiter=60))
    fun_grad, helmholtz = laplacian()
    problems["laplacian"] = (
        fun_grad, np.zeros(256), dict(gtol=1e-4, maxiter=5000))
    problems["laplacian-exact-helmholtz"] = (
        fun_grad, np.zeros(256), dict(gtol=1e-8, precond=helmholtz(1e-3)))
    problems["laplacian-helmholtz-smoother"] = (
        fun_grad, np.zeros(256), dict(gtol=1e-6, precond=helmholtz(0.05)))
    problems["laplacian-box-helmholtz-smoother"] = (
        fun_grad, np.zeros(256),
        dict(lower=-5.0, upper=5.0, gtol=1e-6, maxiter=2000,
             precond=helmholtz(0.05)))
    fun_grad, smoother = many_active_bounds()
    # gtol 1e-5: at 1e-6 the decrease the line search asks for (about
    # gtol^2) is below the round-off of the objective (about 1e-14), and
    # whether the iteration stalls then depends on the arithmetic
    box = dict(lower=0.0, upper=1.0, gtol=1e-5, maxiter=3000)
    problems["many-active"] = (fun_grad, np.full(200, 0.5), box)
    problems["many-active-smoother"] = (
        fun_grad, np.full(200, 0.5), dict(precond=smoother, **box))
    problems["many-active-xtol"] = (
        fun_grad, np.full(200, 0.5),
        dict(lower=0.0, upper=1.0, gtol=0.0, xtol=1e-7))
    problems["many-active-ftol"] = (
        fun_grad, np.full(200, 0.5),
        dict(lower=0.0, upper=1.0, gtol=0.0, ftol=1e-9))
    return problems


PROBLEMS = problem_list()


# ---------------------------------------------------------------------------
# The optimiser against the reference
# ---------------------------------------------------------------------------


@serial_only
@pytest.mark.parametrize("name", PROBLEMS)
def test_check_arithmetic_is_bitwise_the_reference(name):
    """With numpy arithmetic every iterate equals the reference's bit for
    bit: same solution, objective, iterations, evaluations and message."""
    fun_grad, x0, options = PROBLEMS[name]
    x_ref, f_ref, nit_ref, nfev_ref, message_ref = reference_l_bfgs(
        fun_grad, x0, **options)
    result, x, _ = solve_on_fields(fun_grad, x0, arithmetic="numpy", **options)
    np.testing.assert_array_equal(x, x_ref)
    assert result.fun == f_ref
    assert (result.nit, result.nfev) == (nit_ref, nfev_ref)
    assert result.message == message_ref
    assert result.success == message_ref.startswith("CONVERGENCE: NORM") or (
        result.success == message_ref.startswith("CONVERGENCE: REL"))


# Problems on which round-off does not change a decision of the line search:
# there the kernels must follow the reference iteration by iteration.
STABLE = ["quadratic", "quadratic-box", "laplacian-exact-helmholtz",
          "laplacian-helmholtz-smoother"]


@serial_only
@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("name", STABLE)
def test_kernels_follow_the_reference(name, device):
    skip_if_gpu_unavailable(device)
    fun_grad, x0, options = PROBLEMS[name]
    x_ref, f_ref, nit_ref, nfev_ref, message_ref = reference_l_bfgs(
        fun_grad, x0, **options)
    result, x, _ = solve_on_fields(fun_grad, x0, device=device, **options)
    assert (result.nit, result.nfev) == (nit_ref, nfev_ref)
    assert result.message == message_ref
    np.testing.assert_allclose(x, x_ref, atol=1e-10 * abs(x_ref).max())
    assert result.fun == pytest.approx(f_ref, rel=1e-12, abs=1e-14)


# Ill-conditioned problems with many changes of the active set: a rounding
# difference flips a line-search decision sooner or later, and the runs
# take different paths to the same minimum.
SENSITIVE = ["laplacian", "laplacian-box-helmholtz-smoother", "many-active",
             "many-active-smoother", "many-active-xtol", "many-active-ftol"]


@serial_only
@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("name", SENSITIVE)
def test_kernels_reach_the_same_minimum(name, device):
    skip_if_gpu_unavailable(device)
    fun_grad, x0, options = PROBLEMS[name]
    x_ref, f_ref, nit_ref, _, message_ref = reference_l_bfgs(
        fun_grad, x0, **options)
    result, x, _ = solve_on_fields(fun_grad, x0, device=device, **options)
    assert result.success and result.message == message_ref
    if options["gtol"] > 0:
        assert result.max_grad < options["gtol"]
        assert result.fun == pytest.approx(f_ref, rel=1e-9)
    else:
        # stopped by xtol or ftol, which are loose tests
        assert result.fun == pytest.approx(f_ref, rel=1e-5)
    # a comparable effort
    assert result.nit < 2 * nit_ref + 10


@serial_only
def test_solution_on_the_box_is_the_clipped_target():
    fun_grad, y = quadratic(128, 1)
    result, x, _ = solve_on_fields(
        fun_grad, np.full(128, 0.5), lower=0.0, upper=1.0, gtol=1e-10)
    assert result.success
    np.testing.assert_allclose(x, np.clip(y, 0.0, 1.0), atol=1e-8)


@serial_only
def test_start_outside_the_box_is_projected():
    fun_grad, y = quadratic(64)
    result, x, _ = solve_on_fields(
        fun_grad, np.full(64, 7.0), lower=-0.5, upper=0.5, gtol=1e-10)
    assert result.success
    np.testing.assert_allclose(x, np.clip(y, -0.5, 0.5), atol=1e-8)


@serial_only
def test_preconditioner_reduces_the_evaluations():
    # gtol 1e-5: its square stays well above the round-off of the objective
    # (|f| ~ 7e3), so both runs can reach it on any platform; with 1e-6 the
    # unpreconditioned run stalled at maxiter on some CPUs
    fun_grad, helmholtz = laplacian()
    plain, _, _ = solve_on_fields(
        fun_grad, np.zeros(256), gtol=1e-5, maxiter=5000)
    smoothed, _, _ = solve_on_fields(
        fun_grad, np.zeros(256), gtol=1e-5, precond=helmholtz(0.05))
    assert plain.success and smoothed.success
    assert smoothed.nfev < plain.nfev / 5


# ---------------------------------------------------------------------------
# Stopping and bookkeeping
# ---------------------------------------------------------------------------


@serial_only
def test_already_converged_at_start():
    fun_grad, y = quadratic()
    result, x, _ = solve_on_fields(fun_grad, y.copy(), gtol=1e-8)
    assert result.success and result.nit == 0 and result.nfev == 1
    np.testing.assert_array_equal(x, y)


@serial_only
def test_maxiter_is_reported_not_raised():
    fun_grad, _ = weighted_quadratic()
    result, _, _ = solve_on_fields(
        fun_grad, np.full(200, 0.5), lower=0.0, upper=1.0, gtol=1e-12,
        maxiter=3)
    assert not result.success
    assert result.nit == 3
    assert "MAXITERATIONS" in result.message


@serial_only
def test_failed_line_search_is_reported():
    """An objective that grows in every direction the gradient offers."""
    def fun_grad(x):
        return float(np.sum(x)), -np.ones_like(x)      # wrong sign

    result, _, _ = solve_on_fields(fun_grad, np.zeros(8), lower=None,
                                   upper=None, max_halvings=5)
    assert not result.success
    assert "line-search" in result.message


@serial_only
def test_callback_sees_every_step_and_can_stop():
    fun_grad, _ = weighted_quadratic()
    seen = []

    def callback(iteration, state):
        seen.append((iteration, state["fun"], state["max_grad"]))
        assert state["step"] > 0
        return iteration == 4

    result, _, _ = solve_on_fields(
        fun_grad, np.full(200, 0.5), lower=0.0, upper=1.0, gtol=1e-12,
        callback=callback)
    assert result.success and "CALLBACK" in result.message
    assert [it for it, _, _ in seen] == [1, 2, 3, 4]
    # the objective decreases monotonically
    assert all(a[1] > b[1] for a, b in zip(seen, seen[1:]))
    assert result.fun == seen[-1][1] and result.max_grad == seen[-1][2]


@serial_only
@pytest.mark.parametrize("maxiter", [1, 2, 3, 4])
def test_solution_is_in_the_callers_field(maxiter):
    """The iterate alternates between two fields; whichever holds the last
    one, the caller's field has the solution and the result's gradient is
    the gradient there."""
    fun_grad, _ = weighted_quadratic()
    result, x, _ = solve_on_fields(
        fun_grad, np.full(200, 0.5), lower=0.0, upper=1.0, gtol=1e-12,
        maxiter=maxiter)
    assert result.nit == maxiter
    objective, gradient = fun_grad(x)
    assert result.fun == objective
    np.testing.assert_array_equal(result.gradient.s.ravel(), gradient)
    np.testing.assert_array_equal(result.x.s.ravel(), x)


@serial_only
def test_memory_is_fixed_and_scalar_bounds_need_no_fields():
    """Work fields: 8 + 2 per stored pair (2 more with a preconditioner),
    the same after a second solve; none for scalar bounds."""
    fun_grad, helmholtz = laplacian()
    maxcor = 5

    def names(fc):
        return {name for name in fc.field_names
                if name.startswith("lbfgs-work")}

    _, _, fc = solve_on_fields(
        fun_grad, np.zeros(256), lower=-5.0, upper=5.0, gtol=1e-6,
        maxiter=40, maxcor=maxcor)
    # the candidate pair of the next step is one more pair of fields
    assert len(names(fc)) == 6 + 2 * (maxcor + 1)
    assert not any("bound" in name for name in names(fc))

    _, _, fc = solve_on_fields(
        fun_grad, np.zeros(256), lower=-5.0, upper=5.0, gtol=1e-6,
        maxiter=40, maxcor=maxcor, precond=helmholtz(0.05))
    assert len(names(fc)) == 8 + 2 * (maxcor + 1)

    # one bound as a field: the scalar one becomes a field too
    _, _, fc = solve_on_fields(
        fun_grad, np.zeros(256), lower=np.full(256, -5.0), upper=5.0,
        gtol=1e-6, maxiter=5, maxcor=maxcor)
    assert sum("bound" in name for name in names(fc)) == 1


# ---------------------------------------------------------------------------
# On a grid: MPI, GPU, single precision
# ---------------------------------------------------------------------------

NB_GRID_PTS = (16, 12)


def grid_problem():
    """Obstacle-type problem on a periodic grid, on global arrays:
    0.5 u^T (I - Laplace) u - b^T u with 0 <= u <= 0.5, and the smoother
    (I - Laplace)^-1 scaled badly on purpose (so it is a preconditioner,
    not the inverse)."""
    rng = np.random.default_rng(7)
    b = rng.standard_normal(NB_GRID_PTS)

    def operator(u):
        return u + sum(2 * u - np.roll(u, 1, axis) - np.roll(u, -1, axis)
                       for axis in range(2))

    def fun_grad(x):
        u = x.reshape(NB_GRID_PTS)
        Au = operator(u)
        return (0.5 * float(np.sum(u * Au)) - float(np.sum(b * u)),
                (Au - b).ravel())

    return fun_grad, operator, b


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_grid_problem_matches_serial_reference(comm, device, dtype,
                                               preconditioned):
    """A distributed solve on fields with ghosts, on the host or the
    device, finds the minimiser of the serial reference on the whole grid.
    The preconditioner is a Jacobi scaling."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    fun_grad, _, b_global = grid_problem()
    # The line search needs a decrease of about gtol^2 to be visible in
    # the objective (about -8 here): 1e-15 in double, so gtol 1e-6. In
    # single the gradient itself carries a noise of about 1e-6.
    gtol = 1e-6 if dtype == np.float64 else 1e-4

    laplace = muGrid.LaplaceOperator(2, -1.0)
    decomposition = muGrid.CartesianDecomposition(
        comm, NB_GRID_PTS, ghosts=laplace, device=create_device(device))
    location = decomposition.subdomain_locations
    size = decomposition.nb_subdomain_grid_pts
    local = tuple(slice(o, o + n) for o, n in zip(location, size))
    b = xp.asarray(b_global[local], dtype=dtype)
    x = decomposition.real_field("x", dtype=dtype)
    Ax = decomposition.real_field("Ax", dtype=dtype)

    def fun(point, gradient):
        decomposition.communicate_ghosts(point)
        laplace.apply(point, Ax)                     # -Laplace u
        gradient.p[...] = point.p + Ax.p - b
        # The objective is evaluated in double also for float32 fields,
        # operator product included: the line search compares objectives of
        # nearby points, and a float32 product carries a noise (1e-6 here)
        # far above their difference. With ghosts: one layer on each side.
        u = point.pg.astype(np.float64).reshape(
            [n + 2 for n in size])
        centre = u[1:-1, 1:-1]
        Au = (5 * centre - u[:-2, 1:-1] - u[2:, 1:-1]
              - u[1:-1, :-2] - u[1:-1, 2:])
        energy_density = 0.5 * centre * Au - b * centre
        return comm.sum(float(energy_density.sum()))

    def jacobi(v, out):
        out.p[...] = v.p / 5.0                       # diagonal of I - Laplace

    result = l_bfgs_bounded(
        comm, decomposition, fun, x, lower=0.0, upper=0.5,
        precond=jacobi if preconditioned else None, gtol=gtol)
    assert result.success, result.message

    x_ref, f_ref, _, _, _ = reference_l_bfgs(
        fun_grad, np.zeros(b_global.size), lower=0.0, upper=0.5,
        precond=(lambda v: v / 5.0) if preconditioned else None, gtol=1e-10)
    # the smallest eigenvalue of the operator is 1, so the error of x is
    # at most the gradient tolerance, and that of the objective its square
    tolerance = 1e-5 if dtype == np.float64 else 1e-3
    np.testing.assert_allclose(
        to_host(x.p), x_ref.reshape(NB_GRID_PTS)[local], atol=tolerance)
    assert result.fun == pytest.approx(
        f_ref, rel=1e-10 if dtype == np.float64 else 1e-6)
    assert float(to_host(x.p).min()) >= 0.0
    assert float(to_host(x.p).max()) <= 0.5


# ---------------------------------------------------------------------------
# Pairs in host memory
# ---------------------------------------------------------------------------

# Every path through the pair memory: plain and preconditioned two-loop
# recursion, pairs dropped on a bad direction or a failed line search (many
# active bounds), and maxcor overflow (maxcor 3 over hundreds of iterations).
HOST_PAIR_PROBLEMS = ["quadratic-box", "laplacian-helmholtz-smoother",
                      "many-active", "many-active-smoother", "many-active-xtol"]


@serial_only
@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("maxcor", [3, 20])
@pytest.mark.parametrize("name", HOST_PAIR_PROBLEMS)
def test_host_pairs_give_the_iterates_of_device_pairs(name, maxcor, device):
    """The host copies of the pairs are exact: same solution bit for bit,
    same objective, iterations, evaluations and message. On the CPU the
    two stores are the same thing."""
    skip_if_gpu_unavailable(device)
    fun_grad, x0, options = PROBLEMS[name]
    options = dict(options, maxcor=maxcor)
    on_device, x_device, _ = solve_on_fields(
        fun_grad, x0, device=device, **options)
    on_host, x_host, _ = solve_on_fields(
        fun_grad, x0, device=device, pair_store="host", **options)
    np.testing.assert_array_equal(x_host, x_device)
    assert on_host.fun == on_device.fun
    assert (on_host.nit, on_host.nfev) == (on_device.nit, on_device.nfev)
    assert on_host.message == on_device.message


@serial_only
@pytest.mark.parametrize("device", get_test_devices())
def test_host_pairs_keep_one_pair_of_fields(device):
    """With the pairs on the host a GPU run has the work fields and one pair
    of fields (the candidate, which also stages the stored pairs); on the
    CPU the pairs stay fields."""
    skip_if_gpu_unavailable(device)
    fun_grad, helmholtz = laplacian()
    maxcor = 5
    _, _, fc = solve_on_fields(
        fun_grad, np.zeros(256), lower=-5.0, upper=5.0, gtol=1e-6,
        maxiter=40, maxcor=maxcor, precond=helmholtz(0.05),
        device=device, pair_store="host")
    names = {name for name in fc.field_names
             if name.startswith("lbfgs-work")}
    expected_pairs = 1 if device == "gpu" else maxcor + 1
    assert len(names) == 8 + 2 * expected_pairs


def test_unknown_pair_store():
    fun_grad, _ = quadratic()
    with pytest.raises(ValueError, match="unknown pair store"):
        solve_on_fields(fun_grad, np.zeros(64), pair_store="disk")


@pytest.mark.parametrize("device", get_test_devices())
def test_distributed_host_pairs_give_the_iterates_of_device_pairs(comm,
                                                                  device):
    """On a decomposed grid with ghosts (every rank keeps its part of the
    pairs): the same iterates with the pairs on the host."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    _, _, b_global = grid_problem()
    laplace = muGrid.LaplaceOperator(2, -1.0)
    decomposition = muGrid.CartesianDecomposition(
        comm, NB_GRID_PTS, ghosts=laplace, device=create_device(device))
    location = decomposition.subdomain_locations
    size = decomposition.nb_subdomain_grid_pts
    local = tuple(slice(o, o + n) for o, n in zip(location, size))
    b = xp.asarray(b_global[local])
    Ax = decomposition.real_field("Ax")

    def fun(point, gradient):
        decomposition.communicate_ghosts(point)
        laplace.apply(point, Ax)                     # -Laplace u
        gradient.p[...] = point.p + Ax.p - b
        return comm.sum(float(
            (0.5 * point.p * (point.p + Ax.p) - b * point.p).sum()))

    def jacobi(v, out):
        out.p[...] = v.p / 5.0

    solutions = {}
    for store in ("device", "host"):
        x = decomposition.real_field(f"x-{store}")
        result = l_bfgs_bounded(
            comm, decomposition, fun, x, lower=0.0, upper=0.5,
            precond=jacobi, gtol=1e-6, maxcor=4, pair_store=store)
        assert result.success, result.message
        solutions[store] = (to_host(x.p).copy(), result.fun, result.nit,
                            result.nfev)
    np.testing.assert_array_equal(solutions["host"][0],
                                  solutions["device"][0])
    assert solutions["host"][1:] == solutions["device"][1:]
