"""Compare muGrid.Optimize.l_bfgs_bounded with NuMPI's l_bfgs_bounded on NuMPI's test problems.

Run with NuMPI (branch enh/lbfgs-precond) on the PYTHONPATH. For each problem the script prints the
number of iterations and evaluations, the final message and the difference of the solutions, for the
check arithmetic (must be bitwise NuMPI's) and for the kernel arithmetic (round-off).
"""
import numpy as np

import muGrid
from muGrid.Optimize import l_bfgs_bounded
from NuMPI.Optimization.BoundedLBFGS import l_bfgs_bounded as numpi_l_bfgs_bounded

comm = muGrid.Communicator()


def problems():
    """name, fun_grad(x) -> (f, g) on numpy vectors, x0, lower, upper, preconditioner or None, options"""
    rng = np.random.default_rng(0)
    y = rng.normal(size=64)
    yield ('quadratic, no bounds', lambda x: (0.5 * float(np.sum((x - y) ** 2)), x - y), np.zeros(64),
           None, None, None, dict(gtol=1e-10))

    y2 = np.random.default_rng(1).normal(size=128) * 0.6 + 0.3
    yield ('quadratic, solution on the box', lambda x: (0.5 * float(np.sum((x - y2) ** 2)), x - y2),
           np.full(128, 0.5), 0.0, 1.0, None, dict(gtol=1e-10))

    rng = np.random.default_rng(2)
    y3, w3 = rng.normal(size=200) * 0.6 + 0.3, 10.0 ** rng.uniform(-3, 0, 200)

    def weighted(x):
        d = x - y3
        return 0.5 * float(np.sum(w3 * d * d)), w3 * d
    yield ('ill-conditioned quadratic, box', weighted, np.full(200, 0.5), 0.0, 1.0, None, dict(gtol=1e-8))
    lo3, hi3 = rng.uniform(-0.5, 0.2, 200), rng.uniform(0.4, 1.2, 200)
    yield ('ill-conditioned quadratic, array bounds', weighted, np.full(200, 0.3), lo3, hi3, None, dict(gtol=1e-8))

    N, a = 256, 1e-3
    b = np.random.default_rng(10).normal(size=N)
    K_hat = 2 - 2 * np.cos(2 * np.pi * np.fft.fftfreq(N))

    def laplacian(x):
        Ax = a * x + 2 * x - np.roll(x, 1) - np.roll(x, -1)
        return 0.5 * float(x @ Ax) - float(b @ x), Ax - b

    def helmholtz(v):
        return np.real(np.fft.ifft(np.fft.fft(v) / (a + K_hat)))

    def smoother(v):                                            # not the exact inverse: several iterations
        return np.real(np.fft.ifft(np.fft.fft(v) / (0.05 + K_hat)))
    yield ('Laplacian, no preconditioner', laplacian, np.zeros(N), None, None, None, dict(gtol=1e-4, maxiter=5000))
    yield ('Laplacian, exact FFT Helmholtz', laplacian, np.zeros(N), None, None, helmholtz, dict(gtol=1e-8))
    yield ('Laplacian, FFT Helmholtz smoother', laplacian, np.zeros(N), None, None, smoother, dict(gtol=1e-6))
    yield ('Laplacian in a box, FFT Helmholtz smoother', laplacian, np.zeros(N), -5.0, 5.0, smoother,
           dict(gtol=1e-6, maxiter=2000))

    n = 200
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1) + 1e-2 * np.eye(n)
    M = np.linalg.inv(np.eye(n) + A - 1e-2 * np.eye(n))
    b2 = np.random.default_rng(0).normal(size=n)

    def many_active(x):
        return 0.5 * float(x @ A @ x) - float(b2 @ x), A @ x - b2
    kw = dict(gtol=1e-6, maxiter=3000)
    yield ('many active bounds, no preconditioner', many_active, np.full(n, 0.5), 0.0, 1.0, None, kw)
    yield ('many active bounds, smoother', many_active, np.full(n, 0.5), 0.0, 1.0, lambda v: M @ v, kw)
    yield ('many active bounds, xtol', many_active, np.full(n, 0.5), 0.0, 1.0, None, dict(gtol=0.0, xtol=1e-7))
    yield ('many active bounds, ftol', many_active, np.full(n, 0.5), 0.0, 1.0, None, dict(gtol=0.0, ftol=1e-9))


def solve_mugrid(fun_grad, x0, lower, upper, precond, options, arithmetic, device=None):
    """Wrap the numpy problem into field callables and solve with muGrid."""
    kwargs = {} if device is None else {'device': device}
    fc = muGrid.GlobalFieldCollection((x0.size,), **kwargs)
    to_host = (lambda a: a.get()) if device is not None else np.asarray
    x = fc.real_field('x')
    xp = type(x.s).__module__.split('.')[0]
    import importlib
    xp = importlib.import_module(xp)
    x.s[...] = xp.asarray(x0.reshape(x.s.shape))
    evaluations = [0]

    def fun(point, gradient):
        evaluations[0] += 1
        f, g = fun_grad(to_host(point.s).ravel())
        gradient.s[...] = xp.asarray(g.reshape(gradient.s.shape))
        return f

    def field_bound(name, value):
        if not isinstance(value, np.ndarray):
            return value
        field = fc.real_field(name)
        field.s[...] = xp.asarray(value.reshape(field.s.shape))
        return field

    prec = None
    if precond is not None:
        def prec(v, out):
            out.s[...] = xp.asarray(precond(to_host(v.s).ravel()).reshape(out.s.shape))
    result = l_bfgs_bounded(comm, fc, fun, x, lower=field_bound('lower', lower), upper=field_bound('upper', upper),
                            precond=prec, _arithmetic=arithmetic, **options)
    assert result.nfev == evaluations[0]
    return result, to_host(x.s).ravel().copy()


if __name__ == '__main__':
    for name, fun_grad, x0, lower, upper, precond, options in problems():
        evaluations = [0]

        def counted(x):
            evaluations[0] += 1
            return fun_grad(x)
        reference = numpi_l_bfgs_bounded(counted, x0.copy(), jac=True, bounds_lo=lower, bounds_hi=upper, precond=precond,
                                         maxcor=20, **options)
        print(f"{name}\n    NuMPI            : nit {reference.nit:4d}, nfev {evaluations[0]:4d}, f {reference.fun:.15e}, "
              f"{reference.message}")
        for arithmetic in ('numpy', 'kernels'):
            result, x = solve_mugrid(fun_grad, x0, lower, upper, precond, options, arithmetic)
            difference = np.abs(x - reference.x).max()
            same = (result.nit == reference.nit and result.nfev == evaluations[0]
                    and result.message == reference.message)
            print(f"    muGrid, {arithmetic:8s} : nit {result.nit:4d}, nfev {result.nfev:4d}, f {result.fun:.15e}, "
                  f"max|x - x_NuMPI| = {difference:.1e}, same nit/nfev/message: {same}")
