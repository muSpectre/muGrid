"""Vector work per iteration of bound-constrained L-BFGS implementations.

Minimises the separable quadratic 0.5 sum w_i (x_i - y_i)^2 on [0, 1]^n, whose weights span three
decades, so the optimiser iterates for long while the objective itself costs little. Reported: the
time per iteration spent outside the objective, averaged over the iterations made with a full memory
(maxcor pairs).

    python vector_work.py --impl mugrid --device gpu --n 4000000
    --impl: mugrid | lbfgs_xp (muFFTTO_fable's experiment-local optimiser) | numpi
"""
import argparse
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--impl', default='mugrid')
parser.add_argument('--device', default='cpu')
parser.add_argument('--n', type=int, default=4_000_000)
parser.add_argument('--maxcor', type=int, default=20)
parser.add_argument('--iterations', type=int, default=60)
parser.add_argument('--precond', action='store_true', help='a diagonal preconditioner')
args = parser.parse_args()

if args.device == 'gpu':
    import cupy as xp

    def synchronize():
        xp.cuda.Device().synchronize()
else:
    xp = np

    def synchronize():
        pass

rng = np.random.default_rng(0)
y = xp.asarray(rng.normal(size=args.n) * 0.6 + 0.3)
w = xp.asarray(10.0 ** rng.uniform(-3, 0, args.n))
scaling = xp.asarray(1.0 / (0.01 + np.asarray(w.get() if args.device == 'gpu' else w)))
fun_time = [0.0]
stamps = []                                                # (wall time, time in fun so far) after each iteration


def objective(x, g):
    """g = w (x - y) written into g, returns f; x and g are arrays of the namespace"""
    start = time.perf_counter()
    xp.subtract(x, y, out=g)
    f = 0.5 * float(xp.vdot(g * w, g))
    g *= w
    synchronize()
    fun_time[0] += time.perf_counter() - start
    return f


def stamp(*_):
    synchronize()
    stamps.append((time.perf_counter(), fun_time[0]))


options = dict(maxcor=args.maxcor, gtol=1e-14, maxiter=args.iterations)
if args.impl == 'mugrid':
    import muGrid
    from muGrid.Optimize import l_bfgs_bounded
    kwargs = {'device': muGrid.Device.gpu()} if args.device == 'gpu' else {}
    fc = muGrid.GlobalFieldCollection((args.n,), **kwargs)
    x = fc.real_field('x')
    x.s[...] = 0.5
    y, w, scaling = (a.reshape(x.s.shape) for a in (y, w, scaling))

    def precond(v, out):
        xp.multiply(v.s, scaling, out=out.s)
    result = l_bfgs_bounded(muGrid.Communicator(), fc, lambda p, g: objective(p.s, g.s), x, lower=0.0, upper=1.0,
                            precond=precond if args.precond else None, callback=stamp, **options)
    nit, f = result.nit, result.fun
elif args.impl == 'lbfgs_xp':
    import lbfgs_xp
    buffer = xp.empty(args.n)

    def fun(x):
        return objective(x, buffer), buffer

    def precond(v, out=None):
        xp.multiply(v, scaling, out=out)
        return out
    result = lbfgs_xp.l_bfgs_bounded_xp(fun, xp.full(args.n, 0.5), bounds_lo=0.0, bounds_hi=1.0,
                                        precond=precond if args.precond else None, callback=stamp, **options)
    nit, f = result['nit'], result['fun']
elif args.impl == 'numpi':
    from NuMPI.Optimization.BoundedLBFGS import l_bfgs_bounded as numpi_l_bfgs

    def fun(x):
        g = np.empty_like(x)
        return objective(x, g), g
    result = numpi_l_bfgs(fun, np.full(args.n, 0.5), jac=True, bounds_lo=0.0, bounds_hi=1.0,
                          precond=(lambda v: v * scaling) if args.precond else None, callback=stamp, **options)
    nit, f = result.nit, result.fun

# iterations made with a full memory: from iteration maxcor + 5 on
first = args.maxcor + 5
wall = stamps[-1][0] - stamps[first][0]
in_fun = stamps[-1][1] - stamps[first][1]
per_iteration = (wall - in_fun) / (len(stamps) - 1 - first)
print(f"{args.impl:9s} {args.device} n={args.n} precond={args.precond}: nit {nit}, f {f:.12e}, vector work "
      f"{1e3 * per_iteration:.1f} ms/iteration = {1e9 * per_iteration / args.n:.2f} ns/value "
      f"-> {per_iteration * 33.5e6 / args.n:.3f} s at 33.5 M values")
