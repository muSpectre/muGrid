"""
Preconditioners for the matrix-free solvers in :mod:`muGrid.Solvers`.

A preconditioner approximates the inverse of the system operator. The
solvers accept any callable with the signature ``prec(r, z)`` that computes
``z = M⁻¹ r`` for muGrid fields ``r`` and ``z``; the
:class:`Preconditioner` base class below formalizes this contract, so
instances can be passed directly as the ``prec`` argument of
:func:`muGrid.Solvers.conjugate_gradients`.

For conjugate gradients, ``M⁻¹`` must be symmetric positive definite on the
subspace the iteration operates in. A spectral preconditioner whose kernel
vanishes on a mode (e.g. the zero-frequency mode of the periodic Laplacian)
projects that mode out; in that case the right-hand side must not contain
it.
"""

from contextlib import nullcontext

import numpy as np

from . import linalg


class Preconditioner:
    """
    Abstract base class for preconditioners.

    Subclasses implement :meth:`apply`, which computes ``z = M⁻¹ r``.
    Instances are callable with the same signature, matching the ``prec``
    argument of :func:`muGrid.Solvers.conjugate_gradients`.
    """

    def apply(self, r, z):
        """
        Apply the preconditioner: ``z = M⁻¹ r``.

        Parameters
        ----------
        r : muGrid.Field
            Input (residual) field; not modified.
        z : muGrid.Field
            Output field, overwritten with the preconditioned residual.
        """
        raise NotImplementedError

    def __call__(self, r, z):
        self.apply(r, z)


class IdentityPreconditioner(Preconditioner):
    """No-op preconditioner, ``z = r``. Equivalent to passing ``prec=None``."""

    def apply(self, r, z):
        linalg.copy(r, z)


class JacobiPreconditioner(Preconditioner):
    """
    Diagonal (Jacobi) preconditioner, ``z = D⁻¹ r``.

    Useful when the operator's diagonal varies strongly in space, e.g. for
    heterogeneous coefficients: dividing by the diagonal equilibrates the
    spectrum. (For operators with a constant diagonal, such as the plain
    Laplacian, Jacobi only rescales the system and does not change the CG
    iteration.)

    Parameters
    ----------
    diagonal : muGrid.Field, array-like or scalar
        Diagonal entries of the system operator on the local subdomain.
        Either a field created on the same collection as the solver fields,
        an array broadcastable against the interior field values (shape
        ``(*spatial,)`` to share one diagonal across components, or
        ``(*components, *spatial)`` for per-component entries), or a
        scalar. The entries are inverted once at construction; the values
        are copied, later modification of the source has no effect.

    Raises
    ------
    ValueError
        If any diagonal entry is zero.
    """

    def __init__(self, diagonal):
        values = getattr(diagonal, "s", None)  # muGrid field?
        if values is None:
            values = np.asarray(diagonal, dtype=float)
        if not (abs(values) > 0).all():
            raise ValueError(
                "Jacobi preconditioner requires a non-singular diagonal "
                "(got entries equal to zero)"
            )
        self._inverse_diagonal = 1.0 / values

    def apply(self, r, z):
        z.s[...] = r.s * self._inverse_diagonal


class FourierPreconditioner(Preconditioner):
    """
    Spectral preconditioner ``z = F⁻¹ [ k(q) · F r ]``.

    The kernel ``k(q)`` is the Fourier-space representation of ``M⁻¹``,
    typically the inverse symbol of (an approximation to) the system
    operator. It is applied pointwise on the local Fourier subdomain and
    broadcast over field components. Modes where the kernel is zero are
    projected out of the solution; for singular operators (e.g. the
    periodic Laplacian, whose symbol vanishes at ``q = 0``) set the kernel
    to zero there and keep the right-hand side free of that mode.

    The fields passed to :meth:`apply` must belong to the engine's
    real-space field collection (create them with
    ``engine.real_space_field`` or, inside the solver, by passing
    ``engine.real_space_collection`` as the field collection), so that the
    transforms operate without intermediate copies. This also makes the
    preconditioner MPI-transparent: the kernel only ever sees the rank-local
    Fourier subdomain.

    Parameters
    ----------
    engine : muGrid.FFTEngine
        FFT engine defining grid, parallel decomposition and transforms.
    kernel : ndarray or callable
        Either an array of shape ``engine.nb_fourier_subdomain_grid_pts``
        holding ``k(q)`` on the local Fourier subdomain, or a callable
        ``kernel(engine) -> ndarray`` evaluated once at construction.
        Use ``engine.fftfreq`` (normalized frequencies, shape
        ``[dim, *local_fourier_shape]``) to build it.
    name : str, optional
        Prefix for the engine-managed Fourier work fields.
    timer : muTimer.Timer, optional
        Timer for performance profiling. When given, :meth:`apply` records
        the forward transform ("fft"), the pointwise kernel multiplication
        ("scale") and the inverse transform ("ifft").

    Examples
    --------
    Exact inverse of the second-order finite-difference Laplacian
    (five-point stencil divided by ``h**2``), zero mode projected out::

        def inverse_fd_laplacian(engine):
            q = engine.fftfreq  # shape [dim, *local_fourier_shape]
            denom = (4 * np.sin(np.pi * q) ** 2 / h ** 2).sum(axis=0)
            with np.errstate(divide="ignore"):
                k = np.where(denom > 0, 1 / denom, 0.0)
            return k

        prec = FourierPreconditioner(engine, inverse_fd_laplacian)
        conjugate_gradients(comm, engine.real_space_collection,
                            rhs, solution, hessp=hessp, prec=prec)
    """

    def __init__(self, engine, kernel, name="fourier-preconditioner", timer=None):
        self._engine = engine
        self._name = name
        self._timer = timer
        self._work = {}  # work field per components shape

        if callable(kernel):
            kernel = kernel(engine)
        kernel = np.asarray(kernel)

        expected = tuple(engine.nb_fourier_subdomain_grid_pts)
        if kernel.shape != expected:
            raise ValueError(
                f"Kernel shape {kernel.shape} does not match the local "
                f"Fourier subdomain {expected} of the FFT engine"
            )

        # Fold the inverse-transform normalisation into the kernel so apply()
        # is a single pointwise multiplication.
        self._kernel = kernel * engine.normalisation

    def _work_field(self, components_shape):
        key = tuple(components_shape)
        if key not in self._work:
            suffix = "x".join(str(c) for c in key) if key else "scalar"
            self._work[key] = self._engine.fourier_space_field(
                f"{self._name}-work-{suffix}", components=key
            )
        return self._work[key]

    def _timed(self, name):
        return self._timer(name) if self._timer is not None else nullcontext()

    def apply(self, r, z):
        """
        Compute ``z = F⁻¹ [ k(q) · F r ]``.

        ``r`` and ``z`` must be real-valued fields of the engine's
        real-space collection with identical component shapes.
        """
        work = self._work_field(r.components_shape)
        engine = self._engine
        with self._timed("fft"):
            engine.fft(r, work)
        s = work.s
        kernel = self._kernel
        if type(s).__module__.partition(".")[0] == "cupy":
            # Device fields: move the kernel over once, lazily.
            import cupy

            kernel = cupy.asarray(kernel)
            self._kernel = kernel
        # Broadcasts over leading component axes; normalisation is folded in.
        with self._timed("scale"):
            s[...] *= kernel
        with self._timed("ifft"):
            engine.ifft(work, z)
