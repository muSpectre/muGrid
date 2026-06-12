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
from .Field import wrap_field


def _fill_field(field, values):
    """Assign host values to the interior view of a host or device field."""
    s = field.s
    try:
        s[...] = values
    except (TypeError, ValueError):
        # Device view: cupy's __setitem__ does not accept numpy sources;
        # convert once (setup time only).
        import cupy

        s[...] = cupy.asarray(values)


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

    The preconditioner runs wherever the solver fields live: the inverse
    diagonal is stored in a field on the residual's collection (created on
    first application) and applied with the fused ``linalg.copy`` +
    ``linalg.scal`` kernels, on host and device alike.

    Parameters
    ----------
    diagonal : muGrid.Field, array-like or scalar
        Diagonal entries of the system operator on the local subdomain.
        Either a field created on the same collection as the solver fields
        (host or device), an array matching the interior field values
        (shape ``(*spatial,)`` to share one diagonal across components, or
        ``(*components, *spatial)`` for per-component entries), or a
        scalar. The entries are inverted once at construction; the values
        are copied, later modification of the source has no effect.
    name : str, optional
        Prefix for the field holding the inverse diagonal.

    Raises
    ------
    ValueError
        If any diagonal entry is zero.
    """

    def __init__(self, diagonal, name="jacobi-preconditioner"):
        self._name = name
        values = getattr(diagonal, "s", None)  # muGrid field?
        if values is not None:
            # Device fields expose cupy views; pull a host copy
            values = values.get() if hasattr(values, "get") else np.array(values)
        else:
            values = np.asarray(diagonal, dtype=float)
        if not (np.abs(values) > 0).all():
            raise ValueError(
                "Jacobi preconditioner requires a non-singular diagonal "
                "(got entries equal to zero)"
            )
        self._is_scalar = values.ndim == 0
        self._inverse_diagonal = (
            1.0 / float(values) if self._is_scalar else 1.0 / values
        )
        self._field = None

    def _inverse_diagonal_field(self, z):
        """Field holding D⁻¹ on z's collection (created on first use)."""
        if self._field is None:
            values = self._inverse_diagonal
            # Spatial-only diagonals go into a single-component field that
            # linalg.scal broadcasts over z's components; values that do
            # not fit the spatial shape are per-component diagonals.
            nb_component_axes = len(tuple(z.components_shape))
            spatial_shape = tuple(z.s.shape)[nb_component_axes:]
            try:
                np.broadcast_to(values, spatial_shape)
                components = ()
            except ValueError:
                components = tuple(z.components_shape)
            field = wrap_field(
                z.collection.real_field(
                    f"{self._name}-inverse-diagonal", components
                )
            )
            # scal operates on the full buffer; zero ghost entries keep
            # the (later overwritten) ghost values of z finite.
            field.set_zero()
            _fill_field(field, np.broadcast_to(values, field.s.shape))
            self._field = field
        return self._field

    def apply(self, r, z):
        linalg.copy(r, z)
        if self._is_scalar:
            linalg.scal(self._inverse_diagonal, z)
        else:
            linalg.scal(self._inverse_diagonal_field(z), z)


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

        # Store the kernel in a real field on the engine's Fourier
        # collection (host or device, matching the work fields) and fold
        # the inverse-transform normalisation in, so apply() is a single
        # linalg.scal with no array-library dependence in the
        # hot loop.
        self._kernel_field = engine.fourier_space_collection.real_field(
            f"{name}-kernel"
        )
        values = kernel * engine.normalisation
        s = self._kernel_field.s
        try:
            s[...] = values
        except (TypeError, ValueError):
            # Device view: cupy's __setitem__ does not accept numpy
            # sources; convert once at setup.
            import cupy

            s[...] = cupy.asarray(values)

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
        # Fused C++ kernel, host or device; broadcasts over components and
        # has the inverse-transform normalisation folded in.
        with self._timed("scale"):
            linalg.scal(self._kernel_field, work)
        with self._timed("ifft"):
            engine.ifft(work, z)
