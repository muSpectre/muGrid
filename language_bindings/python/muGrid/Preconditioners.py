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


def _real_field_like(field, name):
    """Get-or-create a real field on ``field``'s collection with the same
    component shape and scalar precision (float64 or float32) as ``field``."""
    coll = field.collection
    components = tuple(field.components_shape)
    if np.dtype(field.dtype) == np.dtype(np.float32):
        # register_real32_field is register-only; mirror real_field's
        # get-or-create semantics.
        if coll.field_exists(name):
            return wrap_field(coll.get_field(name))
        return wrap_field(coll.register_real32_field(name, components))
    return wrap_field(coll.real_field(name, components))


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
    dtype : data-type, optional
        Real-space precision of the fields the preconditioner is applied to:
        ``np.float64`` (default) or ``np.float32``. The kernel field and the FFT
        work buffer are created at the matching precision so the internal
        transforms pair with the solver's fields.

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

    def __init__(self, engine, kernel, name="fourier-preconditioner", timer=None,
                 dtype=np.float64):
        self._engine = engine
        self._name = name
        self._timer = timer
        self._work = {}  # work field per components shape

        self._real_dtype = np.dtype(dtype)
        if self._real_dtype == np.dtype(np.float32):
            self._complex_dtype = np.dtype(np.complex64)
        elif self._real_dtype == np.dtype(np.float64):
            self._complex_dtype = np.dtype(np.complex128)
        else:
            raise ValueError(
                f"FourierPreconditioner dtype must be float32 or float64, "
                f"got {self._real_dtype}"
            )

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
            f"{name}-kernel", dtype=self._real_dtype
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
                f"{self._name}-work-{suffix}", components=key,
                dtype=self._complex_dtype
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


class BlockFourierPreconditioner(Preconditioner):
    """
    Block spectral preconditioner ``z = F⁻¹ [ K⁻¹(q) · F r ]`` for an
    ``n``-component vector field.

    The matrix-valued generalization of :class:`FourierPreconditioner`: where
    that class multiplies each Fourier mode by a scalar, this one multiplies the
    ``n``-vector of component amplitudes at each mode ``q`` by an ``n × n``
    matrix ``K⁻¹(q)``. This is exactly what the reference-material (Green's
    function) preconditioner of an FE homogenization problem needs (Ladecký et
    al., Appl. Math. Comput. 446 (2023) 127835): the reference stiffness
    ``Kʳᵉᶠ = Dᵀ W Cʳᵉᶠ D`` built from spatially uniform data is block-circulant,
    hence block-diagonal in Fourier space with one ``n × n`` block per mode
    (``n = d·Nn`` degrees of freedom per stencil), and its (pseudo-)inverse is
    applied mode by mode between a forward and an inverse FFT.

    The blocks are supplied pre-assembled and pre-inverted (the singular
    zero-frequency block, corresponding to the rigid-body modes, must already be
    set to its pseudo-inverse — typically zero — to project those modes out).
    The inverse-transform normalisation should be folded into the blocks by the
    caller, mirroring :class:`FourierPreconditioner`.

    Parameters
    ----------
    engine : muGrid.FFTEngine
        FFT engine defining grid, parallel decomposition and transforms.
    blocks : ndarray
        Per-mode inverse blocks of shape ``(n, n, *nb_fourier_subdomain_grid_pts)``
        (generally complex), already including ``engine.normalisation``. ``z[i] =
        Σ_j blocks[i, j] · F r[j]`` at every Fourier mode.
    name : str, optional
        Prefix for the engine-managed Fourier work field.
    timer : muTimer.Timer, optional
        When given, :meth:`apply` records the forward transform ("fft"), the
        per-mode block multiply ("scale") and the inverse transform ("ifft").
    dtype : data-type, optional
        Real-space precision of the fields the preconditioner will be applied to:
        ``np.float64`` (default) or ``np.float32``. The FFT work buffer and the
        stored symbol are created at the matching precision (``complex128`` /
        ``complex64``) so the internal transforms pair correctly with the
        solver's fields -- a single-precision solve needs a single-precision
        preconditioner, otherwise the internal ``ifft`` mismatches and the
        result is NaN.
    """

    def __init__(self, engine, blocks, name="block-fourier-preconditioner",
                 timer=None, dtype=np.float64):
        self._engine = engine
        self._name = name
        self._timer = timer

        real_dtype = np.dtype(dtype)
        if real_dtype == np.dtype(np.float32):
            complex_dtype = np.dtype(np.complex64)
        elif real_dtype == np.dtype(np.float64):
            complex_dtype = np.dtype(np.complex128)
        else:
            raise ValueError(
                f"BlockFourierPreconditioner dtype must be float32 or float64, "
                f"got {real_dtype}"
            )

        blocks = np.asarray(blocks)
        n = blocks.shape[0]
        if blocks.ndim < 2 or blocks.shape[1] != n:
            raise ValueError(
                f"blocks must have shape (n, n, *fourier_shape); got "
                f"{blocks.shape}"
            )
        expected = tuple(engine.nb_fourier_subdomain_grid_pts)
        if tuple(blocks.shape[2:]) != expected:
            raise ValueError(
                f"blocks Fourier shape {tuple(blocks.shape[2:])} does not match "
                f"the local Fourier subdomain {expected} of the FFT engine"
            )

        self._n = n
        self._work = engine.fourier_space_field(
            f"{name}-work", components=(n,), dtype=complex_dtype)
        # Match the array library of the work field's view (numpy or cupy) so
        # the per-mode multiply runs where the fields live.
        sample = self._work.s
        if type(sample).__module__.startswith("cupy"):
            import cupy

            self._xp = cupy
        else:
            self._xp = np
        xp = self._xp

        # The reference-stiffness symbol K̂(q) is Hermitian (the FE stiffness is
        # real and self-adjoint), so its inverse is Hermitian too. When that
        # holds, store only the upper triangle -- n real diagonals plus
        # n(n-1)/2 complex off-diagonals, i.e. n² reals/mode instead of 2n² for
        # the dense complex block (a 2x saving on the symbol, the largest
        # persistent buffer of the preconditioner). A non-Hermitian operator
        # (general use of this class) keeps the dense block. Either way apply()
        # multiplies component-by-component (no batched einsum, hence no cuBLAS
        # gemm path) and needs only n-1 single-component transients.
        #
        # The detection and triangle extraction run on the host array `blocks`,
        # and only the compressed pieces are moved to the device -- so the
        # device never has to hold the dense n×n complex block, even
        # transiently, during construction.
        herm_scale = float(np.max(np.abs(blocks))) if blocks.size else 0.0
        herm_asym = (
            float(np.max(np.abs(blocks - np.conj(np.swapaxes(blocks, 0, 1)))))
            if blocks.size
            else 0.0
        )
        # The detection must be COLLECTIVE: a rank whose Fourier subdomain is
        # empty (more ranks than modes along the split direction) sees an
        # empty slab and would conclude "Hermitian" while data-carrying ranks
        # conclude the opposite, leaving per-rank state divergent -- and any
        # rank-dependent branching downstream deadlocks the MPI run.
        comm = getattr(engine, "communicator", None)
        if comm is not None:
            herm_scale = float(comm.max(herm_scale))
            herm_asym = float(comm.max(herm_asym))
        self._hermitian = herm_scale == 0.0 or herm_asym <= 1e-10 * herm_scale

        # Store the symbol at the solve precision (real diagonals, complex
        # off-diagonals / dense block): it is the largest persistent buffer, and
        # a single-precision solve should not carry a double-precision symbol.
        if self._hermitian:
            diag = np.empty((n,) + blocks.shape[2:], dtype=real_dtype)
            for i in range(n):
                diag[i] = blocks[i, i].real
            self._diag = xp.asarray(diag)
            self._off = {
                (i, j): xp.asarray(
                    np.ascontiguousarray(blocks[i, j]).astype(complex_dtype))
                for i in range(n)
                for j in range(i + 1, n)
            }
            self._blocks = None
        else:
            self._diag = None
            self._off = None
            self._blocks = xp.asarray(
                np.ascontiguousarray(blocks).astype(complex_dtype))

    def _timed(self, name):
        return self._timer(name) if self._timer is not None else nullcontext()

    def _block(self, i, j):
        """Per-mode entry K⁻¹_ij(q) as a Fourier array, reconstructed from the
        stored upper triangle in the Hermitian case (lower triangle is the
        conjugate of the upper)."""
        if not self._hermitian:
            return self._blocks[i, j]
        if i == j:
            return self._diag[i]
        if i < j:
            return self._off[(i, j)]
        return self._xp.conj(self._off[(j, i)])

    def apply(self, r, z):
        """
        Compute ``z = F⁻¹ [ K⁻¹(q) · F r ]``.

        ``r`` and ``z`` must be real-valued ``n``-component fields of the
        engine's real-space collection.
        """
        engine = self._engine
        work = self._work
        with self._timed("fft"):
            engine.fft(r, work)
        with self._timed("scale"):
            s = work.s
            # z_i(q) = Σ_j K⁻¹_ij(q) r_j(q), per Fourier mode, evaluated as
            # explicit component multiplies/adds (no einsum -> no cuBLAS batched
            # gemm). Each output needs every input and the update is in place on
            # the work buffer, so the first n-1 outputs are buffered and the
            # last is written straight into s while its inputs are still intact
            # -- keeping the transient to n-1 single-component buffers rather
            # than a full n-component einsum temporary. The field view carries a
            # size-1 sub-point axis between the component and Fourier axes; the
            # per-mode blocks (no sub-point axis) broadcast over it.
            n = self._n
            new = []
            for i in range(n - 1):
                acc = self._block(i, 0) * s[0]
                for j in range(1, n):
                    acc = acc + self._block(i, j) * s[j]
                new.append(acc)
            last = self._block(n - 1, 0) * s[0]
            for j in range(1, n):
                last = last + self._block(n - 1, j) * s[j]
            s[n - 1] = last
            for i in range(n - 1):
                s[i] = new[i]
        with self._timed("ifft"):
            engine.ifft(work, z)


class GreenJacobiPreconditioner(Preconditioner):
    r"""
    Green-Jacobi preconditioner ``z = J^{1/2} G J^{1/2} r`` — the J-FFT scheme
    of Ladecký et al. ("Jacobi-accelerated FFT-based solver for smooth
    high-contrast data").

    The standard Green's-function (reference-material) preconditioner ``G`` is a
    *global*, spatially-uniform approximation of the inverse operator, applied in
    Fourier space. Its conditioning degrades when the material data is smoothly
    varying at high contrast — exactly the regime of phase-field topology
    optimization, grid adaptation, and nonlinear effective moduli — where the
    Green-preconditioned spectrum spreads out. Scaling ``G`` symmetrically by the
    *local* Jacobi diagonal ``J = diag(K)^{-1}`` of the actual (heterogeneous)
    system matrix re-clusters the spectrum around one and restores fast CG
    convergence, while keeping the ``O(N log N)`` cost of the FFT-based apply.

    The symmetric split ``J^{1/2} G J^{1/2}`` keeps ``M⁻¹`` symmetric
    positive-definite (``G`` is SPD on the non-rigid-body subspace and
    ``J^{1/2}`` is a positive diagonal), so plain PCG remains valid.

    Parameters
    ----------
    green : Preconditioner or callable
        The inner Green's-function preconditioner (e.g. the result of
        :func:`make_reference_stiffness_preconditioner`). Any object with an
        ``apply(r, z)`` / ``__call__(r, z)`` computing ``z = G r`` works.
    diagonal : muGrid.Field
        The diagonal ``diag(K)`` of the actual system matrix, as a field on the
        solver's (real-space) collection with the same component shape as the
        solver fields. Typically assembled with
        :meth:`IsotropicStiffnessOperator.assemble_diagonal`. Zero (or
        non-positive) entries — e.g. true void — are treated as
        ``J^{1/2} = 1``; those degrees of freedom carry no stiffness and do not
        couple, so the replacement value does not affect the solution.
    void_tol : float, optional
        Diagonal entries ``<= void_tol`` are treated as void (``J^{1/2} = 1``).
        Default ``0.0``.
    name : str, optional
        Prefix for the ``J^{1/2}`` and scratch work fields.
    timer : muTimer.Timer, optional
        When given, :meth:`apply` records the inner Green apply ("green") and the
        two diagonal scalings ("scale").
    """

    def __init__(self, green, diagonal, void_tol=0.0,
                 name="green-jacobi-preconditioner", timer=None):
        self._green = green
        self._name = name
        self._timer = timer
        self._void_tol = float(void_tol)
        self._jhalf = None
        self._work = None
        self.update_diagonal(diagonal)

    def _timed(self, name):
        return self._timer(name) if self._timer is not None else nullcontext()

    def update_diagonal(self, diagonal):
        r"""(Re)compute ``J^{1/2} = diag(K)^{-1/2}`` from a freshly assembled
        diagonal field. Call this whenever the material (and hence the system
        matrix) changes, e.g. once per optimization or Newton step; the inner
        Green preconditioner, built from spatially uniform reference data, does
        not change and is reused."""
        s = diagonal.s
        vals = s.get() if hasattr(s, "get") else np.asarray(s)
        positive = vals > self._void_tol
        # Guard the sqrt against void/roundoff-negative entries; those DOFs get
        # J^{1/2} = 1 via the where() below regardless of the safe denominator.
        safe = np.where(positive, vals, 1.0)
        jhalf = np.where(positive, 1.0 / np.sqrt(safe), 1.0)
        if self._jhalf is None:
            # Match the diagonal's precision so linalg.scal pairs J^{1/2}
            # with the solver fields in a single-precision solve.
            self._jhalf = _real_field_like(diagonal, f"{self._name}-jhalf")
        # Zero the ghosts (as JacobiPreconditioner does); only the interior
        # participates in the CG inner products.
        self._jhalf.set_zero()
        _fill_field(self._jhalf, jhalf)

    def _work_field(self, r):
        if self._work is None:
            self._work = _real_field_like(r, f"{self._name}-work")
            self._work.set_zero()
        return self._work

    def apply(self, r, z):
        r"""Compute ``z = J^{1/2} G ( J^{1/2} r )``."""
        w = self._work_field(r)
        with self._timed("scale"):
            linalg.copy(r, w)
            linalg.scal(self._jhalf, w)  # w = J^{1/2} r
        with self._timed("green"):
            self._green.apply(w, z)  # z = G w
        with self._timed("scale"):
            linalg.scal(self._jhalf, z)  # z = J^{1/2} z


def make_reference_stiffness_preconditioner(
    engine,
    apply_reference_stiffness,
    nb_components,
    name="reference-stiffness-preconditioner",
    timer=None,
    dtype=np.float64,
):
    """
    Build the reference-material (Green's-function) preconditioner of Ladecký et
    al., Appl. Math. Comput. 446 (2023) 127835.

    For an FE problem on a regular periodic grid, the reference-material
    stiffness ``Kʳᵉᶠ = Dᵀ W Cʳᵉᶠ D`` built from spatially *uniform* data is
    block-circulant — every pixel carries the same stencil — hence
    block-diagonal in Fourier space, with one ``n × n`` block ``K̂(q)`` per mode
    (``n`` degrees of freedom per stencil). This routine assembles ``K̂(q)`` by
    the impulse-response method (paper Algorithm 2): it applies ``Kʳᵉᶠ`` to a
    unit nodal impulse in each of the ``n`` directions placed at the global
    origin pixel, and the FFT of the response is the ``β``-th column of the
    symbol. Each block is inverted; the singular zero-frequency block (the
    rigid-body modes) is replaced by its pseudo-inverse (zero), which projects
    those modes out — consistent with a rigid-body-free right-hand side. The
    inverse-transform normalisation is folded in, and a
    :class:`BlockFourierPreconditioner` applying ``F⁻¹ K̂⁻¹(q) F`` is returned.

    The routine is FE-agnostic: it only needs the action of ``Kʳᵉᶠ``. The caller
    supplies that as ``apply_reference_stiffness`` (e.g. built from a uniform
    reference stiffness and the discrete gradient/divergence operators).

    Parameters
    ----------
    engine : muGrid.FFTEngine
        FFT engine; its real-space collection holds the fields, and it provides
        the transforms and the parallel (MPI) decomposition. The impulse
        assembly and the per-mode inverse are computed on the rank-local Fourier
        subdomain, so this is MPI-transparent.
    apply_reference_stiffness : callable
        ``apply_reference_stiffness(u, f)`` computing ``f = Kʳᵉᶠ u`` for
        ``n``-component real fields ``u``, ``f`` on the engine's real-space
        collection (it may use ghost communication internally).
    nb_components : int
        Degrees of freedom per stencil ``n`` (e.g. ``dim`` for one node per
        pixel).
    name : str, optional
        Prefix for the engine-managed work fields and the preconditioner.
    timer : muTimer.Timer, optional
        Forwarded to the returned preconditioner (records "fft"/"scale"/"ifft").
    dtype : data-type, optional
        Real-space precision of the solve, ``np.float64`` (default) or
        ``np.float32``. The impulse-response fields and the returned
        preconditioner's work buffer are created at this precision (with the
        matching complex type) so the internal transforms pair with the solver's
        fields; ``apply_reference_stiffness`` is therefore invoked on fields of
        this precision too. The symbol itself is assembled and inverted in
        double regardless, for accuracy.

    Returns
    -------
    BlockFourierPreconditioner
        The assembled preconditioner, ready to pass as ``prec=`` to
        :func:`muGrid.Solvers.conjugate_gradients`.
    """
    fourier_shape = tuple(engine.nb_fourier_subdomain_grid_pts)
    dim = len(fourier_shape)
    n = int(nb_components)

    real_dtype = np.dtype(dtype)
    if real_dtype == np.dtype(np.float32):
        complex_dtype = np.dtype(np.complex64)
    elif real_dtype == np.dtype(np.float64):
        complex_dtype = np.dtype(np.complex128)
    else:
        raise ValueError(
            f"reference-stiffness preconditioner dtype must be float32 or "
            f"float64, got {real_dtype}"
        )

    # Global-origin pixel(s) in this rank's interior: the nodal coordinate is
    # exactly 0 in every direction only at global index 0 (coord = index / N).
    # In MPI only the rank owning the origin matches; the others contribute no
    # impulse, which is correct for the global impulse response.
    coords = np.asarray(engine.coords)  # [dim, *local_grid]
    origin_mask = np.ones(coords.shape[1:], dtype=bool)
    for d in range(dim):
        origin_mask &= coords[d] == 0.0

    impulse_name = f"{name}-impulse"
    column_name = f"{name}-column"
    column_hat_name = f"{name}-column-hat"
    impulse = engine.real_space_field(
        impulse_name, components=(n,), dtype=real_dtype)
    column = engine.real_space_field(
        column_name, components=(n,), dtype=real_dtype)
    column_hat = engine.fourier_space_field(
        column_hat_name, components=(n,), dtype=complex_dtype)

    # K_hat[alpha, beta, q] = (FFT of Kʳᵉᶠ applied to impulse e_beta)[alpha](q)
    K_hat = np.zeros((n, n) + fourier_shape, dtype=complex)
    for beta in range(n):
        host_impulse = np.zeros(impulse.s.shape)
        # component beta, all (single) sub-points, at the origin pixel(s)
        host_impulse[beta][..., origin_mask] = 1.0
        try:
            impulse.s[...] = host_impulse
        except (TypeError, ValueError):
            import cupy

            impulse.s[...] = cupy.asarray(host_impulse)

        apply_reference_stiffness(impulse, column)
        engine.fft(column, column_hat)

        ch = column_hat.s
        ch = ch.get() if hasattr(ch, "get") else np.asarray(ch)
        # (n, [sub...], *fourier) -> (n, *fourier): collapse and drop the
        # single nodal sub-point.
        ch = ch.reshape((n, -1) + fourier_shape)[:, 0]
        K_hat[:, beta] = ch

    # Invert each n x n block; project out the singular zero-frequency block.
    blocks = np.moveaxis(K_hat, (0, 1), (-2, -1))  # [*fourier, n, n]
    inv = np.zeros_like(blocks)
    q = np.asarray(engine.fftfreq)  # [dim, *fourier]
    zero_mode = np.ones(fourier_shape, dtype=bool)
    for d in range(dim):
        zero_mode &= q[d] == 0.0
    nonzero = ~zero_mode
    inv[nonzero] = np.linalg.inv(blocks[nonzero])
    # [dim, dim, *fourier], with the inverse-transform normalisation folded in.
    K_inv = np.moveaxis(inv, (-2, -1), (0, 1)) * engine.normalisation

    # Release the impulse-response scratch. These three fields (two real, one
    # Fourier) were only needed to assemble the symbol above; left in the
    # engine's collections they would persist through the entire solve. The
    # symbol now lives in K_inv (a plain array, copied onto the device inside
    # the preconditioner), so the only Fourier buffer the solve then needs is
    # the preconditioner's own work field. Freeing here drops ~3 vector-sized
    # buffers from the resident set during the CG iteration.
    engine.real_space_collection.pop_field(impulse_name)
    engine.real_space_collection.pop_field(column_name)
    engine.fourier_space_collection.pop_field(column_hat_name)
    del impulse, column, column_hat

    return BlockFourierPreconditioner(
        engine, K_inv, name=name, timer=timer, dtype=real_dtype)


def make_green_jacobi_preconditioner(
    engine,
    stiffness_op,
    lambda_field,
    mu_field,
    nb_components,
    reference_lambda=None,
    reference_mu=None,
    void_tol=0.0,
    name="green-jacobi-preconditioner",
    timer=None,
    dtype=np.float64,
):
    r"""
    Assemble the Green-Jacobi (J-FFT) preconditioner for FFT-accelerated FE
    homogenization with the fused :class:`IsotropicStiffnessOperator`.

    This wires together the two ingredients of
    :class:`GreenJacobiPreconditioner`:

    * the Green's-function (reference-material) preconditioner ``G``, built from
      the operator's spatially-uniform reference stiffness
      (:meth:`IsotropicStiffnessOperator.apply_uniform`) via
      :func:`make_reference_stiffness_preconditioner`; and
    * the Jacobi diagonal ``diag(K)`` of the actual heterogeneous system matrix,
      assembled by the fused
      :meth:`IsotropicStiffnessOperator.assemble_diagonal` kernel (host/GPU,
      MPI-aware).

    The reference Lamé parameters default to the volume means of the supplied
    ``lambda_field`` / ``mu_field`` (a common, robust choice). The returned
    preconditioner exposes :meth:`GreenJacobiPreconditioner.update_diagonal`
    (and the convenience :meth:`refresh` below) to recompute the Jacobi part
    when the material changes across optimization/Newton steps; the Green part
    is reference-only and is reused unchanged.

    Parameters
    ----------
    engine : muGrid.FFTEngine
        FFT engine; its real-space collection holds the solver fields.
    stiffness_op : IsotropicStiffnessOperator2D or 3D
        The fused stiffness operator (also the system matrix of the solve).
    lambda_field, mu_field : muGrid.Field
        Per-pixel Lamé fields of the actual material (with ghosts filled), on a
        collection whose computable region matches ``engine.real_space_collection``.
    nb_components : int
        Degrees of freedom per node (``dim`` for one node per pixel).
    reference_lambda, reference_mu : float, optional
        Uniform reference Lamé parameters for the Green part. Default: the local
        means of ``lambda_field`` / ``mu_field`` (not MPI-reduced; pass explicit
        values for a deterministic reference under domain decomposition).
    void_tol : float, optional
        Passed to :class:`GreenJacobiPreconditioner`.
    name : str, optional
        Prefix for the managed fields.
    timer : muTimer.Timer, optional
        Forwarded to both sub-preconditioners.
    dtype : data-type, optional
        Real-space precision of the fields the preconditioner will be applied
        to: ``np.float64`` (default) or ``np.float32``. Forwarded to the inner
        Green preconditioner (whose FFT work buffer must pair with the solver
        fields) and used for the Jacobi diagonal and ``J^{1/2}`` fields. Must
        match the precision of ``lambda_field`` / ``mu_field``.

    Returns
    -------
    GreenJacobiPreconditioner
        Ready to pass as ``prec=`` to
        :func:`muGrid.Solvers.conjugate_gradients`, with a ``refresh()`` method
        bound for in-place material updates.
    """
    n = int(nb_components)

    if reference_lambda is None:
        lam_s = lambda_field.s
        reference_lambda = float(
            (lam_s.get() if hasattr(lam_s, "get") else np.asarray(lam_s)).mean()
        )
    if reference_mu is None:
        mu_s = mu_field.s
        reference_mu = float(
            (mu_s.get() if hasattr(mu_s, "get") else np.asarray(mu_s)).mean()
        )

    def apply_reference_stiffness(u, f):
        engine.communicate_ghosts(u)
        stiffness_op.apply_uniform(u, reference_lambda, reference_mu, f)

    green = make_reference_stiffness_preconditioner(
        engine, apply_reference_stiffness, n, name=f"{name}-green",
        timer=timer, dtype=dtype
    )

    diagonal = engine.real_space_field(
        f"{name}-diagonal", components=(n,), dtype=dtype)
    stiffness_op.assemble_diagonal(lambda_field, mu_field, diagonal)

    prec = GreenJacobiPreconditioner(
        green, diagonal, void_tol=void_tol, name=name, timer=timer
    )

    # Keep the ingredients so the Jacobi part can be recomputed in place when
    # the material changes (the Green part is reference-only and stays fixed).
    prec._stiffness_op = stiffness_op
    prec._lambda_field = lambda_field
    prec._mu_field = mu_field
    prec._diagonal = diagonal

    def refresh():
        """Re-assemble diag(K) from the (updated) material fields and refresh
        J^{1/2}. Call after changing ``lambda_field`` / ``mu_field`` in place."""
        stiffness_op.assemble_diagonal(
            prec._lambda_field, prec._mu_field, prec._diagonal
        )
        prec.update_diagonal(prec._diagonal)

    prec.refresh = refresh
    return prec
