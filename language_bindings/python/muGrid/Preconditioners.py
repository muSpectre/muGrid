"""
Preconditioners for the matrix-free solvers in :mod:`muGrid.Solvers`.

A preconditioner ``M ≈ A⁻¹`` approximates the inverse of the system operator
``A`` of ``A x = b``. The solvers accept any callable with the signature
``prec(r, z)`` that computes ``z = M r`` for muGrid fields ``r`` and ``z``;
the :class:`Preconditioner` base class below formalizes this contract, so
instances can be passed directly as the ``prec`` argument of
:func:`muGrid.Solvers.conjugate_gradients`.

For conjugate gradients, ``M`` must be a fixed linear operator that is
symmetric positive definite on the subspace the iteration operates in; CG
then converges at a rate governed by ``κ(M A)``. A spectral preconditioner
whose kernel vanishes on a mode (e.g. the zero-frequency mode of the periodic
Laplacian) projects that mode out; in that case the right-hand side must not
contain it.

Notation: ``d`` is the dimension, ``N_α`` and ``h_α`` the grid points and
spacing along axis ``α``, ``P`` the number of MPI ranks; ``F`` is the
unnormalised forward FFT and ``F⁻¹`` its inverse including the factor
``engine.normalisation = 1/∏N_α``. Fourier frequencies ``q`` are in cycles
per grid point (``engine.fftfreq``, axis 0 half-complex). For FE elasticity
the system matrix is ``K = Bᵀ W C B`` with nodal displacement ``u``, gradient
``B u`` at quadrature points, quadrature weights ``W`` and per-pixel Lamé
parameters ``λ, μ``; ``Kʳᵉᶠ`` is the same operator with uniform reference
parameters ``λʳᵉᶠ, μʳᵉᶠ``.

Available preconditioners:

- :class:`JacobiPreconditioner` -- ``M = D⁻¹`` with ``D = diag(A)``.
- :class:`FourierPreconditioner` -- ``M = F⁻¹ k(q) F`` for a scalar symbol
  ``k(q)``, e.g. the inverse symbol of a constant-coefficient operator.
- :class:`BlockFourierPreconditioner` -- ``M = F⁻¹ K̂⁻¹(q) F`` with an
  ``n × n`` block per mode; built by
  :func:`make_reference_stiffness_preconditioner` as the Green
  (reference-material) preconditioner ``G = (Kʳᵉᶠ)⁺``.
- :class:`AnalyticReferencePreconditioner` -- the same ``G``, with the
  symbol ``K̂(q) = Σ_s S[s] exp(-2πi q·s)`` re-evaluated from the ``3^d``
  stencil ``S`` per mode on each apply instead of stored.
- :class:`GreenJacobiPreconditioner` -- ``M = J^{1/2} G J^{1/2}``,
  ``J = diag(K)⁻¹``, for smooth high-contrast material data;
  :func:`make_green_jacobi_preconditioner` assembles it.
- :class:`MultigridReferencePreconditioner` -- a fixed symmetric V-cycle
  approximating ``(Kʳᵉᶠ)⁺`` with damped-Jacobi smoothing and an exact FFT
  solve on the coarsest grid.
- :class:`HybridFourierTridiagonalPreconditioner` -- the exact ``(Kʳᵉᶠ)⁺``
  by FFT along the rank-local axes and a block-tridiagonal (block-Thomas plus
  parallel cyclic reduction) solve along the distributed axis.
"""

import os
import warnings
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

    Subclasses implement :meth:`apply`, which computes ``z = M r`` with
    ``M ≈ A⁻¹``. Instances are callable with the same signature, matching the
    ``prec`` argument of :func:`muGrid.Solvers.conjugate_gradients`.
    """

    def apply(self, r, z):
        """
        Apply the preconditioner: ``z = M r``.

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
    Diagonal (Jacobi) preconditioner, ``z = D⁻¹ r`` with ``D = diag(A)``.

    Useful when the operator's diagonal varies strongly in space, e.g. for
    heterogeneous coefficients: ``D^{-1/2} A D^{-1/2}`` has unit diagonal, and
    among all diagonal scalings it is within a factor of the number of
    non-zeros per row of the optimal condition number (van der Sluis, 1969).
    (For operators with a constant diagonal, such as the plain Laplacian,
    Jacobi only rescales the system and does not change the CG iteration.)
    Cost per apply: one copy and one pointwise scaling, ``O(N)``.

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
    communicator : muGrid.Communicator, optional
        Communicator of the parallel run. With it, the diagonal check below is
        collective, so every rank raises together; without it, a rank whose
        own entries fail raises alone and the others block in their next
        collective operation. Pass it under MPI.

    Raises
    ------
    ValueError
        If any diagonal entry is zero, negative or not finite. CG needs a
        positive definite preconditioner, and the diagonal of a positive
        definite operator is positive: a negative entry means the operator is
        not positive definite, and dividing by it makes the preconditioner
        indefinite.
    """

    def __init__(self, diagonal, name="jacobi-preconditioner",
                 communicator=None):
        self._name = name
        values = getattr(diagonal, "s", None)  # muGrid field?
        if values is not None:
            # Device fields expose cupy views; pull a host copy
            values = values.get() if hasattr(values, "get") else np.array(values)
        else:
            values = np.asarray(diagonal, dtype=float)
        nb_bad = int(np.count_nonzero(~(np.isfinite(values) & (values > 0))))
        if communicator is not None:
            from .Parallel import Communicator

            nb_bad = int(Communicator(communicator).sum(nb_bad))
        if nb_bad > 0:
            raise ValueError(
                "Jacobi preconditioner requires a positive, finite diagonal; "
                f"{nb_bad} entries are zero, negative or not finite"
            )
        self._is_scalar = values.ndim == 0
        self._inverse_diagonal = (
            1.0 / float(values) if self._is_scalar else 1.0 / values
        )
        self._field = None

    def _inverse_diagonal_field(self, z):
        """Field holding D⁻¹ on z's collection (created on first use)."""
        # One field per precision: linalg.scal requires the scaling field and
        # the scaled field to share a dtype, so a float32 solve needs a
        # float32 D⁻¹.
        dtype = np.dtype(z.dtype)
        if self._field is None or np.dtype(self._field.dtype) != dtype:
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
            from .Wrappers import _typed_field

            field = wrap_field(_typed_field(
                z.collection, f"{self._name}-inverse-diagonal-{dtype.name}",
                components, "pixel", dtype,
                {np.dtype(np.float64): "real_field",
                 np.dtype(np.float32): "register_real32_field"},
                "real_field"))
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

    The kernel ``k(q)`` is the Fourier-space representation of ``M``,
    typically the inverse symbol ``1/Â(q)`` of (an approximation to) a
    constant-coefficient system operator. For a translation-invariant
    operator on a periodic grid ``F`` diagonalises ``A``, so with the exact
    inverse symbol ``M = A⁺`` and CG converges in one iteration. For a
    variable-coefficient operator ``A = Bᵀ C(x) B`` preconditioned by the
    inverse of a reference ``Bᵀ Cʳᵉᶠ B`` with
    ``c_min Cʳᵉᶠ ≤ C(x) ≤ c_max Cʳᵉᶠ``, the effective condition number is
    ``κ(M A) ≤ c_max / c_min``, independent of the grid. Cost per apply: one
    forward and one inverse FFT,
    ``O(N log N)`` with ``N = ∏N_α``, plus an ``O(N)`` pointwise product.

    The kernel is applied pointwise on the local Fourier subdomain and
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
    ``Kʳᵉᶠ = Bᵀ W Cʳᵉᶠ B`` built from spatially uniform data is block-circulant,
    hence block-diagonal in Fourier space with one ``n × n`` block per mode
    (``n = d·Nn`` degrees of freedom per pixel, ``Nn`` nodes per pixel), and
    its (pseudo-)inverse is applied mode by mode between a forward and an
    inverse FFT. Cost per apply: two FFTs of ``n`` components,
    ``O(n N log N)``, plus ``O(n² N)`` for the per-mode products; memory
    ``n²`` stored values per Fourier mode (fewer when Hermitian, see below).

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

    Notes
    -----
    If the blocks are Hermitian to within :func:`_hermitian_tol` (decided
    collectively over all ranks), only the real diagonal and the complex
    upper triangle are stored, ``n²`` reals per mode instead of ``2n²``.
    On a device with ``n <=`` :attr:`FUSED_MAX_COMPONENTS` the per-mode
    product is one generated elementwise kernel.
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
        # Component pair at a time: `np.abs(blocks)` and
        # `blocks - conj(swapaxes(blocks))` each build an array the size of the
        # whole symbol (n^2 values per Fourier point, 4.8 GB at 512^3 even in
        # complex64), and the second one is complex. Per pair the temporaries
        # are n^2 times smaller, and only the upper triangle needs the
        # asymmetry check.
        herm_scale = 0.0
        herm_asym = 0.0
        for i in range(n):
            for j in range(n):
                b_ij = blocks[i, j]
                if not b_ij.size:
                    continue
                herm_scale = max(herm_scale, float(np.abs(b_ij).max()))
                if j >= i:
                    herm_asym = max(
                        herm_asym,
                        float(np.abs(b_ij - np.conj(blocks[j, i])).max()),
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
        self._hermitian = herm_scale == 0.0 or herm_asym <= (
            _hermitian_tol(blocks.dtype) * herm_scale)

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

        # Single fused kernel for the per-mode multiply, when the fields live
        # on a device (None on the host, where apply() keeps the loop).
        self._fused, self._fused_blocks = self._build_fused_kernel(n)

    #: Largest `n` for which apply() uses the fused kernel. The fused variant
    #: takes one kernel argument per stored block, so the parameter list grows
    #: as n^2 and eventually runs into the kernel argument-space limit (and
    #: into register pressure) for no gain. The vector-field cases this class
    #: is used for are n = 2 and n = 3.
    FUSED_MAX_COMPONENTS = 4

    def _build_fused_kernel(self, n):
        r"""Generate one elementwise kernel computing the whole per-mode
        product ``z_i(q) = Σ_j K⁻¹_ij(q) r_j(q)``.

        :meth:`apply`'s component-by-component form issues a kernel per term --
        for n = 3 that is nine multiplies, six adds and three copies, each
        reading and writing a full Fourier-sized array. Fusing them makes one
        launch and one pass: the blocks and the input components are read once
        and the result written once.

        Returns ``(None, None)`` on the host (numpy has no equivalent
        facility, and the loop's temporaries are cheap there) or when ``n``
        exceeds :attr:`FUSED_MAX_COMPONENTS`.
        """
        xp = self._xp
        if not xp.__name__.startswith("cupy") or n > self.FUSED_MAX_COMPONENTS:
            return None, None

        if self._hermitian:
            # Real diagonals and the stored upper triangle; the lower triangle
            # is the conjugate, exactly as _block() reconstructs it -- but here
            # the conjugation happens in-register instead of materialising an
            # array.
            params = [f"R d{i}" for i in range(n)]
            params += [f"C u{i}_{j}" for i in range(n) for j in range(i + 1, n)]
            arrays = [self._diag[i] for i in range(n)]
            arrays += [self._off[(i, j)]
                       for i in range(n) for j in range(i + 1, n)]

            def term(i, j):
                if i == j:
                    return f"d{i} * s{j}"
                return f"u{i}_{j} * s{j}" if i < j else f"conj(u{j}_{i}) * s{j}"
        else:
            params = [f"C b{i}_{j}" for i in range(n) for j in range(n)]
            arrays = [self._blocks[i, j] for i in range(n) for j in range(n)]

            def term(i, j):
                return f"b{i}_{j} * s{j}"

        params += [f"C s{i}" for i in range(n)]
        out_params = [f"C o{i}" for i in range(n)]
        # apply() runs in place (the outputs are the inputs), and every output
        # needs every input, so form all n results in registers before writing
        # any of them back.
        body = [f"C t{i} = " + " + ".join(term(i, j) for j in range(n)) + ";"
                for i in range(n)]
        body += [f"o{i} = t{i};" for i in range(n)]
        kernel = xp.ElementwiseKernel(
            ", ".join(params), ", ".join(out_params), "\n".join(body),
            f"mugrid_block_matvec_{n}")
        return kernel, arrays

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
            if self._fused is not None:
                # One launch, one pass: the kernel forms every output in
                # registers, so passing the components as both inputs and
                # outputs updates the work buffer in place.
                comps = [s[i] for i in range(n)]
                self._fused(*self._fused_blocks, *comps, *comps)
            else:
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

    The symmetric split ``M = J^{1/2} G J^{1/2}`` keeps ``M`` symmetric
    positive-definite (``G`` is SPD on the non-rigid-body subspace and
    ``J^{1/2}`` is a positive diagonal), so plain PCG remains valid. Cost per
    apply: one apply of ``G`` plus two ``O(N)`` pointwise scalings; the
    output field doubles as the scratch, so no work field is allocated
    beyond ``J^{1/2}``.

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
    communicator : muGrid.Communicator, optional
        Communicator of the parallel run (e.g. ``engine.communicator``). Used
        only for the degenerate-diagonal diagnostic in
        :meth:`update_diagonal`, which must be evaluated collectively so all
        ranks agree; without it the check sees only the local subdomain.

    Warns
    -----
    RuntimeWarning
        When the diagonal makes Green-Jacobi degenerate to the plain Green
        preconditioner: entirely void (typically material fields that were
        never filled before assembly) or spatially constant (typically a
        stale diagonal after an in-place material update without
        :meth:`update_diagonal` / ``refresh()``).
    """

    def __init__(self, green, diagonal, void_tol=0.0,
                 name="green-jacobi-preconditioner", timer=None,
                 communicator=None):
        self._green = green
        self._name = name
        self._timer = timer
        self._void_tol = float(void_tol)
        if communicator is not None:
            # Accept raw C++ communicators too; the factory wraps them (and
            # passes wrappers through) so the collective reductions below are
            # available.
            from .Parallel import Communicator

            communicator = Communicator(communicator)
        self._communicator = communicator
        self._jhalf = None
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
        self._check_degenerate(vals, positive)
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

    def _check_degenerate(self, vals, positive):
        """Warn when the diagonal makes ``J^{1/2} G J^{1/2}`` degenerate to
        (a scalar multiple of) plain Green: entirely void, or constant with no
        void entries. Both are usually accidents — material fields never
        filled before assembly, or a stale diagonal after an in-place material
        update — and silently forfeit the Jacobi acceleration.

        With a communicator the verdict is formed COLLECTIVELY and
        unconditionally on every rank (empty subdomains contribute reduction
        identities), so all ranks warn — or stay silent — together."""
        has_positive = bool(positive.any())
        has_void = not bool(positive.all())
        # Identity values for ranks holding no positive entries.
        vmax = float(vals[positive].max()) if has_positive else -np.inf
        vmin = float(vals[positive].min()) if has_positive else np.inf
        comm = self._communicator
        if comm is not None:
            has_positive = bool(comm.any(has_positive))
            has_void = bool(comm.any(has_void))
            vmax = float(comm.reduce_max(np.asarray([vmax])))
            vmin = float(comm.reduce_min(np.asarray([vmin])))
        if not has_positive:
            warnings.warn(
                "Green-Jacobi diagonal is entirely void (all entries <= "
                f"void_tol = {self._void_tol}): J^{{1/2}} = 1 everywhere, so "
                "this preconditioner is identical to the plain Green "
                "preconditioner. The material fields were probably not "
                "filled before the diagonal was assembled.",
                RuntimeWarning, stacklevel=3,
            )
        elif not has_void and vmax - vmin <= 1e-12 * abs(vmax):
            warnings.warn(
                "Green-Jacobi diagonal is spatially constant: the "
                "preconditioner only rescales the plain Green preconditioner "
                "and cannot accelerate convergence. If the material was "
                "updated in place, re-assemble the diagonal and call "
                "update_diagonal() (or refresh()).",
                RuntimeWarning, stacklevel=3,
            )

    def apply(self, r, z):
        r"""Compute ``z = J^{1/2} G ( J^{1/2} r )``.

        The output doubles as the scratch for the inner Green apply, which
        needs no separate work field of its own: it consumes its input into
        the Fourier work buffer (``fft(in, work)``) before writing its output
        (``ifft(work, out)``), so it is safe in place.
        """
        with self._timed("scale"):
            linalg.copy(r, z)
            linalg.scal(self._jhalf, z)  # z = J^{1/2} r
        with self._timed("green"):
            self._green.apply(z, z)  # z = G z, in place
        with self._timed("scale"):
            linalg.scal(self._jhalf, z)  # z = J^{1/2} z


#: Working-set budget for the per-mode inversion in
#: :func:`make_reference_stiffness_preconditioner`. The symbol is inverted in
#: double precision one slab of Fourier modes at a time; this bounds the size
#: of the double-precision copy, independent of the grid. Module-level so the
#: tests can shrink it and exercise the multi-slab path -- at any realistic
#: test resolution the whole symbol is one slab.
SYMBOL_INVERSION_SLAB_BYTES = 128 * 1024 * 1024


class AnalyticReferencePreconditioner(Preconditioner):
    r"""``Kʳᵉᶠ⁻¹`` with the symbol *evaluated* per mode, never stored.

    Same operator as :class:`BlockFourierPreconditioner` built by
    :func:`make_reference_stiffness_preconditioner`, reached differently. That
    class holds the inverse symbol: ``n²`` complex values per Fourier mode,
    2.3 GB at 512³ for three components in single precision, growing with the
    grid. A uniform operator is a ``3^dim`` stencil -- 243 numbers in 3D -- so
    the symbol is the closed-form sum ``K̂(q) = Σ_s S[s] exp(-2πi q·s)`` over
    the offsets ``s ∈ {-1, 0, 1}^d`` and can be rebuilt for one mode, inverted
    and applied entirely in registers. Per mode the apply computes

        ``ẑ(q) = engine.normalisation · K̂(q)⁻¹ r̂(q)``  (``ẑ(0) = 0``),

    i.e. ``z = F⁻¹ K̂⁻¹ F r`` with the rigid-body translations at ``q = 0``
    projected out. What this class stores is the stencil.

    The trade is arithmetic for memory. Rebuilding costs ~630 flops per mode
    against ~36 bytes read for the stored symbol, so it wants about 17.6
    flops/byte to break even. A CPU socket is nearer 5, and there the apply is
    about 28% slower (measured at 128³, float32). On a GPU it wins outright:
    measured on an H200 at 512³, three components, the apply is 9.3 ms against
    25.5 ms stored in float32 and 16.1 against 31.2 ms in float64, with 4.9 GB
    less device memory and a 1.3 s setup instead of 168 s. That is why
    :func:`make_reference_stiffness_preconditioner` picks this class on a
    device and the stored symbol on the host.

    In single precision the low-frequency phase sum cancels O(S) terms to
    O(S q²), so a symbol built in float32 alone loses the lowest modes
    (measured: zeroed outright from N ≈ 2048). On the host the symbol is therefore built and
    inverted in double for either field precision; only the mode's components
    are read and written in the field's. On a device a float32 field is
    finished in single precision per mode, and redone in double only where an
    a-posteriori test on the block says the single-precision result is off by
    more than about 1e-4 -- the long wavelengths, about 0.05% of the modes for
    an isotropic medium. That keeps the float32 apply at single-precision speed
    on GPUs with weak double throughput (measured on an L40S at 512³: 45 ms,
    against 87 ms with the symbol in double throughout).

    Modes along the fastest axis share the frequencies of every other axis,
    so the kernel hoists those partial sums out of that loop: the symbol build
    costs 216 flops per mode instead of the 1944 of a direct evaluation.

    Parameters
    ----------
    engine : muGrid.FFTEngine
        Defines the grid, the decomposition and the transforms.
    stencil : ndarray
        ``(3,) * dim + (dim, dim)``, from :func:`reference_stencil`.
    name : str, optional
        Prefix for the engine-managed Fourier work field.
    timer : muTimer.Timer, optional
        When given, :meth:`apply` records ``"fft"``, ``"symbol"`` and
        ``"ifft"``.
    dtype : data-type, optional
        Real-space precision of the fields this will be applied to,
        ``np.float64`` (default) or ``np.float32``.

    Raises
    ------
    ValueError
        If `dtype` is not float32/float64, the stencil shape is not
        ``(3,) * dim + (dim, dim)`` with dim 2 or 3, or the Fourier work
        field is not laid out axis-0-fastest.
    RuntimeError
        If the fields live on a device but this build has no device kernel.
    """

    def __init__(self, engine, stencil, name="analytic-reference-preconditioner",
                 timer=None, dtype=np.float64):
        self._engine = engine
        self._name = name
        self._timer = timer

        real_dtype = np.dtype(dtype)
        if real_dtype == np.dtype(np.float32):
            complex_dtype, suffix = np.dtype(np.complex64), "_f32"
        elif real_dtype == np.dtype(np.float64):
            complex_dtype, suffix = np.dtype(np.complex128), ""
        else:
            raise ValueError(
                f"AnalyticReferencePreconditioner dtype must be float32 or "
                f"float64, got {real_dtype}")

        stencil = np.asarray(stencil)
        dim = stencil.ndim - 2
        if dim not in (2, 3) or stencil.shape != (3,) * dim + (dim, dim):
            raise ValueError(
                f"stencil must have shape (3,)*dim + (dim, dim) with dim 2 or "
                f"3; got {stencil.shape}")
        self._dim = dim
        # Real, and 243 numbers at most -- this is the whole operator. Kept in
        # double for either field precision: the kernels build and invert the
        # symbol in double wherever single precision is not enough, because
        # there the low-frequency phase sum cancels to O(S q^2) and loses the
        # long-wavelength modes this preconditioner exists for (relative error
        # ~ eps * N^2).
        self._stencil = np.ascontiguousarray(stencil.ravel(), dtype=np.float64)

        self._fourier_shape = tuple(engine.nb_fourier_subdomain_grid_pts)
        # One frequency table per axis rather than one frequency per mode: the
        # per-mode form would be 809 MB at 512³ in single precision, which is
        # most of what not storing the symbol saves. Read off the engine's own
        # meshgrid so the convention cannot drift -- muGrid makes axis 0 the
        # half-complex one, unlike numpy.fft.rfftn. Double, like the stencil;
        # the device kernel reads these through raw addresses, so the dtype
        # is load-bearing.
        #
        # A rank can own *no* Fourier modes -- more ranks than planes along the
        # split axis -- and then the meshgrid has a zero-length axis that
        # cannot be indexed to read the others off. Such a rank has nothing to
        # scale; it still takes part in the transforms, which are collective.
        self._nb_modes = int(np.prod(self._fourier_shape))
        q = np.asarray(engine.fftfreq)
        self._q = []
        for d in range(dim):
            if self._nb_modes == 0:
                self._q.append(np.empty(self._fourier_shape[d],
                                        dtype=np.float64))
                continue
            index = [0] * dim
            index[d] = slice(None)
            self._q.append(
                np.ascontiguousarray(q[d][tuple(index)], dtype=np.float64))

        self._work = engine.fourier_space_field(
            f"{name}-work", components=(dim,), dtype=complex_dtype)
        self._normalisation = float(engine.normalisation)

        # The work field's view says where the transforms leave the data: a
        # cupy array means the device kernel, which takes device addresses.
        # The frequency tables go to the device once, here -- a few kilobytes,
        # not worth a transfer per apply. The stencil stays on the host; the
        # kernel receives it by value, in constant memory.
        sample = self._work.s
        self._on_device = type(sample).__module__.startswith("cupy")
        if self._on_device:
            import cupy

            self._kernel = getattr(
                linalg, f"apply_green_symbol_gpu_{dim}d{suffix}", None)
            if self._kernel is None:
                raise RuntimeError(
                    "the work field lives on a device but this muGrid build "
                    "has no device kernel for the evaluated symbol")
            self._q_device = [cupy.asarray(qd) for qd in self._q]
            self._q_ptrs = [qd.data.ptr for qd in self._q_device]
        else:
            self._kernel = getattr(linalg, f"apply_green_symbol_{dim}d{suffix}")

        # Element strides of the component and first Fourier axes, read off
        # the view rather than assumed. The host buffer is array-of-structures
        # (components fastest, strides 1 and dim) and the device buffer is
        # structure-of-arrays (strides nb_modes and 1); both must have the
        # Fourier axes Fortran-ordered, axis 0 fastest, which is how the kernel
        # decomposes the flat mode index.
        itemsize = sample.dtype.itemsize
        shape, strides = sample.shape, sample.strides
        if shape[1] != 1:
            raise ValueError(
                f"expected one sub-point per Fourier mode, got {shape[1]}")
        self._stride_component = strides[0] // itemsize
        self._stride_mode = strides[2] // itemsize
        for d in range(dim - 1):
            if self._nb_modes and strides[3 + d] != strides[2 + d] * shape[2 + d]:
                raise ValueError(
                    f"Fourier work field is not axis-0-fastest over its "
                    f"modes: shape {shape}, strides {strides}")

    def _timed(self, label):
        return self._timer(label) if self._timer is not None else nullcontext()

    def apply(self, r, z):
        engine = self._engine
        with self._timed("fft"):
            engine.fft(r, self._work)
        with self._timed("symbol"):
            if self._nb_modes == 0:
                # No local modes; the transforms above and below are still
                # collective and must not be skipped.
                pass
            elif self._on_device:
                # In place through the device address; strides as read off
                # the view in __init__ (structure-of-arrays on a device).
                # The kernel takes raw addresses and runs on the current
                # device, which muGrid's own calls restore to the caller's:
                # select the buffer's.
                import cupy

                work = self._work.s
                with cupy.cuda.Device(work.device.id):
                    self._kernel(work.data.ptr, self._stride_component,
                                 self._stride_mode, self._stencil,
                                 self._q_ptrs, list(self._fourier_shape),
                                 self._normalisation)
            else:
                # The Fourier buffer is Fortran-ordered, so an order="F"
                # ravel is a contiguous *view*: components fastest, then modes
                # with axis 0 fastest, exactly the layout the kernel indexes.
                # Nothing is copied and the kernel writes back through it.
                work = np.asarray(self._work.s)
                flat = work.ravel(order="F")
                # A copy here would take the result and leave the field
                # unchanged without an error.
                assert np.shares_memory(flat, work)
                self._kernel(flat, self._stride_component, self._stride_mode,
                             self._stencil, self._q,
                             list(self._fourier_shape), self._normalisation)
        with self._timed("ifft"):
            engine.ifft(self._work, z)
        return z


#: Relative asymmetry below which a symbol assembled and inverted in double
#: is taken for Hermitian (and stored as its upper triangle).
_HERMITIAN_TOL_DOUBLE = 1e-10


def _hermitian_tol(cdtype):
    """Relative asymmetry that counts as rounding for blocks of `cdtype`:
    ``max(1e-10, 100 eps)``, i.e. 1e-10 in double and 100 eps in single
    precision, where the rounding of the entries alone exceeds 1e-10."""
    return max(_HERMITIAN_TOL_DOUBLE, 100 * float(np.finfo(cdtype).eps))


def _invert_symbol_slab(blocks, singular, normalisation, hermitian_tol):
    """Invert a slab ``[..., n, n]`` of the symbol in double, in place.

    The inversion is batched over the slab, never the whole symbol: at once it
    would hold the double copy, np.linalg.inv's output and the masked gather
    and scatter together, four or five symbol-sized arrays (~40 GB at 512^3).
    The singular (zero-mode) blocks are made the identity for the inversion
    and zeroed afterwards, the pseudo-inverse of the rigid-body null space.

    A slab Hermitian to within `hermitian_tol` is made exactly Hermitian: its
    entries and their conjugate partners then round to the same values when
    cast to the solve precision, so BlockFourierPreconditioner can store the
    upper triangle. Without the symmetrisation, single-precision rounding of
    a slightly asymmetric double inverse fails BlockFourierPreconditioner's
    Hermitian test and forces the dense block.
    """
    n = blocks.shape[-1]
    blocks[singular] = np.eye(n, dtype=blocks.dtype)
    blocks = np.linalg.inv(blocks)
    # Fold in the inverse-transform normalisation while the slab is hot,
    # instead of scaling the whole symbol afterwards (another copy).
    blocks *= normalisation
    blocks[singular] = 0.0
    adjoint = np.conj(np.swapaxes(blocks, -1, -2))
    scale = float(np.abs(blocks).max()) if blocks.size else 0.0
    if scale == 0.0 or float(np.abs(blocks - adjoint).max()) <= (
            hermitian_tol * scale):
        blocks += adjoint
        blocks *= 0.5
    return blocks


def make_reference_stiffness_preconditioner(
    engine,
    apply_reference_stiffness=None,
    nb_components=None,
    name="reference-stiffness-preconditioner",
    timer=None,
    dtype=np.float64,
    element=None,
    grid_spacing=None,
    lambda_ref=None,
    mu_ref=None,
    evaluate_symbol=None,
):
    """
    Build the reference-material (Green's-function) preconditioner of Ladecký et
    al., Appl. Math. Comput. 446 (2023) 127835.

    For an FE problem on a regular periodic grid, the reference-material
    stiffness ``Kʳᵉᶠ = Bᵀ W Cʳᵉᶠ B`` built from spatially *uniform* data is
    block-circulant — every pixel carries the same stencil — hence
    block-diagonal in Fourier space, with one ``n × n`` block ``K̂(q)`` per mode
    (``n`` degrees of freedom per pixel). The returned preconditioner applies

        ``G r = F⁻¹ [ K̂(q)⁺ · F r ]``,

    where ``K̂(q)⁺ = K̂(q)⁻¹`` for ``q ≠ 0`` and ``K̂(0)⁺ = 0``: the singular
    zero-frequency block (the rigid-body translations) is replaced by its
    pseudo-inverse, which projects those modes out — consistent with a
    rigid-body-free right-hand side. For material data with
    ``c_min Cʳᵉᶠ ≤ C(x) ≤ c_max Cʳᵉᶠ``, ``κ(G K) ≤ c_max / c_min``
    independently of the grid.

    ``K̂(q)`` is obtained by one of two routes:

    * **Impulse response** (paper Algorithm 2), when
      ``apply_reference_stiffness`` is given: ``Kʳᵉᶠ`` is applied to a unit
      nodal impulse in each of the ``n`` directions placed at the global
      origin pixel, and the FFT of the response is the ``β``-th column of the
      symbol. FE-agnostic -- only the action of ``Kʳᵉᶠ`` is needed -- at the
      cost of ``n`` operator applies and ``n`` full-grid FFTs at setup.
    * **Analytic**, when ``element``, ``grid_spacing``, ``lambda_ref`` and
      ``mu_ref`` name the operator: ``K̂(q) = Σ_s S[s] exp(-2πi q·s)`` from its
      ``3^d`` stencil ``S`` (:func:`reference_stencil`,
      :func:`stencil_symbol`), with no setup transforms. The symbol is
      assembled in double precision because the stencil rows sum to zero, so
      at low ``q`` the sum cancels ``O(S)`` terms to ``O(S q²)`` and a
      single-precision assembly carries a relative error of about
      ``eps N²`` (measured: 3e-4 at N = 256, 0.13 at N = 4096).

    Each block is inverted in double precision, a slab of Fourier modes at a
    time (at most :data:`SYMBOL_INVERSION_SLAB_BYTES`), the inverse-transform
    normalisation is folded in, and the result is stored at the solve
    precision in a :class:`BlockFourierPreconditioner`; with
    ``evaluate_symbol`` an :class:`AnalyticReferencePreconditioner` is
    returned instead, which stores nothing per mode.

    Parameters
    ----------
    engine : muGrid.FFTEngine
        FFT engine; its real-space collection holds the fields, and it provides
        the transforms and the parallel (MPI) decomposition. The impulse
        assembly and the per-mode inverse are computed on the rank-local Fourier
        subdomain, so this is MPI-transparent.
    apply_reference_stiffness : callable, optional
        ``apply_reference_stiffness(u, f)`` computing ``f = Kʳᵉᶠ u`` for
        ``n``-component real fields ``u``, ``f`` on the engine's real-space
        collection (it may use ghost communication internally). Required
        unless ``element`` is given; ignored when it is.
    nb_components : int, optional
        Degrees of freedom per pixel ``n``. Default: the dimension of the
        engine (one node per pixel).
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
        double wherever single precision would lose accuracy.
    element : muGrid.FEMElement, optional
        Names the reference operator instead of passing
        ``apply_reference_stiffness`` (analytic route): the symbol is then
        built from its ``3^dim`` stencil (:func:`reference_stencil`), with no
        impulse response and no setup transforms. Requires the next three.
    grid_spacing : sequence of float, optional
        Grid spacing ``h_α`` per direction (analytic route).
    lambda_ref, mu_ref : float, optional
        Uniform reference Lamé parameters (analytic route).
    evaluate_symbol : bool, optional
        Analytic route only. ``True`` returns an
        :class:`AnalyticReferencePreconditioner`, which keeps only the stencil
        and rebuilds the symbol per mode on every apply; ``False`` assembles and
        stores the inverse symbol (``n²`` complex values per Fourier mode,
        2.3 GB at 512³ in single precision). The default, ``None``, evaluates
        when the engine's fields live on a device and stores on the host: on
        a GPU evaluating is both faster and smaller than storing, on a CPU it
        is ~28% slower (see :class:`AnalyticReferencePreconditioner`).

    Returns
    -------
    BlockFourierPreconditioner or AnalyticReferencePreconditioner
        Ready to pass as ``prec=`` to
        :func:`muGrid.Solvers.conjugate_gradients`.

    Raises
    ------
    ValueError
        If neither ``apply_reference_stiffness`` nor ``element`` is given;
        if ``element`` is given without ``grid_spacing``, ``lambda_ref`` and
        ``mu_ref``; if ``evaluate_symbol`` is true without ``element``; if
        `dtype` is not float32/float64; or if the stencil's component count
        differs from ``nb_components``.
    """
    fourier_shape = tuple(engine.nb_fourier_subdomain_grid_pts)
    dim = len(fourier_shape)
    if nb_components is None:
        nb_components = dim
    n = int(nb_components)
    if element is None and apply_reference_stiffness is None:
        raise ValueError(
            "make_reference_stiffness_preconditioner needs either "
            "`apply_reference_stiffness` (the general route) or `element` "
            "with `grid_spacing`, `lambda_ref` and `mu_ref` (the analytic "
            "route, which needs no impulse response and no full-grid FFT)")
    if element is not None and None in (grid_spacing, lambda_ref, mu_ref):
        raise ValueError(
            "the analytic route needs `grid_spacing`, `lambda_ref` and "
            "`mu_ref` alongside `element`")
    if evaluate_symbol and element is None:
        raise ValueError(
            "evaluate_symbol needs the analytic route: pass `element` with "
            "`grid_spacing`, `lambda_ref` and `mu_ref`")

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

    # Two routes to the same symbol.
    #
    # Analytic: the caller named the operator (element, spacing, Lame), so it
    # is a 3^dim stencil -- 243 numbers in 3D -- and the symbol is the
    # closed-form sum over its offsets. No impulse, no full-grid FFT, no
    # device->host round trip, and the assembly is per-mode so it streams in
    # slabs.
    #
    # Impulse: the caller handed over an opaque callable. Recovering the same
    # information then means one impulse response per component on the *full*
    # grid, each followed by a full-grid FFT. It is the only route that works
    # for an operator muGrid cannot name.
    analytic = element is not None
    if analytic:
        S = reference_stencil(dim, grid_spacing, element, lambda_ref, mu_ref)
        if S.shape[-1] != n:
            raise ValueError(
                f"reference stencil has {S.shape[-1]} components but the "
                f"preconditioner was asked for {n}")
        if evaluate_symbol is None:
            evaluate_symbol = not _engine_on_host(engine)
        if evaluate_symbol:
            # Decided before K_hat below is allocated: the point is that the
            # full-size symbol never exists, not even during setup.
            return AnalyticReferencePreconditioner(
                engine, S, name=name, timer=timer, dtype=real_dtype)

    impulse_name = f"{name}-impulse"
    column_name = f"{name}-column"
    column_hat_name = f"{name}-column-hat"

    # K_hat[q, alpha, beta], at the *solve* precision and in [*fourier, n, n]
    # layout. Both matter at scale: the symbol is n^2 values per Fourier point,
    # 4.5 GB at 512^3 in complex64 and twice that in complex128, and in single
    # precision the extra width buys nothing -- only the per-mode n x n
    # inversion below gains from double, and that is done in double on one slab
    # at a time. [*fourier, n, n] is what np.linalg.inv batches over, so it also
    # avoids a full-size transposed copy.
    K_hat = np.zeros(fourier_shape + (n, n), dtype=complex_dtype)

    fourier_q = np.asarray(engine.fftfreq)  # [dim, *fourier]
    # Slab bound shared by assembly and inversion; see the inversion comment.
    bytes_per_index = int(np.prod(fourier_shape[1:], dtype=np.int64)) * n * n * 16
    slab = max(1, SYMBOL_INVERSION_SLAB_BYTES // max(bytes_per_index, 1))

    # The q = 0 block is singular: the reference operator has the rigid-body
    # translations in its null space. Invert everything else and project it out.
    zero_mode = np.ones(fourier_shape, dtype=bool)
    for d in range(dim):
        zero_mode &= fourier_q[d] == 0.0

    if analytic:
        # Built *and inverted* a slab at a time, in double, whatever the solve
        # precision; only the inverse is stored at it. The stencil's rows sum
        # to zero, so at low frequency the phase sum cancels O(S) terms down to
        # O(S q^2): assembled in single precision the lowest modes carried a
        # relative error of about eps * N^2 (3e-4 at N = 256, 0.13 at 4096),
        # and inverting that afterwards in double could not recover it. Slabs
        # also bound stencil_symbol's 3^dim transients of the mode shape.
        for lo in range(0, fourier_shape[0], slab):
            hi = min(lo + slab, fourier_shape[0])
            K_hat[lo:hi] = _invert_symbol_slab(
                stencil_symbol(S, [fq[lo:hi] for fq in fourier_q],
                               cdtype=np.complex128),
                zero_mode[lo:hi], engine.normalisation,
                hermitian_tol=_HERMITIAN_TOL_DOUBLE).astype(
                    complex_dtype, copy=False)
    else:
        # Global-origin pixel(s) in this rank's interior: the nodal coordinate is
        # exactly 0 in every direction only at global index 0 (coord = index / N).
        # In MPI only the rank owning the origin matches; the others contribute no
        # impulse, which is correct for the global impulse response.
        coords = np.asarray(engine.coords)  # [dim, *local_grid]
        origin_mask = np.ones(coords.shape[1:], dtype=bool)
        for d in range(dim):
            origin_mask &= coords[d] == 0.0

        impulse = engine.real_space_field(
            impulse_name, components=(n,), dtype=real_dtype)
        column = engine.real_space_field(
            column_name, components=(n,), dtype=real_dtype)
        column_hat = engine.fourier_space_field(
            column_hat_name, components=(n,), dtype=complex_dtype)

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
            K_hat[..., :, beta] = np.moveaxis(ch, 0, -1)

        # The impulse response is the operator's own output, so it carries
        # the operator's precision; invert it in double, a slab at a time, in
        # place. It is Hermitian only to within that precision.
        for lo in range(0, fourier_shape[0], slab):
            hi = min(lo + slab, fourier_shape[0])
            K_hat[lo:hi] = _invert_symbol_slab(
                K_hat[lo:hi].astype(np.complex128), zero_mode[lo:hi],
                engine.normalisation,
                hermitian_tol=_hermitian_tol(complex_dtype)).astype(
                    complex_dtype, copy=False)

    # [n, n, *fourier], the layout BlockFourierPreconditioner expects. moveaxis
    # returns a view, so this costs nothing.
    K_inv = np.moveaxis(K_hat, (-2, -1), (0, 1))

    # Release the impulse-response scratch. These three fields (two real, one
    # Fourier) were only needed to assemble the symbol above; left in the
    # engine's collections they would persist through the entire solve. The
    # symbol lives in K_inv (a plain array, copied onto the device inside
    # the preconditioner), so the only Fourier buffer the solve then needs is
    # the preconditioner's own work field. Freeing here drops ~3 vector-sized
    # buffers from the resident set during the CG iteration.
    if not analytic:
        engine.real_space_collection.pop_field(impulse_name)
        engine.real_space_collection.pop_field(column_name)
        engine.fourier_space_collection.pop_field(column_hat_name)
        del impulse, column, column_hat

    return BlockFourierPreconditioner(
        engine, K_inv, name=name, timer=timer, dtype=real_dtype)


def _engine_on_host(engine):
    """Whether the engine's fields live in host memory."""
    device = getattr(engine, "device", None)
    return device is None or device.is_host


def make_green_jacobi_preconditioner(
    engine,
    stiffness_op,
    lambda_field,
    mu_field,
    nb_components,
    reference_lambda=None,
    reference_mu=None,
    element=None,
    grid_spacing=None,
    void_tol=0.0,
    name="green-jacobi-preconditioner",
    timer=None,
    dtype=None,
    evaluate_symbol=None,
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
    ``lambda_field`` / ``mu_field``. A common scale factor of the reference
    does not affect CG (it scales ``M`` by a constant, which leaves the
    iterates unchanged); only the ratio ``λʳᵉᶠ/μʳᵉᶠ`` matters.

    .. TODO: rationale for the volume mean as default reference ratio?

    The returned
    preconditioner exposes :meth:`GreenJacobiPreconditioner.update_diagonal`
    (and the convenience ``refresh()`` attached to it) to recompute the Jacobi part
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
        Uniform reference Lamé parameters for the Green part. Default: the
        global (MPI-reduced) means of ``lambda_field`` / ``mu_field``, so the
        reference — and hence the assembled Green symbol — is identical on
        every rank regardless of the domain decomposition.
    element : muGrid.FEMElement, optional
        Element of ``stiffness_op``. Together with ``grid_spacing`` it selects
        the analytic (stencil-based) assembly of the Green part in
        :func:`make_reference_stiffness_preconditioner`; without them the
        impulse-response route is used, since ``stiffness_op`` does not
        expose its element or spacing.
    grid_spacing : sequence of float, optional
        Grid spacing ``h_α`` of ``stiffness_op``; needed with ``element``.
    void_tol : float, optional
        Passed to :class:`GreenJacobiPreconditioner`.
    name : str, optional
        Prefix for the managed fields.
    timer : muTimer.Timer, optional
        Forwarded to both sub-preconditioners.
    dtype : data-type, optional
        Real-space precision of the preconditioner (``np.float32`` or
        ``np.float64``): sets the dtype of the Jacobi diagonal and the inner
        Green preconditioner's fields so their transforms pair with the solver's
        fields. Defaults to the dtype of ``lambda_field``, so a single-precision
        material yields a single-precision preconditioner automatically.
    evaluate_symbol : bool, optional
        Forwarded to :func:`make_reference_stiffness_preconditioner`; only
        meaningful with ``element``. By default the Green symbol is evaluated
        per mode on a device and stored on the host.

    Returns
    -------
    GreenJacobiPreconditioner
        Ready to pass as ``prec=`` to
        :func:`muGrid.Solvers.conjugate_gradients`, with a ``refresh()`` method
        bound for in-place material updates.
    """
    n = int(nb_components)

    # Match the Jacobi diagonal and the inner Green preconditioner to the
    # precision of the material fields (the FFT engine and fused operators
    # dispatch on the field dtype). Inferred from ``lambda_field`` when not
    # given, so a single-precision (float32) material yields a single-precision
    # preconditioner and a float64 material a double-precision one.
    if dtype is None:
        lam_p = lambda_field.s
        dtype = (lam_p.get() if hasattr(lam_p, "get")
                 else np.asarray(lam_p)).dtype

    # Global (count-weighted, MPI-reduced) means so the reference stiffness is
    # deterministic under domain decomposition; reduce_mean handles host and
    # device views alike. Engines without a communicator (serial stand-ins)
    # fall back to the local mean, which is then the global one.
    comm = getattr(engine, "communicator", None)

    def _global_mean(field):
        s = field.s
        if comm is not None:
            return float(comm.reduce_mean(s))
        return float((s.get() if hasattr(s, "get") else np.asarray(s)).mean())

    if reference_lambda is None:
        reference_lambda = _global_mean(lambda_field)
    if reference_mu is None:
        reference_mu = _global_mean(mu_field)

    def apply_reference_stiffness(u, f):
        engine.communicate_ghosts(u)
        stiffness_op.apply_uniform(u, reference_lambda, reference_mu, f)

    # Name the operator when the caller can: `element` + `grid_spacing` let the
    # Green part be assembled from its 3^dim stencil instead of by impulse
    # response plus n full-grid FFTs. Same symbol to round-off, but no setup
    # transforms and no device->host round trip. Without them the general
    # impulse route still applies -- `stiffness_op` does not expose its element
    # or spacing, so they cannot be recovered here.
    green = make_reference_stiffness_preconditioner(
        engine, apply_reference_stiffness, n, name=f"{name}-green",
        timer=timer, dtype=dtype, element=element, grid_spacing=grid_spacing,
        lambda_ref=reference_lambda if element is not None else None,
        mu_ref=reference_mu if element is not None else None,
        evaluate_symbol=evaluate_symbol,
    )

    diagonal = engine.real_space_field(
        f"{name}-diagonal", components=(n,), dtype=dtype)
    stiffness_op.assemble_diagonal(lambda_field, mu_field, diagonal)

    prec = GreenJacobiPreconditioner(
        green, diagonal, void_tol=void_tol, name=name, timer=timer,
        communicator=comm,
    )

    # Keep the ingredients so the Jacobi part can be recomputed in place when
    # the material changes (the Green part is reference-only and stays fixed).
    prec._stiffness_op = stiffness_op
    prec._lambda_field = lambda_field
    prec._mu_field = mu_field
    prec._diagonal = diagonal
    # Introspection (and MPI-determinism tests): the reference the Green
    # symbol was built from.
    prec._reference_lambda = reference_lambda
    prec._reference_mu = reference_mu

    def refresh():
        """Re-assemble diag(K) from the (updated) material fields and refresh
        J^{1/2}. Call after changing ``lambda_field`` / ``mu_field`` in place."""
        stiffness_op.assemble_diagonal(
            prec._lambda_field, prec._mu_field, prec._diagonal
        )
        prec.update_diagonal(prec._diagonal)

    prec.refresh = refresh
    return prec


def _project_constants_out(field):
    """Remove the nullspace of the periodic stiffness operator: the ``dim``
    constant translations. (Rigid rotations are not periodic, so they are not
    in it.)

    Done on the array view because ``linalg`` has no interior-only sum and no
    add-a-constant kernel -- the one place this preconditioner touches an
    array library in its hot path. Two calls per apply, not per level.
    """
    values = field.s
    spatial = tuple(range(1, values.ndim))
    values -= values.mean(axis=spatial, keepdims=True)


# --------------------------------------------------------------------------- #
# Multigrid approximation of the reference-stiffness inverse
# --------------------------------------------------------------------------- #


class _MultigridLevel:
    """One level of the hierarchy, carrying the *uniform* reference operator.

    Every level runs the same spatially uniform ``Kʳᵉᶠ`` at its own grid
    spacing, so no material field is ever restricted and the coarse operator is
    a plain rediscretisation rather than a Galerkin product. Heterogeneity is
    handled outside the cycle, by the symmetric Jacobi scaling of
    :class:`GreenJacobiPreconditioner`.
    """

    def __init__(self, decomposition, spacing, element, lam, mu, dim,
                 name, dtype, with_fft=False):
        from .Wrappers import IsotropicStiffnessOperator

        self.dim = dim
        self.decomp = decomposition
        self.lam = lam
        self.mu = mu
        self.op = IsotropicStiffnessOperator(dim, tuple(spacing), element)
        self.fc = (decomposition.real_space_collection if with_fft
                   else decomposition.collection)

        def field(suffix):
            if np.dtype(dtype) == np.dtype(np.float32):
                return wrap_field(self.fc.register_real32_field(
                    f"{name}-{suffix}", (dim,)))
            return wrap_field(self.fc.real_field(f"{name}-{suffix}", (dim,)))

        self.r = field("r")
        self.z = field("z")
        self.t = field("t")
        self.inv_diag = field("inv-diag")

        self.node_block = self._probe_node_block()
        self._set_inverse_diagonal(name)

    # -- operator ---------------------------------------------------------- #

    def apply(self, u, f):
        """``f = Kʳᵉᶠ u``, ghosts of ``u`` refreshed first."""
        self.decomp.communicate_ghosts(u)
        self.op.apply_uniform(u, self.lam, self.mu, f)

    # -- setup -------------------------------------------------------------- #

    def _probe_node_block(self):
        """The nodal ``dim x dim`` block of the uniform operator.

        On a uniform grid every node has an identical element neighbourhood, so
        one matrix describes the whole level and a single impulse response per
        direction recovers it. Rank-local by construction: every rank probes
        its own interior and gets the same answer, so this needs no
        communication.
        """
        interior = tuple(self.decomp.nb_subdomain_grid_pts)
        node = tuple(n // 2 for n in interior)
        block = np.zeros((self.dim, self.dim))
        for beta in range(self.dim):
            self.z.set_zero()
            self.z.s[(beta, 0) + node] = 1.0
            self.apply(self.z, self.t)
            response = self.t.s[(slice(None), 0) + node]
            block[:, beta] = (response.get() if hasattr(response, "get")
                              else np.asarray(response))
        self.z.set_zero()
        self.t.set_zero()
        return block

    def _set_inverse_diagonal(self, name):
        """Store ``D⁻¹`` as a per-component field.

        For Q1 in any dimension, and for P1 in 3D, the nodal block is diagonal
        (``c·I`` at isotropic grid spacing), so the smoother is a component-wise
        scaling that ``linalg.scal`` applies directly. 2D P1 is the exception:
        the two-triangle Kuhn split breaks the x-y symmetry that cancels the
        off-diagonal, leaving ``K01 = λ + μ``. That needs a component-mixing
        smoother, which is not implemented.
        """
        off_diagonal = np.abs(
            self.node_block - np.diag(np.diag(self.node_block))).max()
        scale = np.abs(np.diag(self.node_block)).max()
        if off_diagonal > 1e-10 * max(scale, 1.0):
            raise NotImplementedError(
                "the multigrid smoother needs a diagonal nodal block, but the "
                f"probed block is\n{self.node_block}\n(largest off-diagonal "
                f"{off_diagonal:.3e}). This happens for P1 elements in 2D, "
                "whose two-triangle split breaks the symmetry that cancels "
                "the off-diagonal term; use Q1, or 3D."
            )
        diagonal = np.diag(self.node_block)
        if not (np.abs(diagonal) > 0).all():
            raise ValueError(
                f"the uniform operator has a zero nodal diagonal: {diagonal}")
        self.inv_diag.set_zero()
        values = np.broadcast_to(
            (1.0 / diagonal).reshape((self.dim,) + (1,) * (self.dim + 1)),
            self.inv_diag.s.shape)
        _fill_field(self.inv_diag, values)

    #: Relative change between power iterations at which lambda_max stops.
    LAMBDA_MAX_RTOL = 1e-4

    def lambda_max(self, communicator=None, nb_it=500):
        """Largest eigenvalue of ``D⁻¹K`` by power iteration.

        ``D⁻¹K`` is invariant under uniform refinement — ``D`` and ``K`` carry
        the same power of ``h`` — so one level's estimate serves the whole
        hierarchy, and the coarsest is the cheapest place to measure it.
        """
        rng = np.random.default_rng(0)
        _fill_field(self.z, rng.standard_normal(self.z.s.shape))

        def norm(field):
            local = linalg.norm_sq(field)
            if communicator is not None:
                local = float(communicator.sum(float(local)))
            return np.sqrt(local)

        linalg.scal(1.0 / norm(self.z), self.z)
        eigenvalue = 0.0
        change = np.inf
        for _ in range(nb_it):
            self.apply(self.z, self.t)
            linalg.scal(self.inv_diag, self.t)
            previous, eigenvalue = eigenvalue, norm(self.t)
            linalg.copy(self.t, self.z)
            linalg.scal(1.0 / eigenvalue, self.z)
            change = abs(eigenvalue - previous) / eigenvalue
            if change < self.LAMBDA_MAX_RTOL:
                break
        else:
            # Power iteration approaches lambda_max from below, and the damping
            # SAFETY / lambda is only safe while it stays below 2 / lambda_max:
            # an estimate low by more than 15% makes the smoother diverge.
            warnings.warn(
                f"lambda_max: power iteration did not converge in {nb_it} "
                f"iterations (last relative change {change:.1e}); the "
                "smoother damping may be too large. Pass omega explicitly or "
                "raise nb_it.", RuntimeWarning, stacklevel=2)
        self.z.set_zero()
        self.t.set_zero()
        return eigenvalue

    # -- smoother ----------------------------------------------------------- #

    def smooth(self, nb_steps, omega):
        """``nb_steps`` damped-Jacobi sweeps of ``z += ω D⁻¹ (r - K z)``."""
        for _ in range(nb_steps):
            self.apply(self.z, self.t)
            linalg.axpby(1.0, self.r, -1.0, self.t)   # t = r - K z
            linalg.scal(self.inv_diag, self.t)        # t = D⁻¹ t
            linalg.axpy(omega, self.t, self.z)        # z += ω t


class MultigridReferencePreconditioner(Preconditioner):
    r"""Multigrid approximation of ``Kʳᵉᶠ⁻¹``, with an exact FFT solve at the
    coarsest level.

    A drop-in replacement for :func:`make_reference_stiffness_preconditioner`
    that trades the fine-grid FFT for a V-cycle. The motivation is parallel
    scaling: an FFT-based apply costs two all-to-all transposes per transform
    (four per apply), each a full barrier across all ranks, and it forces the
    solver onto the FFT engine's slab decomposition, ``[1, 1, P]``, in which
    only the last axis is ever distributed -- so every rank holds the full
    extent of the other two, and no run can use more ranks than the grid has
    planes. A V-cycle needs only nearest-neighbour halo exchange at every level
    plus one small collective at the bottom, and it leaves the solver free to
    use a genuine 3D Cartesian decomposition.

    On a single shared-memory node the slab is not itself a cost -- it moves
    more halo bytes but through 2 neighbours rather than 6, and latency wins
    (measured on one MI300A node at 128³ with
    ``examples/decomposition_baseline.py``: the slab's halo exchange took
    0.75-0.90x the time of a 3D Cartesian split at 2-16 ranks while moving up
    to 2.23x the bytes). Its
    liability is the rank ceiling and the halo volume, both of which bite
    only where messages stop being latency-bound.

    One V-cycle on level ``l`` (grid spacing ``2^l h``, operator ``K_l`` the
    uniform reference operator rediscretised there, ``D_l`` its diagonal)
    applied to ``r_l`` computes, starting from ``z_l = 0``:

    1. ``ν`` damped-Jacobi pre-smoothing sweeps
       ``z_l ← z_l + ω D_l⁻¹ (r_l - K_l z_l)``;
    2. restriction of the residual, ``r_{l+1} = R (r_l - K_l z_l)``, with
       ``R = Pᵀ`` the adjoint of multilinear prolongation ``P``
       (:class:`muGrid.GridTransfer`);
    3. the coarse correction ``z_{l+1}`` by a recursive V-cycle, or on the
       coarsest level exactly by the block-Fourier pseudo-inverse of
       :func:`make_reference_stiffness_preconditioner`;
    4. ``z_l ← z_l + P z_{l+1}``;
    5. ``ν`` post-smoothing sweeps as in 1.

    The constant translations (the null space of the periodic ``Kʳᵉᶠ``) are
    removed from ``r`` before and from ``z`` after the cycle. Damped Jacobi is
    stable for ``0 < ω < 2/λ_max(D⁻¹K)``; ``ω`` is set from a power-iteration
    estimate of ``λ_max`` on the coarsest level, which equals that of every
    level because ``D`` and ``K`` scale with the same power of ``h``. Cost per
    apply: ``O(N)`` operator applies (``2ν + 1`` per level, geometric in the
    level size) plus one small FFT solve; communication is nearest-neighbour
    halo exchange per level.

    The cycle runs on the **uniform** reference operator only, so no material
    field is ever restricted and every level is a plain rediscretisation of
    ``Kʳᵉᶠ`` at its own grid spacing. Heterogeneity belongs outside: wrap this
    in :class:`GreenJacobiPreconditioner` exactly as you would wrap the FFT
    version, by passing it as the ``green`` argument.

    The preconditioner is a **fixed** linear operator — a fixed cycle count and
    a fixed number of smoothing steps, with symmetric pre- and post-smoothing
    and ``R = Pᵀ`` — because plain CG requires ``M`` to be symmetric positive
    definite. Do not replace the cycle count by an inner convergence test; that
    would make the preconditioner non-linear and require a flexible Krylov
    method.

    Parameters
    ----------
    decomposition : muGrid.CartesianDecomposition
        The solver's (fine) decomposition. Its collection must be the one the
        fields passed to :meth:`apply` live on, and it needs one ghost layer
        per side — which the stiffness stencil already requires.
    grid_spacing : sequence of float
        Fine-level grid spacing, one entry per direction.
    lambda_ref, mu_ref : float
        Uniform reference Lamé parameters. The volume means of the actual
        material are the usual choice, as in ``examples/homogenization.py``.
    communicator : muGrid.Communicator, optional
        Communicator of the parallel run. The cycle is serial: a communicator
        of size > 1, or a decomposition with more than one subdivision,
        raises :class:`NotImplementedError`. A distributed cycle requires
        nested power-of-two subdivisions pinned across levels and a redundant
        coarsest level.
    element : muGrid.FEMElement, optional
        Finite element, default Q1. P1 in 2D is rejected by the smoother (see
        :class:`_MultigridLevel`).
    nb_levels : int, optional
        Number of levels including the coarsest. Default: coarsen while every
        direction stays even and at least ``min_coarse`` points wide.
    min_coarse : int, optional
        Smallest coarse grid to coarsen towards. Convergence is insensitive to
        this over a wide range, so the choice is a parallel one; default 32.
    nu : int, optional
        Pre- and post-smoothing steps per level (equal, to keep ``M``
        symmetric). Default 2.
    omega : float, optional
        Jacobi damping. Default ``1.7 / λ_max(D⁻¹K)``, measured by power
        iteration at setup. A hardcoded value is unsafe: the stability limit is
        ``2/λ_max`` and ``λ_max`` moves with dimension and element kind, so a
        value that is optimal in 2D diverges in 3D.
    nb_cycles : int, optional
        V-cycles per apply, default 1. Raise it (never the smoothing count
        alone) if the cycle turns out too weak; it stays linear and symmetric.
    timer : muTimer.Timer, optional
        When given, :meth:`apply` records ``"vcycle"`` and ``"coarse"``.
    dtype : data-type, optional
        Precision of the level work fields, ``np.float64`` (default) or
        ``np.float32``.
    name : str, optional
        Prefix for the per-level work fields.

    Raises
    ------
    NotImplementedError
        For a distributed decomposition or communicator (see
        `communicator`); from :meth:`apply` when the fields live on a device,
        because :class:`muGrid.GridTransfer` has host kernels only; and for
        2D P1 elements (see :class:`_MultigridLevel`).
    ValueError
        If fewer than two levels are possible, or `nb_levels` does not divide
        the grid.
    """

    #: ω = SAFETY / λ_max(D⁻¹K). The measured optimum is 1.7 across
    #: {2D, 3D} x {Q1, P1}; the stability limit is 2.0 and divergence sets in
    #: sharply at 1.9, so this keeps ~15% margin.
    SAFETY = 1.7

    def __init__(self, decomposition, grid_spacing, lambda_ref, mu_ref,
                 communicator=None, element=None, nb_levels=None,
                 min_coarse=32, nu=2, omega=None, nb_cycles=1, timer=None,
                 dtype=np.float64,
                 name="multigrid-reference-preconditioner"):
        from .Parallel import Communicator
        from .Wrappers import CartesianDecomposition, FFTEngine, GridTransfer, _muGrid

        if element is None:
            element = _muGrid.FEMElement.q1
        # Check the decomposition, not only the communicator argument: a
        # parallel decomposition passed without a communicator would otherwise
        # run the serial algorithm with rank-local reductions.
        distributed = int(np.prod(list(decomposition.nb_subdivisions))) > 1
        if distributed or (communicator is not None and communicator.size > 1):
            raise NotImplementedError(
                "MultigridReferencePreconditioner supports serial runs only. "
                "A distributed V-cycle needs nested power-of-two subdivisions "
                "pinned across levels and a redundant coarsest level."
            )

        nb_grid_pts = tuple(decomposition.nb_domain_grid_pts)
        dim = len(nb_grid_pts)
        spacing = np.asarray(grid_spacing, dtype=float)
        if spacing.size != dim:
            raise ValueError(
                f"grid_spacing has {spacing.size} entries for a {dim}D grid")

        self.nu = int(nu)
        self.nb_cycles = int(nb_cycles)
        self._timer = timer
        self._name = name

        nb_levels = self._resolve_nb_levels(nb_grid_pts, nb_levels, min_coarse)
        if nb_levels < 2:
            # A single level is the coarsest level, which is solved by FFT --
            # so there is no cycle left, only the plain Fourier preconditioner,
            # and the fine decomposition handed in would have to be an FFT
            # engine for it to work at all. Say so here rather than failing
            # later on a missing `real_space_collection`.
            raise ValueError(
                f"a V-cycle needs at least two levels, but {nb_grid_pts} "
                f"cannot be coarsened towards min_coarse={min_coarse}: every "
                "extent must stay even and at least that wide. Use a finer "
                "grid, lower min_coarse, or -- if one level is really what you "
                "want -- make_reference_stiffness_preconditioner, which is "
                "exactly that."
            )
        self.nb_levels = nb_levels

        # Coarse levels must live where the fine one does: the cycle moves
        # fields straight between levels (restrict/prolong), so a host coarse
        # grid under a device fine grid is a device mismatch, not a slow path.
        level_kwargs = {"nb_ghosts_left": (1,) * dim,
                        "nb_ghosts_right": (1,) * dim,
                        "device": decomposition.device}
        comm = communicator if communicator is not None else Communicator()

        self.levels = []
        for lvl in range(nb_levels):
            coarsening = 2 ** lvl
            level_pts = tuple(n // coarsening for n in nb_grid_pts)
            with_fft = lvl == nb_levels - 1
            if lvl == 0:
                level_decomp = decomposition
            elif with_fft:
                # The coarsest level is solved exactly in Fourier space, so its
                # decomposition has to be an FFT engine.
                level_decomp = FFTEngine(level_pts, comm, **level_kwargs)
            else:
                level_decomp = CartesianDecomposition(
                    comm, list(level_pts),
                    nb_subdivisions=list(decomposition.nb_subdivisions),
                    **level_kwargs)
            self.levels.append(_MultigridLevel(
                level_decomp, spacing * coarsening, element, lambda_ref,
                mu_ref, dim, f"{name}-l{lvl}", dtype, with_fft=with_fft))

        self.omega = (float(omega) if omega is not None
                      else self.SAFETY / self.levels[-1].lambda_max(comm))

        self.transfer = GridTransfer(dim)
        self._on_device = not decomposition.device.is_host

        # Coarsest level: the exact block-Fourier inverse of Kʳᵉᶠ, reusing the
        # impulse-response assembly. It already replaces the singular q = 0
        # block by its pseudo-inverse, which is the nullspace handling the
        # bottom of the cycle needs.
        bottom = self.levels[-1]
        self._coarse_prec = make_reference_stiffness_preconditioner(
            bottom.decomp,
            lambda u_in, f_out: bottom.apply(u_in, f_out),
            dim, name=f"{name}-coarse", timer=timer, dtype=dtype)

    # -- setup helpers ------------------------------------------------------ #

    @staticmethod
    def _resolve_nb_levels(nb_grid_pts, nb_levels, min_coarse):
        """Coarsen while every direction stays even and wide enough."""
        if nb_levels is not None:
            nb_levels = int(nb_levels)
            for lvl in range(nb_levels):
                if any(n % (2 ** lvl) for n in nb_grid_pts):
                    raise ValueError(
                        f"{nb_levels} levels need every grid extent divisible "
                        f"by {2 ** (nb_levels - 1)}, got {nb_grid_pts}")
            return nb_levels
        nb_levels = 1
        while (all(n % (2 ** nb_levels) == 0 for n in nb_grid_pts) and
               all(n // (2 ** nb_levels) >= min_coarse for n in nb_grid_pts)):
            nb_levels += 1
        return nb_levels

    def _timed(self, label):
        return self._timer(label) if self._timer is not None else nullcontext()

    # -- the cycle ---------------------------------------------------------- #

    def _vcycle(self, lvl):
        level = self.levels[lvl]
        if lvl == self.nb_levels - 1:
            with self._timed("coarse"):
                self._coarse_prec.apply(level.r, level.z)
            return

        coarser = self.levels[lvl + 1]
        level.z.set_zero()
        level.smooth(self.nu, self.omega)

        # Residual r - K z, restricted to the coarser level.
        level.apply(level.z, level.t)
        linalg.axpby(1.0, level.r, -1.0, level.t)
        level.decomp.communicate_ghosts(level.t)
        self.transfer.restrict(level.t, coarser.r)

        self._vcycle(lvl + 1)

        # Coarse-grid correction, interpolated back and added.
        coarser.decomp.communicate_ghosts(coarser.z)
        self.transfer.prolong(coarser.z, level.t)
        linalg.axpy(1.0, level.t, level.z)

        level.smooth(self.nu, self.omega)

    def apply(self, r, z):
        """``z = M r``: ``nb_cycles`` V-cycles from ``z = 0``."""
        if self._on_device:
            # Everything else in the cycle runs on the device; the two grid
            # transfers do not, because GridTransfer declares host-space
            # overloads only (src/libmugrid/operators/transfer.hh). Say so
            # here, rather than letting pybind report an overload mismatch
            # from three frames down. Construction is deliberately still
            # allowed: the per-level pieces are individually timeable on the
            # device, which is how examples/vcycle_vs_fft.py prices a cycle
            # there.
            raise NotImplementedError(
                "the V-cycle cannot run on the device yet: restrict/prolong "
                "have no device kernels, only host ones. Run the solve on the "
                "CPU, or use make_reference_stiffness_preconditioner, which is "
                "GPU-capable."
            )
        fine = self.levels[0]
        with self._timed("vcycle"):
            linalg.copy(r, fine.r)
            _project_constants_out(fine.r)
            if self.nb_cycles == 1:
                self._vcycle(0)
                linalg.copy(fine.z, z)
            else:
                z.set_zero()
                for cycle in range(self.nb_cycles):
                    if cycle:
                        # Re-form the residual against the accumulated z.
                        fine.apply(z, fine.t)
                        linalg.copy(r, fine.r)
                        _project_constants_out(fine.r)
                        linalg.axpy(-1.0, fine.t, fine.r)
                    self._vcycle(0)
                    linalg.axpy(1.0, fine.z, z)
            _project_constants_out(z)


# --------------------------------------------------------------------------- #
# Hybrid: FFT in the rank-local axes, tridiagonal solve in the distributed one
# --------------------------------------------------------------------------- #


def _array_module(on_device):
    """``cupy`` for device arrays, ``numpy`` for host ones."""
    if on_device:
        import cupy
        return cupy
    return np


def _mpi_is_gpu_aware():
    """Whether MPI may be handed device buffers directly.

    The C++ side's answer (``mpi/gpu_aware_mpi.hh``, exposed as
    ``Communicator.gpu_aware_mpi``): ``MUGRID_GPU_AWARE_MPI`` when set, else
    the MPI library's own report through the OpenMPI extension. That probe
    cannot see Cray MPICH's switch, ``MPICH_GPU_SUPPORT_ENABLED=1``, which is
    what makes that MPI accept device memory, so with the variable unset it
    counts as a yes too. A ``_muGrid`` that does not expose the property
    falls back to the variable alone, read as the C++ side reads it.
    """
    from .Wrappers import _muGrid

    env = os.environ.get("MUGRID_GPU_AWARE_MPI", "")
    probed = getattr(_muGrid.Communicator, "gpu_aware_mpi", None)
    if probed is None:
        probed = bool(env) and env[0] in "1yYtT"
    if env:
        return bool(probed)
    return bool(probed) or (
        os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "") == "1")


#: Tolerance on ``|A A^-1 - I|``, in units of the working precision's eps. Both
#: ends are measured: a healthy block lands within a few eps of zero (5.9e-16 in
#: double, 4.6e-7 in single) and a singular one at 6e-4 and above, so this sits
#: a couple of orders clear of each in single and ten or more in double. It is
#: deliberately *not* an absolute number -- the hybrid runs at complex64
#: whenever the solver does, where an absolute threshold such as 1e-8 is below
#: the rounding noise of a healthy block and would reject them by the hundred.
_INVERSE_RESIDUAL_EPS = 1e3


def _batched_inverse(matrices, xp=np, singular_mode=None):
    """Inverse per mode; the one mode that has no inverse is zeroed, not fudged.

    ``singular_mode`` is the index of the all-zero mode, where every matrix
    built along the distributed axis inherits the reference operator's rigid
    translation and may therefore be singular. Its inverse is set to zero --
    unconditionally, whether or not that particular matrix turned out to be
    invertible. :func:`make_reference_stiffness_preconditioner` sets the same
    block at ``q = 0`` to zero as its pseudo-inverse; here zero is safe for a
    stronger reason: :meth:`apply` replaces that mode's entire solve with
    :meth:`_solve_zero_mode`, so nothing computed from it is ever read.
    Passing ``None`` states that no mode may be singular.

    What this deliberately does not do is *discover* the singular mode by the
    size of its residual and repair it with a pseudo-inverse. That is the trap
    :meth:`HybridFourierTridiagonalPreconditioner._build_zero_mode` documents
    -- a pseudo-inverse separates kernel from range by a magnitude threshold,
    and at complex64 a threshold below the rounding noise of a healthy block
    fires on hundreds of them, each then replaced (in a Python loop) by a
    pseudo-inverse computed at that same precision.

    Everything else is verified instead: a mode that is singular and not
    declared so raises, because the alternative is a preconditioner that is
    quietly wrong in one mode rather than one that stops.
    """
    index = (None if singular_mode is None
             else int(np.ravel_multi_index(singular_mode, matrices.shape[:-2])))
    flat_in = matrices.reshape((-1,) + matrices.shape[-2:])
    eye = xp.eye(flat_in.shape[-1], dtype=flat_in.dtype)

    def refuse(what):
        return RuntimeError(
            f"{what}, in a batch of {flat_in.shape[-1]}x{flat_in.shape[-1]} "
            "Fourier-mode blocks. Only the all-zero mode may be singular here, "
            "where the reference operator keeps the rigid translation and the "
            "hybrid replaces that mode's whole solve with _solve_zero_mode. "
            "Anything else is a nullspace this preconditioner does not know "
            "about, and using it would be silently wrong rather than slow.")

    try:
        flat_out = xp.linalg.inv(flat_in)
    except xp.linalg.LinAlgError as error:
        # inv() refuses the whole batch if any one member is exactly singular,
        # and says only "Singular matrix". Substituting the identity for the
        # mode that is allowed to be singular -- on a copy, since `matrices`
        # may be the caller's live array -- takes the batch through in one
        # call. If some other mode was the singular one, it refuses again, and
        # that is the answer.
        if index is None:
            raise refuse("LAPACK rejected a block as exactly singular") \
                from error
        patched = flat_in.copy()
        patched[index] = eye
        try:
            flat_out = xp.linalg.inv(patched)
        except xp.linalg.LinAlgError:
            raise refuse(
                "LAPACK rejected a block as exactly singular, and it was not "
                f"the declared mode {tuple(singular_mode)}") from error

    # inv() does not always fail on a singular mode; more often it returns
    # something large and finite. Only the residual catches that, and it is one
    # batched matmul.
    residual = xp.abs(flat_in @ flat_out - eye).max(axis=(-2, -1))
    if index is not None:
        flat_out[index] = 0
        residual[index] = 0

    # np.finfo, not xp.finfo: the dtype of a device array is a numpy dtype
    # either way, and this does not depend on cupy mirroring the call.
    tolerance = _INVERSE_RESIDUAL_EPS * float(np.finfo(flat_in.dtype).eps)
    finite = xp.isfinite(residual)
    bad = ~finite | (residual > tolerance)
    nb_bad = int(bad.sum())
    if nb_bad:
        measured = bad & finite
        worst = (float(residual[measured].max())
                 if bool(measured.any()) else float("inf"))
        raise refuse(
            f"{nb_bad} of {residual.size} Fourier modes invert to "
            f"|A A^-1 - I| as large as {worst:.3e}, past a tolerance of "
            f"{tolerance:.3e}")
    return flat_out.reshape(matrices.shape)


def reference_stencil(dim, grid_spacing, element, lambda_ref, mu_ref, probe=8):
    """The uniform reference operator's stencil, by impulse response.

    **The single definition of "the uniform reference operator".** Everything
    that needs that operator in Fourier space -- the block preconditioner's
    symbol, the hybrid's z-coupling blocks -- derives from this stencil via
    :func:`stencil_symbol` rather than re-deriving a symbol of its own.

    Because it *probes the real operator* rather than re-deriving `B(q)` from
    the element tables, it matches the C++ discretisation by construction:
    there is no second expression of the same maths to drift out of sync.

    Returns ``S`` with ``f(y) = sum_d S[d] u(y - d)``, indexed ``S[dx+1, ...]``.
    Measured on a small *serial* grid: the operator is uniform, so every rank
    gets the same answer and this needs no communication. `probe` only has to
    exceed the stencil's reach.
    """
    from .Parallel import Communicator
    from .Wrappers import CartesianDecomposition, IsotropicStiffnessOperator

    with warnings.catch_warnings():
        # A serial communicator under an MPI launcher is exactly the intent:
        # the operator is uniform, so every rank probes the same stencil for
        # itself and nothing is exchanged. muGrid warns about the redundancy,
        # which here is the point rather than a mistake.
        warnings.filterwarnings("ignore", message=".*serial muGrid Communicator.*")
        decomposition = CartesianDecomposition(
            Communicator(), [probe] * dim, nb_subdivisions=[1] * dim,
            nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim)
    op = IsotropicStiffnessOperator(dim, tuple(grid_spacing), element)
    collection = decomposition.collection
    u = collection.real_field("hybrid-stencil-u", (dim,))
    f = collection.real_field("hybrid-stencil-f", (dim,))

    S = np.zeros((3,) * dim + (dim, dim))
    centre = probe // 2
    for beta in range(dim):
        u.set_zero()
        u.s[(beta, 0) + (centre,) * dim] = 1.0
        decomposition.communicate_ghosts(u)
        op.apply_uniform(u, lambda_ref, mu_ref, f)
        response = np.asarray(f.s)[:, 0]
        for offset in np.ndindex((3,) * dim):
            S[offset + (slice(None), beta)] = response[
                (slice(None),) + tuple(centre + o - 1 for o in offset)]
    return S


def _z_coupling_blocks(dim, grid_spacing, element, lambda_ref, mu_ref,
                       local_shape, cdtype, xp=np):
    """Transform the rank-local axes; return the three coupling blocks per mode.

    ``A[m]``, ``m = dz + 1``, has shape ``(*mode_shape, dim, dim)`` and the
    operator along the distributed axis is
    ``(T v)[k] = A[0] v[k+1] + A[1] v[k] + A[2] v[k-1]``.

    The modes are laid out in *memory* order: the rank-local axes reversed,
    ``(..., y, x)``, with the first of them, ``x``, the real (halved)
    transform -- ``numpy.fft.rfftn`` over the reversed axes. ``x`` varies
    fastest in both of muGrid's storage orders, so this is the order in which
    the field already lies, and transforming in it needs no transposition of
    the spatial axes (see `HybridFourierTridiagonalPreconditioner._layout`).
    """
    # The stencil probe is a tiny host computation whatever the target is; only
    # the per-mode blocks are big enough to want to live on the device.
    S = reference_stencil(dim, grid_spacing, element, lambda_ref, mu_ref)
    qs = [2 * xp.pi * xp.fft.fftfreq(n) for n in local_shape]
    qs[0] = 2 * xp.pi * xp.fft.rfftfreq(local_shape[0])
    # Grids indexed (..., y, x), handed over in axis order (x, y, ...).
    grids = xp.meshgrid(*reversed(qs), indexing="ij")[::-1]
    return stencil_symbol(S, grids, offset_axis=dim - 1, cdtype=cdtype, xp=xp,
                          angular=True)


#: Alias of :func:`reference_stencil` under its private spelling, for callers
#: that import that name. Deprecated.
_reference_stencil = reference_stencil


def stencil_symbol(S, q, offset_axis=None, cdtype=complex, xp=np,
                   angular=False):
    r"""Fourier symbol of a real-space stencil, assembled analytically.

    For ``f(y) = Σ_d S[d] u(y - d)`` the symbol is
    ``Ŝ(q) = Σ_d S[d] exp(-i q·d)``, a closed-form sum over the stencil's
    ``3^dim`` offsets. This replaces recovering the same information by
    impulse response on the full grid plus a full-grid FFT: the stencil is
    243 numbers in 3D and determines the symbol everywhere.

    The caller supplies the frequencies, because the two users of this
    function do not share a convention: muGrid's FFT engine makes **axis 0**
    the half-complex one (``engine.fftfreq``), while the hybrid preconditioner
    transforms its rank-local axes in memory order with ``numpy.fft.rfftn``
    semantics, which halve the **last** of them -- the fastest-varying axis,
    ``x`` (see :func:`_z_coupling_blocks`). Owning the phase algebra here and
    the convention at the call site keeps both honest.

    Parameters
    ----------
    S : ndarray
        Stencil of shape ``(3,) * dim + (dim, dim)``, indexed ``S[d + 1]``,
        as returned by :func:`reference_stencil`.
    q : sequence of ndarray
        One frequency array per *phased* axis, all of the mode shape. In
        cycles unless ``angular`` is set.
    offset_axis : int, optional
        An axis to leave *unphased*. The result then carries a leading axis of
        length three indexed by that offset (``m = d + 1``), which is what a
        direction-split solver needs: the operator along it is
        ``(T v)[k] = A[0] v[k+1] + A[1] v[k] + A[2] v[k-1]``. ``None`` (the
        default) phases every axis and returns a single block per mode.
    angular : bool, optional
        ``q`` is already in radians (``2π f``) rather than cycles.

    Returns
    -------
    ndarray
        ``(*mode_shape, dim, dim)``, or ``(3, *mode_shape, dim, dim)`` when
        ``offset_axis`` is given.

    Notes
    -----
    The stencil of a self-adjoint operator satisfies ``S[d] = S[-d]ᵀ``
    exactly, so the symbol is Hermitian by construction rather than to within
    round-off -- measured at 8.7e-17 for Q1 elasticity, against an assembled
    symbol it reproduces to 4.9e-16.
    """
    dim = S.ndim - 2
    phased = [a for a in range(dim) if a != offset_axis]
    if len(q) != len(phased):
        raise ValueError(
            f"stencil_symbol: expected {len(phased)} frequency arrays for a "
            f"{dim}D stencil with offset_axis={offset_axis}, got {len(q)}")
    scale = 1.0 if angular else 2.0 * np.pi
    mode_shape = xp.asarray(q[0]).shape if phased else ()

    # One transcendental pass per phased axis, not one per (offset, axis).
    # Stencil offsets are d in {-1, 0, +1}: d = 0 contributes nothing, and
    # e^{+i0} is the conjugate of e^{-i0}. So `dim` calls to exp over the mode
    # grid cover all 3^dim offsets, where the naive loop makes 3^dim * dim of
    # them -- 81 in 3D, and exp on a complex grid is the dominant cost.
    fwd = [xp.exp(-1j * scale * xp.asarray(qk)).astype(cdtype) for qk in q]
    phase_for = [{-1: xp.conj(f), 0: None, +1: f} for f in fwd]

    lead = () if offset_axis is None else (3,)
    out = xp.zeros(lead + mode_shape + (dim, dim), dtype=cdtype)
    for offset in np.ndindex((3,) * dim):
        phase = None
        for k, axis in enumerate(phased):
            factor = phase_for[k][offset[axis] - 1]
            if factor is None:
                continue
            phase = factor if phase is None else phase * factor
        block = xp.asarray(S[offset])
        block = block if phase is None else phase[..., None, None] * block
        if offset_axis is None:
            out += block
        else:
            out[offset[offset_axis]] += block
    return out


class HybridFourierTridiagonalPreconditioner(Preconditioner):
    r"""``Kʳᵉᶠ⁻¹`` by FFT in the rank-local axes and a tridiagonal solve in the
    distributed one.

    The same operator :func:`make_reference_stiffness_preconditioner` applies,
    reached without ever transforming the distributed axis -- and so without the
    all-to-all that forces. Unlike :class:`MultigridReferencePreconditioner`
    this is *exact*, so it costs the FFT preconditioner's iteration count rather
    than roughly 1.5x of it.

    Two properties make it work, and neither is separability:

    1. ``Kʳᵉᶠ`` is translation-invariant in the undistributed axes -- it is
       uniform by construction -- so transforming them decouples every mode
       exactly.
    2. The stencil reaches exactly one node along the distributed axis, for Q1
       and P1 alike, so what remains there is block-tridiagonal with
       ``dim x dim`` blocks.

    Q1 elasticity is not a Kronecker sum, so the usual "the Laplacian
    factorises" argument does not apply here; (2) is a statement about stencil
    support, and it does.

    The distributed solve is the Spike/partitioned-Thomas scheme. Each rank
    eliminates its own slab against three right-hand sides -- the residual, and
    the unit responses to its two neighbours' interface planes -- which leaves a
    reduced system in the ``2 * dim`` interface unknowns per rank. That reduced
    system is small (``2 * dim * nb_ranks`` per mode), is assembled and inverted
    once at setup, and **absorbs the periodic wrap-around**, so the
    Sherman-Morrison correction a periodic tridiagonal would otherwise need
    disappears.

    Per apply this exchanges only interface planes -- ``2 * dim`` complex numbers
    per mode per rank -- against an all-to-all of the entire field.

    In formulas: let ``z`` be the last axis (distributed over ``P`` ranks,
    ``N_z`` planes, ``n_z = N_z / P`` per rank) and ``q`` the frequencies of
    the other, rank-local axes. After the real FFT over those axes, the
    reference operator for each ``q`` is the periodic block-tridiagonal
    system

        ``A₀(q) v_{k+1} + A₁(q) v_k + A₂(q) v_{k-1} = r̂_k(q)``,
        ``k = 0 … N_z - 1`` (indices mod ``N_z``),

    with the ``d × d`` coupling blocks ``A_m(q)`` (:func:`stencil_symbol`
    with ``offset_axis``). Per rank and mode:

    1. **Block-Thomas** on the rank's ``n_z`` planes without the periodic or
       inter-rank coupling: ``D₀ = A₁``, ``D_k = A₁ - A₂ D_{k-1}⁻¹ A₀``,
       forward ``y_k = r̂_k - A₂ D_{k-1}⁻¹ y_{k-1}``, backward
       ``v_k = D_k⁻¹ (y_k - A₀ v_{k+1})``; ``O(d³ n_z)`` per mode. The
       factors ``D_k⁻¹`` are computed once at setup; the recursion is a
       fixed-point iteration in ``k``, so only the first
       :attr:`HEAD_LENGTH` are stored except for the slowly converging
       modes.
    2. **Interface (Spike) system** for the ``2d`` end-plane unknowns per
       rank, which couples rank ``p`` to ``p ± 1`` cyclically. For a
       power-of-two ``P`` it is solved by **parallel cyclic reduction** in
       ``log₂ P`` neighbour exchanges (see :meth:`_build_pcr`); otherwise its
       ``2dP × 2dP`` inverse per mode is assembled at setup and applied after
       an all-gather.
    3. A second block-Thomas sweep with the right-hand side corrected at the
       two end planes by the neighbours' interface values.
    4. The ``q = 0`` mode, whose operator is singular (constant
       translations), is solved separately on its gathered ``N_z``-line by a
       deflated dense inverse projected onto the complement of the null space
       (see :meth:`_build_zero_mode`).

    Cost per apply: two batched real FFTs over the local axes,
    ``O(N log N)`` with no inter-rank communication, two ``O(d² N)``
    tridiagonal sweeps, and ``log₂ P + 1`` interface exchanges of
    ``O(d N / N_z)`` values each. Setup is ``O(d³ N / P)`` per rank for the
    factors and spikes.

    Host and device are the same code: the array module follows the
    decomposition, so ``numpy`` and ``cupy`` take the identical path. Only the
    interface exchange can differ: device buffers go to MPI directly when it is
    GPU-aware (``MUGRID_GPU_AWARE_MPI=1``, or ``MPICH_GPU_SUPPORT_ENABLED=1`` on
    Cray MPICH) and are staged through the host otherwise, so that any MPI build
    works.

    Parameters
    ----------
    decomposition : muGrid.CartesianDecomposition
        The solver's decomposition, which must be a **slab**: one subdivision in
        every axis but the last. That is the split ``muGrid.FFTEngine`` itself
        chooses, and it is what leaves the other axes rank-local.
    grid_spacing : sequence of float
        Grid spacing, one entry per direction.
    lambda_ref, mu_ref : float
        Uniform reference Lamé parameters; normally the volume means.
    communicator : muGrid.Communicator, optional
        Communicator of the parallel run; its ``mpi4py_comm`` carries the
        interface exchanges. Default: a serial communicator.
    element : muGrid.FEMElement, optional
        Default Q1.
    timer : muTimer.Timer, optional
        When given, :meth:`apply` records ``"fft"``, ``"tridiag"``,
        ``"interface"`` and ``"ifft"``.
    dtype : data-type, optional
        Real precision of the solve, ``np.float64`` (default) or
        ``np.float32``; the per-mode blocks and factors are complex128 or
        complex64 accordingly.
    name : str, optional
        Name of the preconditioner (stored, used for nothing else).

    Raises
    ------
    ValueError
        If the decomposition is not a slab, the distributed axis does not
        divide evenly among the ranks, the ranks are not ordered along it, or
        a rank holds fewer than two planes.
    RuntimeError
        If a Fourier block other than the ``q = 0`` one is singular, or the
        spike blocks differ between ranks.
    """

    def __init__(self, decomposition, grid_spacing, lambda_ref, mu_ref,
                 communicator=None, element=None, timer=None,
                 dtype=np.float64, name="hybrid-fourier-tridiagonal"):
        # Everything below allocates through cupy, which places arrays on its
        # *current* device -- device 0 unless the caller switched it. With one
        # GPU per rank that puts rank 1's factors on GPU 0 next to fields on
        # GPU 1, and the first kernel that mixes them faults. Pin the
        # decomposition's device instead of relying on the caller to.
        self._device_id = (None if decomposition.device.is_host
                           else decomposition.device.device_id)
        with self._on_own_device():
            self._setup(decomposition, grid_spacing, lambda_ref, mu_ref,
                        communicator, element, timer, dtype, name)

    def _on_own_device(self):
        """Make the decomposition's GPU cupy's current device (no-op on host)."""
        if self._device_id is None:
            return nullcontext()
        import cupy
        return cupy.cuda.Device(self._device_id)

    def _setup(self, decomposition, grid_spacing, lambda_ref, mu_ref,
               communicator, element, timer, dtype, name):
        from .Parallel import Communicator
        from .Wrappers import _muGrid

        if element is None:
            element = _muGrid.FEMElement.q1
        comm = communicator if communicator is not None else Communicator()
        self._comm = comm
        self._mpi = comm.mpi4py_comm if comm.size > 1 else None
        self._timer = timer
        self._name = name

        self._on_device = not decomposition.device.is_host
        xp = _array_module(self._on_device)
        self._xp = xp
        # Whether the interface exchange hands its buffers to MPI as they are.
        # Host arrays always; device arrays only to a GPU-aware MPI -- staging
        # those through pageable host memory cost 5 ms per apply on two MI300A
        # and 17 ms on four at 256^3, for a few MB of interface planes.
        self._mpi_direct = (not self._on_device) or _mpi_is_gpu_aware()

        dim = len(tuple(decomposition.nb_domain_grid_pts))
        self.dim = dim
        subdivisions = tuple(int(x) for x in decomposition.nb_subdivisions)
        if subdivisions[:-1] != (1,) * (dim - 1):
            raise ValueError(
                "the hybrid preconditioner needs a slab decomposition -- one "
                "subdivision in every axis but the last -- but got "
                f"{list(subdivisions)}. That is the split muGrid.FFTEngine "
                "chooses; build the solver's CartesianDecomposition with "
                f"nb_subdivisions={[1] * (dim - 1) + [comm.size]}.")

        self.decomposition = decomposition
        self.nb_ranks = subdivisions[-1]
        self.rank = comm.rank
        interior = tuple(int(x) for x in decomposition.nb_subdomain_grid_pts)
        self.local_shape = interior[:-1]
        self.nz_local = interior[-1]
        self._cdtype = (np.complex64 if np.dtype(dtype) == np.dtype(np.float32)
                        else np.complex128)

        nz_global = tuple(decomposition.nb_domain_grid_pts)[-1]
        if nz_global % self.nb_ranks:
            raise ValueError(
                f"the distributed axis ({nz_global} points) must divide evenly "
                f"among {self.nb_ranks} ranks: the interface exchange and the "
                "zero-mode gather both assume every rank holds the same number "
                "of planes.")
        if int(decomposition.subdomain_locations[-1]) != self.rank * self.nz_local:
            raise ValueError(
                "ranks are not laid out in order along the distributed axis, "
                "which the zero-mode gather assumes.")
        if self.nz_local < 2:
            raise ValueError(
                "each rank needs at least two planes of the distributed axis, "
                f"but rank {comm.rank} has {self.nz_local}. The slab caps at "
                f"{tuple(decomposition.nb_domain_grid_pts)[-1] // 2} ranks.")

        self.A = _z_coupling_blocks(dim, grid_spacing, element, lambda_ref,
                                    mu_ref, self.local_shape, self._cdtype, xp)
        self._mode_shape = self.A.shape[1:-2]
        # The one mode whose blocks may be singular, and the only one whose
        # solve is thrown away and redone by _solve_zero_mode. Every inverse
        # taken below is told about it rather than left to discover it.
        self._zero_index = (0,) * len(self._mode_shape)

        self.nz_global = nz_global
        self._factorise_local()
        # Parallel cyclic reduction needs a power-of-two rank count for the
        # stride doubling to close on the cyclic wrap. Where it does not apply,
        # fall back to assembling and inverting the reduced system densely --
        # correct either way, but O(P) in communication and O(P^2) in work per
        # mode, which dominates the apply by eight ranks.
        self._pcr_nb_levels = (self.nb_ranks.bit_length() - 1
                               if self.nb_ranks > 1
                               and self.nb_ranks & (self.nb_ranks - 1) == 0
                               else 0)
        self._use_pcr = self._pcr_nb_levels > 0
        if self._use_pcr:
            self._build_pcr()
        else:
            self._build_reduced_system()
        self._build_corrections()
        self._build_zero_mode()

    # -- setup -------------------------------------------------------------- #

    def _factorise_local(self):
        """Thomas factors of this rank's slab, plus its two spikes.

        The spikes are the slab's response to a unit value on each neighbour's
        interface plane. They do not depend on the residual, so they and the
        reduced system they build are computed once here, not per apply.
        """
        xp = self._xp
        dim, nz = self.dim, self.nz_local
        A0, A1, A2 = self.A[0], self.A[1], self.A[2]

        # Everything along the distributed axis is stored z-major,
        # ``(nz, *modes, ...)``. Neighbouring modes are then adjacent in memory
        # at each step, which is what lets a wavefront in the fused kernel read
        # contiguous bytes; the mode-major alternative strides by ``nz * dim``
        # between neighbouring threads and gives up most of the bandwidth. It
        # costs nothing to choose: the transform emits ``(dim, *modes, nz)`` and
        # a permutation has to be materialised either way.
        Dinv = xp.empty((nz,) + self._mode_shape + (dim, dim),
                        dtype=self._cdtype)
        zero = self._zero_index
        Dinv[0] = _batched_inverse(A1, xp, singular_mode=zero)
        for k in range(1, nz):
            Dinv[k] = _batched_inverse(A1 - A2 @ Dinv[k - 1] @ A0, xp,
                                       singular_mode=zero)
        self.Dinv = Dinv

        eye = xp.broadcast_to(xp.eye(dim, dtype=self._cdtype),
                              self._mode_shape + (dim, dim))
        left = xp.zeros((nz,) + self._mode_shape + (dim, dim),
                        dtype=self._cdtype)
        right = xp.zeros_like(left)
        left[0] = -A2 @ eye
        right[nz - 1] = -A0 @ eye
        V = self._solve_local(left)
        W = self._solve_local(right)
        # Only the four interface blocks outlive setup. The full spikes are the
        # largest arrays here -- two more copies of the factor storage -- and
        # apply() does not need them: adding ``V left + W right`` to the
        # solution is the same as solving once more against a right-hand side
        # corrected at the two ends, which is one extra sweep instead of a pass
        # over both of them.
        self._spike_ends = xp.stack([V[0], V[-1], W[0], W[-1]], axis=-3)
        # Host and device alike: the fused sweeps on both stream these factors,
        # so compressing them cuts traffic as well as storage. It runs after the
        # spikes, which are built through the elementwise path and want the
        # uncompressed array.
        self._compress_factors()

    #: Factors kept per mode before the fixed point takes over. The
    #: convergence distribution is a property of the operator, not the grid --
    #: median 10 steps, 90th percentile 16, and only ~0.65% of modes need more
    #: than 64 at any size measured -- so this trades a dense head against a
    #: sparse exception table. Around 32 minimises the total.
    HEAD_LENGTH = 32

    def _compress_factors(self):
        """Replace the factor array by a head plus an exception table.

        ``D_k = A1 - A2 D_{k-1}^-1 A0`` has constant coefficients, so it is a
        fixed-point iteration and its factors stop changing after a few steps.
        Storing the first ``HEAD_LENGTH`` of them and reusing the last for the
        rest is exact for every mode that has converged by then; the few that
        have not keep a full line.

        This is not only memory. The sweep *streams* these factors, so the
        traffic falls with the storage.
        """
        xp = self._xp
        nz = self.nz_local
        head_len = min(self.HEAD_LENGTH, nz)
        head = xp.ascontiguousarray(self.Dinv[:head_len])

        # A mode is exceptional if reusing the last head factor would be wrong
        # anywhere along the remaining axis.
        tail = self.Dinv[head_len:]
        if tail.shape[0]:
            deviation = xp.abs(tail - head[-1]).max(axis=(0, -2, -1))
            scale = xp.abs(head[-1]).max(axis=(-2, -1))
            # Multiplied through rather than divided: `scale` is real32 when
            # the solve is, where a guard floor like 1e-300 is not a small
            # number but zero, and the all-zero factor line of the singular
            # mode then divides 0 by 0. This form needs no floor -- a constant
            # line, zero included, is exactly what the head reproduces.
            exceptional = deviation > 1e-12 * scale
        else:
            exceptional = xp.zeros(self._mode_shape, dtype=bool)

        flat = exceptional.reshape(-1)
        nb_modes = int(flat.size)
        index = xp.full(nb_modes, -1, dtype=xp.int32)
        picked = xp.flatnonzero(flat)
        nb_exc = int(picked.size)
        index[picked] = xp.arange(nb_exc, dtype=xp.int32)

        dim = self.dim
        full = self.Dinv.reshape(nz, nb_modes, dim, dim)
        self._head = head
        self._exc = (xp.ascontiguousarray(full[:, picked])
                     if nb_exc else xp.empty((nz, 1, dim, dim),
                                             dtype=self._cdtype))
        self._exc_index = index
        self._picked = picked
        self._nb_exc = nb_exc
        self._head_length = head_len
        # Contiguous, kernel-shaped views of the factors for the device sweep,
        # built on its first call and reused: they do not change after setup.
        self._device_sweep_inputs = None
        # The full array is what this exists to get rid of.
        self.Dinv = None

    def _dinv_at(self, k):
        """``D_k^-1``, from whichever representation is in use.

        Before compression -- and always on the host -- this is a plain slice.
        Afterwards it rebuilds the plane from the head and the exception table,
        which is why the elementwise path stays usable on a device: it is the
        reference the fused kernel is tested against, and a reference that
        could not run would be no reference at all.
        """
        if self.Dinv is not None:
            return self.Dinv[k]
        base = self._head[min(k, self._head_length - 1)]
        if not self._nb_exc:
            return base
        patched = base.copy()
        patched.reshape(-1, self.dim, self.dim)[self._picked] = self._exc[k]
        return patched

    def _solve_local(self, rhs, fused=True):
        """``T_local x = rhs`` with the stored factors.

        ``T_local`` is this rank's slab with no wrap-around and no coupling to
        its neighbours; both enter through the spikes. `rhs` is
        ``(nz, *modes, dim, ncols)``, so one code path serves the residual
        (``ncols = 1``) and the spikes (``ncols = dim``).

        With a single column this hands over to a fused kernel -- the C++ one on
        the host, the device kernel on a device. The loop below issues four
        array products per plane, each streaming the whole mode array, so it
        costs ``4 * nz`` passes over memory where the sweeps cost two. It
        remains the setup path for the spikes, which have ``dim`` columns, and
        the reference both fused kernels are tested against; pass
        ``fused=False`` to take it deliberately.
        """
        if fused and rhs.shape[-1] == 1 and self.Dinv is None:
            if self._on_device:
                return self._solve_local_fused(rhs)
            return self._solve_local_fused_host(rhs)

        xp = self._xp
        nz = self.nz_local
        matmul = xp.matmul

        y = xp.empty_like(rhs)
        y[0] = rhs[0]
        for k in range(1, nz):
            y[k] = rhs[k] - matmul(self.A[2],
                                   matmul(self._dinv_at(k - 1), y[k - 1]))

        out = xp.empty_like(rhs)
        out[nz - 1] = matmul(self._dinv_at(nz - 1), y[nz - 1])
        for k in range(nz - 2, -1, -1):
            out[k] = matmul(self._dinv_at(k),
                            y[k] - matmul(self.A[0], out[k + 1]))
        return out

    def _solve_local_fused_host(self, rhs):
        """The same solve as one C++ call instead of ``4 * nz`` array products.

        The elementwise path streams the whole mode array four times per plane,
        so a solve costs ``4 * nz`` passes over memory where this costs two.
        Measured on the host that was the dominant cost of the preconditioner
        by an order of magnitude -- far more than the transform it replaces.

        The kernel runs plane-outer, mode-inner, which is the opposite of the
        device's one-thread-per-mode march: a single host thread walking one
        mode would stride by ``nb_modes * dim`` between planes, where this way
        every access is contiguous.
        """
        xp = self._xp
        nz, dim = self.nz_local, self.dim
        nb_modes = int(np.prod(self._mode_shape))

        flat = xp.ascontiguousarray(rhs.reshape(nz, nb_modes, dim))
        scratch = xp.empty_like(flat)
        out = xp.empty_like(flat)
        single = np.dtype(self._cdtype) == np.dtype(np.complex64)
        kernel = getattr(
            linalg, f"block_thomas_{dim}d" + ("_f32" if single else ""))
        kernel(flat,
               self._head.reshape(self._head_length, nb_modes, dim, dim),
               self._exc.reshape(nz, max(self._nb_exc, 1), dim, dim),
               self._exc_index,
               self.A[0].reshape(nb_modes, dim, dim),
               self.A[2].reshape(nb_modes, dim, dim),
               scratch, out)
        return out.reshape(rhs.shape)

    def _build_reduced_system(self):
        """The interface system, assembled once and inverted.

        Rank ``p`` satisfies ``lambda_p = x_p + V_p lambda_{p-1}^R +
        W_p lambda_{p+1}^L`` at both of its ends, which closes into a
        block-cyclic system of size ``2 * dim * nb_ranks`` per mode. Being small
        and dense it swallows the periodic wrap-around for free.
        """
        xp = self._xp
        dim, P = self.dim, self.nb_ranks
        gathered = self._allgather(
            xp.ascontiguousarray(self._spike_ends))

        size = 2 * dim * P
        M = xp.zeros(self._mode_shape + (size, size), dtype=self._cdtype)
        diagonal = xp.arange(size)
        M[..., diagonal, diagonal] = 1.0
        for p in range(P):
            V0, V1, W0, W1 = (gathered[p][..., i, :, :] for i in range(4))
            row_l = slice(2 * dim * p, 2 * dim * p + dim)
            row_r = slice(2 * dim * p + dim, 2 * dim * (p + 1))
            prev_r = slice(2 * dim * ((p - 1) % P) + dim,
                           2 * dim * ((p - 1) % P + 1))
            next_l = slice(2 * dim * ((p + 1) % P),
                           2 * dim * ((p + 1) % P) + dim)
            M[..., row_l, prev_r] -= V0
            M[..., row_l, next_l] -= W0
            M[..., row_r, prev_r] -= V1
            M[..., row_r, next_l] -= W1
        self._reduced_inv = _batched_inverse(
            M, xp, singular_mode=self._zero_index)

    def _solve_local_fused(self, rhs):
        """The same solve as one kernel launch instead of ``4 * nz``: the
        device block-Thomas sweep compiled into libmuGrid
        (``linalg/block_thomas_gpu.cc``).

        The binding takes device addresses rather than buffers, since nothing
        in the Python buffer protocol describes device memory, so every array
        handed over must be contiguous and of the exact shape the kernel
        indexes. Nothing on the C++ side can check that.
        """
        xp = self._xp
        nz, dim = self.nz_local, self.dim
        nb_modes = int(np.prod(self._mode_shape))
        nb_exc = max(self._nb_exc, 1)

        if self._device_sweep_inputs is None:
            single = np.dtype(self._cdtype) == np.dtype(np.complex64)
            self._device_sweep_inputs = (
                getattr(linalg, f"block_thomas_gpu_{dim}d"
                        + ("_f32" if single else "")),
                xp.ascontiguousarray(
                    self._head.reshape(self._head_length, nb_modes, dim, dim)),
                xp.ascontiguousarray(self._exc.reshape(nz, nb_exc, dim, dim)),
                xp.ascontiguousarray(self._exc_index.astype(xp.int32)),
                xp.ascontiguousarray(self.A[0].reshape(nb_modes, dim, dim)),
                xp.ascontiguousarray(self.A[2].reshape(nb_modes, dim, dim)),
            )
        kernel, head, exc, index, a0, a2 = self._device_sweep_inputs

        flat = xp.ascontiguousarray(rhs.reshape(nz, nb_modes, dim))
        scratch = xp.empty_like(flat)
        out = xp.empty_like(flat)
        # Raw addresses: run on the buffers' device, not the current one.
        with xp.cuda.Device(flat.device.id):
            kernel(flat.data.ptr, head.data.ptr, exc.data.ptr, index.data.ptr,
                   a0.data.ptr, a2.data.ptr, scratch.data.ptr, out.data.ptr,
                   nz, nb_modes, self._head_length, nb_exc)
        return out.reshape(rhs.shape)

    def _build_pcr(self):
        """Precompute the parallel-cyclic-reduction coefficients.

        The reduced system is block-circulant: every rank holds the same number
        of planes and the operator is uniform, so each rank's spike blocks are
        *bit-identical* (verified, not assumed -- checked below on every rank).
        The elimination coefficients at every level are therefore the same on
        every rank and can be built here without communication, leaving only
        right-hand sides to exchange per apply.

        Writing the system as ``u_p + A u_{p-s} + C u_{p+s} = x_p`` with stride
        ``s = 2^k``, one elimination step against the neighbours at ``±s`` gives::

            D       = I - A C - C A
            A'      = -D^-1 A^2 ,   C' = -D^-1 C^2
            x'_p    = D^-1 (x_p - A x_{p-s} - C x_{p+s})

        and doubles the stride. After ``log2(P)`` steps the stride reaches ``P``,
        where the cyclic wrap makes ``u_{p±P} = u_p`` and the equation closes
        locally as ``(I + A + C) u_p = x_p``.

        Cost per apply falls from one all-gather of ``P`` interface planes and a
        dense ``(2 d P)^2`` solve per mode, to ``log2(P) + 1`` neighbour
        exchanges and ``log2(P)`` products of ``2d x 2d`` blocks.
        """
        xp = self._xp
        dim = self.dim
        m = 2 * dim

        # A and C from this rank's spikes, identical everywhere by
        # construction -- and checked: the coefficients below are built from
        # the local spikes alone, so a rank whose spikes differed would
        # eliminate with the wrong ones and the result would be silently
        # wrong. Two fingerprints, reduced to min and max: identical blocks
        # give identical floats on every rank.
        comm = self._comm
        if comm.size > 1:
            spikes = self._spike_ends
            weights = xp.arange(1, spikes.size + 1, dtype=np.float64).reshape(
                spikes.shape)
            prints = (float(xp.abs(spikes).sum()),
                      float((xp.real(spikes) * weights).sum()))
            for value in prints:
                if comm.max(value) != comm.min(value):
                    raise RuntimeError(
                        "block-Thomas preconditioner: the spike blocks differ "
                        "between ranks, so the parallel cyclic reduction "
                        "would eliminate with wrong coefficients. The planes "
                        "must be distributed evenly over the ranks.")
        V0, V1, W0, W1 = (self._spike_ends[..., i, :, :] for i in range(4))
        zero = xp.zeros_like(V0)
        # u_p = (L_p, R_p); L_p couples to R_{p-1} and L_{p+1}, likewise R_p.
        A = -xp.concatenate([xp.concatenate([zero, V0], axis=-1),
                             xp.concatenate([zero, V1], axis=-1)], axis=-2)
        C = -xp.concatenate([xp.concatenate([W0, zero], axis=-1),
                             xp.concatenate([W1, zero], axis=-1)], axis=-2)

        identity = xp.zeros(self._mode_shape + (m, m), dtype=self._cdtype)
        diagonal = xp.arange(m)
        identity[..., diagonal, diagonal] = 1.0

        self._pcr_levels = []
        matmul = xp.matmul
        for _ in range(self._pcr_nb_levels):
            D_inv = _batched_inverse(
                identity - matmul(A, C) - matmul(C, A), xp,
                singular_mode=self._zero_index)
            # The level keeps the coefficients it eliminates *with*, so store
            # them before the stride doubles.
            self._pcr_levels.append((A, C, D_inv))
            A, C = (-matmul(D_inv, matmul(A, A)),
                    -matmul(D_inv, matmul(C, C)))
        self._pcr_final_inv = _batched_inverse(
            identity + A + C, xp, singular_mode=self._zero_index)

    def _build_corrections(self):
        """Fold the end corrections into the interface operators.

        apply() never needs the neighbours' interface planes themselves, only
        ``A2 left`` and ``A0 right`` -- what those planes contribute to this
        slab's first and last plane. Both are linear in the values the
        interface step starts from, through operators that do not change
        between applies, so multiplying ``A2`` and ``A0`` through once here
        leaves apply() one batched product per end instead of an einsum, a
        reshape, two index selections and two products. On a GPU every one of
        those is a kernel launch, and launches -- not arithmetic -- are what
        the interface step costs there (measured for the unfolded form: 0.35 ms
        per apply at 32^3 and 0.42 ms at 128^3 on one MI300A, with no MPI
        involved at all).
        """
        xp, dim = self._xp, self.dim
        A0, A2 = self.A[0], self.A[2]
        if self._use_pcr:
            # u_p = (L_p, R_p) = final_inv x_p on every rank, and rank p needs
            # A2 R_{p-1} and A0 L_{p+1}. final_inv is the same on every rank
            # (see _build_pcr), so each rank applies its *neighbours'* products
            # to its own u and sends them the result: dim numbers per mode in
            # each direction instead of 2*dim, and nothing left to do with them
            # on arrival but subtract.
            zero = xp.zeros_like(A0)
            take_R = xp.concatenate([zero, A2], axis=-1)  # A2 [0 I]
            take_L = xp.concatenate([A0, zero], axis=-1)  # A0 [I 0]
            self._pcr_to_right = xp.matmul(take_R, self._pcr_final_inv)
            self._pcr_to_left = xp.matmul(take_L, self._pcr_final_inv)
        else:
            # lambda = reduced_inv rhs, and the corrections read the right
            # plane of the rank before and the left plane of the rank after.
            m, P = 2 * dim, self.nb_ranks
            prev_R = ((self.rank - 1) % P) * m + dim
            next_L = ((self.rank + 1) % P) * m
            self._corr_first = xp.matmul(
                A2, self._reduced_inv[..., prev_R:prev_R + dim, :])
            self._corr_last = xp.matmul(
                A0, self._reduced_inv[..., next_L:next_L + dim, :])

    def _exchange(self, to_high, distance, to_low=None):
        """Send to the ranks at ``±distance`` and receive what they send back.

        ``to_high`` goes to rank ``p+distance`` and ``to_low`` -- ``to_high``
        again unless given -- to rank ``p-distance``. Returns
        ``(from_low, from_high)``: what rank ``p-distance`` sent up and what
        rank ``p+distance`` sent down, wrapping cyclically. Device buffers go
        to MPI as they are when it is GPU-aware, else through the host (see
        :meth:`_outgoing`).
        """
        if to_low is None:
            to_low = to_high
        if self._mpi is None:
            return to_high, to_low

        P = self.nb_ranks
        send_high = self._outgoing(to_high)
        send_low = send_high if to_low is to_high else self._outgoing(to_low)
        from_low = self._incoming(send_high.shape, send_high.dtype)
        from_high = self._incoming(send_low.shape, send_low.dtype)
        lo = (self.rank - distance) % P
        hi = (self.rank + distance) % P
        self._before_mpi()
        # Receive from the low side while sending to the high side, then the
        # reverse; two Sendrecvs rather than four blocking calls.
        self._mpi.Sendrecv(send_high, dest=hi, sendtag=0,
                           recvbuf=from_low, source=lo, recvtag=0)
        self._mpi.Sendrecv(send_low, dest=lo, sendtag=1,
                           recvbuf=from_high, source=hi, recvtag=1)
        return self._arrived(from_low), self._arrived(from_high)

    def _solve_reduced_pcr(self, ends):
        """The interface solve by parallel cyclic reduction.

        ``ends`` is this rank's two interface planes; the return is the pair
        ``(A2 R_{p-1}, A0 L_{p+1})`` -- the corrections to this slab's first and
        last plane -- each ``(*modes, dim, 1)``.
        """
        xp = self._xp
        x = xp.ascontiguousarray(
            ends.reshape(self._mode_shape + (2 * self.dim, 1)))

        matmul = xp.matmul
        for level, (A, C, D_inv) in enumerate(self._pcr_levels):
            from_low, from_high = self._exchange(x, 1 << level)
            x = matmul(D_inv, x - matmul(A, from_low) - matmul(C, from_high))

        # The neighbours' products, computed here and sent to them (see
        # _build_corrections); what arrives is this slab's two corrections.
        return self._exchange(matmul(self._pcr_to_right, x), 1,
                              to_low=matmul(self._pcr_to_left, x))

    def _build_zero_mode(self):
        """The all-zero mode, which the tridiagonal path cannot solve.

        Its z-operator keeps the constant-in-z nullspace -- the rigid
        translation -- so it is singular, and both the local Thomas factors and
        the reduced system degenerate there. It is a single mode, so it is
        cheaper to gather its whole z-line and solve it directly than to rescue
        the general path: that is the same treatment
        :func:`make_reference_stiffness_preconditioner` gives ``q = 0``.

        The nullspace is *deflated* rather than discovered. ``T`` is Hermitian
        and its kernel is known exactly -- the ``dim`` constant-in-z
        translations, which the projection folded into ``_zero_op`` removes
        from both ends -- so adding ``shift`` times the orthogonal projector onto them
        moves the kernel to ``shift`` and leaves the complement untouched. A
        plain inverse of that then reproduces ``T^+`` on every right-hand side
        this is given, and is better conditioned than ``T`` restricted to its
        range.

        ``pinv`` cannot do the same job reliably, because it has to separate
        kernel from range by magnitude and the gap is not one it can resolve:
        the computed zeros sit at ``~1e-16 * sigma_max``, one digit below the
        ``1e-15 * sigma_max`` cutoff. Which side of it they land on is a
        property of the LAPACK build, not of the problem -- numpy 2.5 returns
        ``3.8e-14`` for one of them at 32 planes, which keeps a kernel direction
        scaled by ``1e13`` and silently costs the preconditioner four digits of
        exactness. Nor would a looser threshold settle it: the smallest singular
        value it must *not* cut is the longest-wavelength mode along the axis,
        ``2.5e-3 * sigma_max`` at 32 planes and falling as the square of the
        grid, so the safe window closes as the problem grows.
        """
        xp = self._xp
        dim, nz = self.dim, self.nz_global
        A0, A1, A2 = (self.A[i][self._zero_index] for i in range(3))
        T = xp.zeros((nz * dim, nz * dim), dtype=self._cdtype)
        for k in range(nz):
            row = slice(k * dim, (k + 1) * dim)
            T[row, ((k + 1) % nz) * dim:((k + 1) % nz) * dim + dim] += A0
            T[row, row] += A1
            T[row, ((k - 1) % nz) * dim:((k - 1) % nz) * dim + dim] += A2

        projector = xp.zeros_like(T)
        for component in range(dim):
            projector[component::dim, component::dim] = 1.0 / nz
        self._zero_inv = xp.linalg.inv(T + xp.abs(T).max() * projector)

        # What apply() uses: Q T^+ Q, with Q = I - projector the projection off
        # the nullspace, and only this rank's rows of it. The projection is
        # needed at both ends -- the right-hand side must be orthogonal to the
        # nullspace for the system to be consistent, and the solution is only
        # defined up to it -- and both ends are linear, so they fold into the
        # matrix once here instead of costing two means and two subtractions
        # per apply. Means are reductions, the slow case on a GPU (see the
        # note above apply()), and on a mode this small every launch shows.
        complement = xp.eye(nz * dim, dtype=self._cdtype) - projector
        start = self.rank * self.nz_local * dim
        self._zero_op = xp.ascontiguousarray(
            (complement @ self._zero_inv @ complement)[
                start:start + self.nz_local * dim])

    def _solve_zero_mode(self, v):
        """``T^+ v`` for the all-zero mode, over the whole distributed axis.

        Returns this rank's planes of the solution, ``(nz_local, dim)``.
        """
        local = self._xp.ascontiguousarray(
            v[(slice(None),) + self._zero_index][..., 0])
        full = self._allgather(local).reshape(-1)
        return (self._zero_op @ full).reshape(self.nz_local, self.dim)

    # -- collectives -------------------------------------------------------- #

    def _allgather(self, local):
        """Gather one array per rank along a new leading axis."""
        if self._mpi is None:
            return local[None]
        send = self._outgoing(local)
        out = self._incoming((self.nb_ranks,) + send.shape, send.dtype)
        self._before_mpi()
        self._mpi.Allgather(send, out)
        return self._arrived(out)

    # Buffers for MPI. A host array, or a device array when MPI is GPU-aware
    # (_mpi_direct), is handed over as it is. Otherwise the device buffer is
    # staged through host memory, which works with any MPI build but costs a
    # synchronising copy each way per message -- measured as the dominant
    # cost of the whole apply on MI300A once there are two or more GPUs.

    def _outgoing(self, a):
        """``a`` as a contiguous buffer MPI can read."""
        if self._mpi_direct:
            return self._xp.ascontiguousarray(a)
        return np.ascontiguousarray(a.get())

    def _incoming(self, shape, dtype):
        """A buffer MPI can write ``shape`` elements of ``dtype`` into."""
        return (self._xp if self._mpi_direct else np).empty(shape, dtype=dtype)

    def _arrived(self, buffer):
        """A received buffer as an array of this preconditioner's module."""
        return buffer if self._mpi_direct else self._xp.asarray(buffer)

    def _before_mpi(self):
        """Let the kernels that fill device send buffers finish.

        MPI reads a device buffer outside the stream it was written on, so the
        stream must be drained first; the staged path needs no such step,
        because the copy to the host already synchronises.
        """
        if self._on_device and self._mpi_direct:
            self._xp.cuda.get_current_stream().synchronize()

    def _timed(self, label):
        return self._timer(label) if self._timer is not None else nullcontext()

    # Why the projection off the nullspace happens on the zero mode's z-line,
    # folded into _zero_op, and not on the field: the rigid translations are
    # exactly the ``q = 0`` coefficient, and under this transform that is the
    # z-mean of the all-zero mode's line -- ``nz * dim`` numbers rather than the
    # whole field. Doing it in real space instead needs a reduction over every
    # point, which is the one thing to avoid here: cupy reductions on this ROCm
    # build run at ~6 GB/s against ~3200 GB/s for elementwise work, so a global
    # mean of a 256**3 field costs more than the rest of the apply put together.

    # -- the preconditioner ------------------------------------------------- #

    def apply(self, r, z):
        """``z = M r``."""
        with self._on_own_device():
            self._apply(r, z)

    def _layout(self, field):
        """How to lay a ``(dim, *local, nz)`` field view out for the transforms.

        Returns ``(perm, axes, components_first)``: ``field.transpose(perm)``
        orders the axes as they lie in memory but with the distributed one
        first, ``axes`` are the rank-local axes in that order (to transform,
        the last of them ``x``, halved), and ``components_first`` says whether
        the components sit *before* the local axes in it.

        muGrid stores fields component-fastest on the host (``(nz, ..., y, x,
        dim)`` in memory) and component-slowest on a device (``(dim, nz, ...,
        y, x)``); either way ``x`` varies fastest of the spatial axes.
        Transforming in the logical ``(x, y)`` order instead runs over axes
        that are neither trailing nor contiguous, which no single batched plan
        describes (measured on MI300A at 256^3: twice the cost of the reference
        preconditioner's full 3D transform). Following memory order, the copy
        into the transform buffer moves at most whole planes (the device
        layout's component and ``z`` blocks change places), and on the host
        nothing at all but the ghost layers it drops.
        """
        dim = self.dim
        local_reversed = tuple(range(dim - 1, 0, -1))
        if field.strides[0] <= field.strides[1]:
            # host: (nz, ..., y, x, dim); the result is already z-major
            return ((dim,) + local_reversed + (0,),
                    tuple(range(1, dim)), False)
        # device: (nz, dim, ..., y, x)
        return (dim, 0) + local_reversed, tuple(range(2, dim + 1)), True

    def _apply(self, r, z):
        xp = self._xp
        dim = self.dim
        source = xp.asarray(r.p)
        perm, axes, components_first = self._layout(source)

        with self._timed("fft"):
            staged = xp.ascontiguousarray(source.transpose(perm))
            hat = xp.fft.rfftn(staged, axes=axes).astype(
                self._cdtype, copy=False)
            # -> (nz, *modes, dim); see _factorise_local for why the
            # distributed axis leads. On the host the transform leaves it so.
            if components_first:
                hat = xp.ascontiguousarray(xp.moveaxis(hat, 1, -1))
            v = hat[..., None]

        with self._timed("tridiag"):
            x = self._solve_local(v)

        with self._timed("interface"):
            ends = xp.stack([x[0, ..., 0], x[-1, ..., 0]], axis=-2)
            # first = A2 left, last = A0 right: the neighbours' interface
            # planes as they enter this slab's two end planes, each
            # (*modes, dim, 1). See _build_corrections.
            if self._use_pcr:
                first, last = self._solve_reduced_pcr(ends)
            else:
                if self.nb_ranks == 1:
                    rhs = ends.reshape(self._mode_shape + (2 * dim, 1))
                else:
                    gathered = self._allgather(xp.ascontiguousarray(ends))
                    rhs = xp.moveaxis(gathered, 0, -3).reshape(
                        self._mode_shape + (2 * dim * self.nb_ranks, 1))
                first = xp.matmul(self._corr_first, rhs)
                last = xp.matmul(self._corr_last, rhs)
            # From the *uncorrected* right-hand side: the all-zero mode is
            # solved globally rather than through the interface system, so the
            # end corrections below would double-count it.
            zero_solution = self._solve_zero_mode(v)

        with self._timed("tridiag"):
            # z_local = T^-1 (rhs - A2 left e_0 - A0 right e_last): the
            # neighbours' interface planes enter as a correction to the two end
            # planes of the right-hand side, so the spikes never have to be
            # stored or re-read.
            v[0] -= first
            v[-1] -= last
            sol = self._solve_local(v)[..., 0]

        with self._timed("ifft"):
            sol[(slice(None),) + self._zero_index] = zero_solution
            # Back the same way: into the transform layout, the inverse over
            # the same axes, and out through the inverse permutation, which
            # again follows memory order on both sides of the assignment.
            if components_first:
                sol = xp.ascontiguousarray(xp.moveaxis(sol, -1, 1))
            out = xp.fft.irfftn(sol, axes=axes,
                                s=tuple(reversed(self.local_shape)))
            # irfftn already returns a real array, so no xp.real() copy is
            # needed; the assignment does any cast. It goes through z's view
            # in the same memory order as `out`: an elementwise copy walks its
            # output in logical order, and z's logical order (dim, x, y, z) is
            # the opposite of its memory order, so `z.p[...] = out.transpose()`
            # would put neighbouring threads far apart on both sides (measured:
            # 8 ms per apply at 256^3 on MI300A, three times the inverse FFT
            # itself).
            xp.asarray(z.p).transpose(perm)[...] = out
