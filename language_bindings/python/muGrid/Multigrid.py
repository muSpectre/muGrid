"""
Geometric multigrid preconditioner for heterogeneous nodal operators.

A symmetric V(ν, ν)-cycle with Chebyshev-Jacobi smoothing, to be used as the
preconditioner of :func:`muGrid.Solvers.conjugate_gradients`. It is the port of
the multigrid of muFFTTO (``muFFTTO/multigrid.py``) with two differences:

* **Galerkin coarse operators, measured by probing.** The multigrid knows the
  fine operator only through its action ``apply(x, y)``. The first coarse
  operator ``A₁ = R A₀ P`` is measured by applying ``A₀`` to prolongated,
  coloured coarse unit vectors; the coarser ones are formed algebraically,
  ``A_{l+1} = R A_l P``, from the explicit stencils. No element, quadrature
  rule or material model enters the hierarchy.
* **A guaranteed bound for the smoother.** The Chebyshev interval
  ``[b/α, b]`` needs ``b ≥ λmax(D⁻¹A)`` on every level. It is computed, not
  estimated (see :class:`ScaledLocalMatrices`).

Hierarchy
---------
Grids N, N/2, …, ``n_coarsest``. ``P`` is multilinear interpolation acting
component-wise (:class:`muGrid.GridTransfer`), ``R = Pᵀ``. Coarse operators are
:class:`muGrid.VariableStencilOperator` stencils: a symmetric ``3^d`` stencil
of ``nc × nc`` blocks at every node, half of the blocks stored.

This module is built up stage by stage (``docs/heterogeneous_multigrid.md``).
"""

import itertools
import warnings
from contextlib import nullcontext

import numpy as np

from . import linalg
from .Parallel import Communicator
from .Preconditioners import Preconditioner
from .Wrappers import (CartesianDecomposition, GridTransfer,
                       VariableStencilOperator, _muGrid)

# --------------------------------------------------------------------------- #
# Stencil layout (one definition: the C++ operator's, read through bindings)
# --------------------------------------------------------------------------- #


class StencilLayout:
    """Where each block of a symmetric nodal ``3^d`` stencil is stored.

    Mirrors ``VariableStencilOperator``: block ``k`` of node ``n`` holds
    ``A(n, n + s_k)`` for the stored ("positive") offsets ``s_k``, row-major in
    entries ``k nc² … (k+1) nc² - 1``; the centre block ``A(n, n)`` follows as
    its upper triangle, row by row.
    """

    def __init__(self, dim, nb_components):
        cpp = (_muGrid.VariableStencilOperator2D if dim == 2
               else _muGrid.VariableStencilOperator3D)
        self.dim = dim
        self.nb_components = nc = int(nb_components)
        self.half_offsets = [tuple(int(o) for o in s)
                             for s in cpp.half_offsets]
        self.index = {s: k for k, s in enumerate(self.half_offsets)}
        self.centre = len(self.half_offsets) * nc * nc
        #: entry of the centre block (i, j), i <= j
        self.centre_entry = {}
        t = self.centre
        for i in range(nc):
            for j in range(i, nc):
                self.centre_entry[(i, j)] = t
                t += 1
        self.nb_entries = t

    def block_entry(self, k, i, j):
        """Entry of ``A(n, n + s_k)_ij``."""
        return k * self.nb_components ** 2 + i * self.nb_components + j

    def offset_kind(self, s):
        """``('half', k)``, ``('centre', None)`` or ``('mirror', k)`` for an
        offset ``s ∈ {-1, 0, 1}^d``; a mirrored offset ``-s_k`` is stored as
        the transposed block of the neighbour."""
        s = tuple(s)
        if s in self.index:
            return "half", self.index[s]
        if all(o == 0 for o in s):
            return "centre", None
        return "mirror", self.index[tuple(-o for o in s)]


# --------------------------------------------------------------------------- #
# Decomposition helpers (CartesianDecomposition or FFTEngine)
# --------------------------------------------------------------------------- #


def _collection(decomposition):
    """The field collection of a CartesianDecomposition or an FFTEngine."""
    collection = getattr(decomposition, "real_space_collection", None)
    return collection if collection is not None else decomposition.collection


def _communicator(decomposition, communicator=None):
    """The communicator of a run: the one given, the FFT engine's, or serial
    (CartesianDecomposition does not keep its communicator)."""
    if communicator is None:
        communicator = getattr(decomposition, "communicator", None)
    return Communicator(communicator) if communicator is not None \
        else Communicator()


def _real_field(decomposition, name, nb_components, dtype=np.float64):
    """Get-or-create a nodal real field with ``nb_components`` components."""
    return _collection(decomposition).real_field(
        name, (int(nb_components),), dtype=dtype)


def _array_module(field):
    """numpy for host fields, cupy for device fields."""
    if type(field.s).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np


def _to_host(values):
    """A numpy copy (or view) of a host or device array."""
    return values.get() if hasattr(values, "get") else np.asarray(values)


def _ghosts_left(field):
    """Ghost layers on the low side of every axis of ``field``'s collection."""
    return tuple(int(g) for g in field._cpp.collection.nb_ghosts_left)


# --------------------------------------------------------------------------- #
# Colouring for probing
# --------------------------------------------------------------------------- #


def axis_colouring(n):
    """Colours of the ``n`` points of a periodic axis such that any two points
    at periodic distance 1 or 2 differ.

    Then every point ``J`` has, for each colour, at most one point of that
    colour among ``J - 1, J, J + 1``. Period 3 when ``n`` is divisible by 3,
    period 4 when by 4 (power-of-two grids), otherwise blocks of 3 and 4
    (4 colours; 5 when ``n = 5``). A periodic colouring makes the nodes of one
    colour a regular lattice, which the probing reads with strided views. The
    colouring depends only on the global index, so it is the same for every
    MPI decomposition.
    """
    n = int(n)
    if n < 3:
        raise ValueError(
            f"a periodic axis of {n} point(s) cannot carry a 3-point stencil "
            "without aliasing its neighbours; grids must have >= 3 points")
    if n % 3 == 0:
        colours = [0, 1, 2] * (n // 3)
    elif n % 4 == 0:                                  # periodic as well
        colours = [0, 1, 2, 3] * (n // 4)
    elif n == 5:
        colours = [0, 1, 2, 3, 4]
    elif n % 3 == 1:                                  # n = 3a + 4
        colours = [0, 1, 2] * ((n - 4) // 3) + [0, 1, 2, 3]
    else:                                             # n = 3a + 8
        colours = [0, 1, 2] * ((n - 8) // 3) + [0, 1, 2, 3] * 2
    colours = np.asarray(colours)
    for d in (1, 2):
        assert np.all(colours != np.roll(colours, d)), (n, colours)
    return colours


# --------------------------------------------------------------------------- #
# Galerkin coarse operators
# --------------------------------------------------------------------------- #


def galerkin_by_probing(apply_fine, transfer, fine_decomposition,
                        coarse_decomposition, layout, stencil,
                        name="mg-probe", dtype=np.float64, fine_vectors=None):
    """Measure ``R A P`` into the coarse stencil field ``stencil``.

    ``apply_fine(x, y)`` computes ``y = A x`` on nodal fields of
    ``fine_decomposition`` (exchanging the ghosts of ``x`` itself). The coarse
    nodes are coloured per axis (:func:`axis_colouring`); for every colour
    ``c`` and component ``j`` the probe ``e`` is 1 in component ``j`` of the
    nodes of colour ``c``. At every coarse node ``J``, ``(R A P e)(J)`` is then
    block column ``j`` of ``A(J, I)`` for the one node ``I`` of colour ``c``
    next to ``J``. Blocks of the stored offsets and of the centre are kept;
    the mirrored ones are redundant (symmetry).

    Cost: ``nc ∏_α k_α`` fine applies, ``k_α`` the colours of axis ``α`` (192
    in 3D elasticity on power-of-two grids). The stencil field's ghosts are
    exchanged at the end.

    ``dtype`` is the precision of the probe vectors, i.e. of ``apply_fine``;
    the measured blocks are written into ``stencil`` in its own precision.

    ``fine_vectors``, optional: two nodal fields of ``fine_decomposition`` in
    ``dtype`` to hold ``P e`` and ``A P e`` (e.g. a multigrid level's own work
    vectors, free while the setup runs). Without them two fields are created
    (named ``<name>-Pe``, ``<name>-APe``). Their content on entry does not
    matter: the prolongation and ``apply_fine`` overwrite the interior, and
    ghosts are exchanged before they are read.
    """
    nc = layout.nb_components
    dim = layout.dim
    coarse_pts = tuple(int(n) for n in coarse_decomposition.nb_domain_grid_pts)
    location = tuple(int(o) for o in coarse_decomposition.subdomain_locations)
    local = tuple(int(n) for n in coarse_decomposition.nb_subdomain_grid_pts)

    probe = _real_field(coarse_decomposition, f"{name}-e", nc, dtype)
    if fine_vectors is None:
        fine_in = _real_field(fine_decomposition, f"{name}-Pe", nc, dtype)
        fine_out = _real_field(fine_decomposition, f"{name}-APe", nc, dtype)
    else:
        fine_in, fine_out = fine_vectors
        if any(np.dtype(v.dtype) != np.dtype(dtype) for v in fine_vectors):
            raise ValueError("fine_vectors must have the probe precision "
                             f"{np.dtype(dtype)}")
    response = _real_field(coarse_decomposition, f"{name}-RAPe", nc, dtype)
    xp = _array_module(probe)

    # Per axis: the colour of every local node, and for every colour and
    # neighbour offset t the local nodes J whose neighbour J + t has it.
    colours = [axis_colouring(n) for n in coarse_pts]
    local_colour = [colours[a][(location[a] + np.arange(local[a])) % coarse_pts[a]]
                    for a in range(dim)]
    nodes_with = []
    for a in range(dim):
        table = {}
        for c in range(colours[a].max() + 1):
            for t in (-1, 0, 1):
                neighbour = colours[a][(location[a] + np.arange(local[a]) + t)
                                       % coarse_pts[a]]
                table[(c, t)] = xp.asarray(np.flatnonzero(neighbour == c))
        nodes_with.append(table)

    values = stencil.s                       # (nb_entries, 1, *local)
    values[...] = 0
    # A periodic colouring (period p | n on every axis) puts the nodes whose
    # neighbour J + t has colour c on the lattice J ≡ c - t (mod p): one
    # strided view per offset. Otherwise index arrays (np.ix_).
    periods = [len(c) // (len(c) // (c.max() + 1))
               if np.array_equal(c, np.tile(np.arange(c.max() + 1),
                                            len(c) // (c.max() + 1)))
               else None for c in colours]
    periodic = all(p is not None for p in periods)

    def nodes(colour, t):
        if periodic:
            return tuple(slice((c - o - loc) % p, None, p) for c, o, loc, p
                         in zip(colour, t, location, periods))
        index = xp.ix_(*(nodes_with[a][(colour[a], t[a])]
                         for a in range(dim)))
        return None if any(len(i) == 0 for i in index) else index

    for colour in itertools.product(*(range(c.max() + 1) for c in colours)):
        mask = np.ones(local, dtype=bool)
        for a in range(dim):
            shape = [1] * dim
            shape[a] = local[a]
            mask = mask & (local_colour[a] == colour[a]).reshape(shape)
        mask = xp.asarray(mask)
        for j in range(nc):
            probe.set_zero()
            probe.s[j, 0][mask] = 1.0
            coarse_decomposition.communicate_ghosts(probe)
            transfer.prolong(probe, fine_in)
            apply_fine(fine_in, fine_out)
            fine_decomposition.communicate_ghosts(fine_out)
            transfer.restrict(fine_out, response)
            column = response.s[:, 0]        # (nc, *local)
            for t in itertools.product((-1, 0, 1), repeat=dim):
                kind, k = layout.offset_kind(t)
                if kind == "mirror":
                    continue
                index = nodes(colour, t)
                if index is None:
                    continue
                if kind == "half":           # entries k nc² + i nc + j, all i
                    first = layout.block_entry(k, 0, j)
                    if periodic:
                        values[first:first + nc * nc:nc, 0][
                            (slice(None),) + index] = column[
                                (slice(None),) + index]
                    else:
                        for i in range(nc):
                            values[first + i * nc, 0][index] = \
                                column[i][index]
                    continue
                for i in range(j + 1):       # centre: upper triangle i <= j
                    entry = layout.centre_entry[(i, j)]
                    values[entry, 0][index] = column[i][index]
    coarse_decomposition.communicate_ghosts(stencil)


def _per_axis_options(m):
    """Fine offset ``m = a + s ∈ {-2, …, 2}`` of one axis split into the coarse
    neighbour offset ``t`` and the position ``b = m - 2 t`` within the support
    of the coarse basis function, with the interpolation weight of ``b``."""
    return {-2: [(-1, 1.0)],
            -1: [(0, 0.5), (-1, 0.5)],
            0: [(0, 1.0)],
            1: [(0, 0.5), (1, 0.5)],
            2: [(1, 1.0)]}[m]


def galerkin_rap(fine_stencil, fine_decomposition, coarse_stencil,
                 coarse_decomposition, layout, name="mg-rap"):
    """``coarse_stencil = R A P`` for the explicit fine stencil ``fine_stencil``.

    Formula, for coarse nodes ``J`` and ``I = J + t``::

        A_H(J, I) = Σ_{a, s} w(a) w(b) A_h(2J + a, 2J + a + s),
        2J + a + s = 2I + b,   a, b, s ∈ {-1, 0, 1}^d,

    with the multilinear weights ``w(a) = ∏_α 2^-|a_α|`` of ``P``. Done one
    fine offset ``s`` at a time: the block ``A_h(n, n + s)`` of every fine node
    is gathered into a work field (the transposed neighbour block for a
    mirrored ``s``), its ghosts exchanged, and its contributions added to the
    coarse blocks of the stored offsets and the centre, accumulated in double
    precision. No fine operator applies; ``O(27² nc²)`` operations per coarse
    node.
    """
    nc = layout.nb_components
    dim = layout.dim
    nn = nc * nc
    block = _real_field(fine_decomposition, f"{name}-block", nn, np.float64)
    xp = _array_module(block)
    fine_values = fine_stencil.sg            # (entries, 1, *with ghosts)
    fine_ghosts = _ghosts_left(fine_stencil)
    fine_local = tuple(fine_stencil.s.shape[2:])
    coarse_local = tuple(coarse_stencil.s.shape[2:])
    if any(f != 2 * c for f, c in zip(fine_local, coarse_local)):
        raise ValueError(f"grids are not nested: fine {fine_local}, coarse "
                         f"{coarse_local}")

    def fine_interior(shift):
        """index of the fine stencil at the interior nodes n + shift"""
        return tuple(slice(g + o, g + o + n)
                     for g, o, n in zip(fine_ghosts, shift, fine_local))

    halves = xp.zeros((len(layout.half_offsets), nn) + coarse_local)
    centre = xp.zeros((nn,) + coarse_local)
    block_ghosts = _ghosts_left(block)
    for s in itertools.product((-1, 0, 1), repeat=dim):
        kind, k = layout.offset_kind(s)
        target = block.s[:, 0]
        for i in range(nc):
            for j in range(nc):
                if kind == "half":
                    target[i * nc + j] = fine_values[
                        (layout.block_entry(k, i, j), 0) + fine_interior(
                            (0,) * dim)]
                elif kind == "centre":
                    entry = layout.centre_entry[(min(i, j), max(i, j))]
                    target[i * nc + j] = fine_values[
                        (entry, 0) + fine_interior((0,) * dim)]
                else:                        # A(n, n - s_k) = A_k(n - s_k)^T
                    target[i * nc + j] = fine_values[
                        (layout.block_entry(k, j, i), 0) + fine_interior(s)]
        fine_decomposition.communicate_ghosts(block)
        source = block.sg[:, 0]               # (nn, *with ghosts)
        for a in itertools.product((-1, 0, 1), repeat=dim):
            weight_a = 0.5 ** sum(1 for o in a if o)
            view = source[(slice(None),) + tuple(
                slice(g + o, g + o + 2 * n, 2)
                for g, o, n in zip(block_ghosts, a, coarse_local))]
            options = [_per_axis_options(o + so) for o, so in zip(a, s)]
            for choice in itertools.product(*options):
                t = tuple(c[0] for c in choice)
                weight = weight_a * float(np.prod([c[1] for c in choice]))
                kind_t, k_t = layout.offset_kind(t)
                if kind_t == "half":
                    halves[k_t] += weight * view
                elif kind_t == "centre":
                    centre += weight * view
    values = coarse_stencil.s
    values[...] = 0
    for k in range(len(layout.half_offsets)):
        for i in range(nc):
            for j in range(nc):
                values[layout.block_entry(k, i, j), 0] = halves[k, i * nc + j]
    for (i, j), entry in layout.centre_entry.items():
        values[entry, 0] = 0.5 * (centre[i * nc + j] + centre[j * nc + i])
    coarse_decomposition.communicate_ghosts(coarse_stencil)


def make_stencil_field(decomposition, layout, name, dtype=np.float64):
    """A stencil field (``layout.nb_entries`` components) on ``decomposition``."""
    return _real_field(decomposition, name, layout.nb_entries, dtype)


def stencil_operator(layout):
    """The :class:`muGrid.VariableStencilOperator` of ``layout``."""
    return VariableStencilOperator(layout.dim, layout.nb_components)


# --------------------------------------------------------------------------- #
# Scaled local matrices: the structure behind the guaranteed smoother bound
# --------------------------------------------------------------------------- #


def _corner_index(dim):
    """Corners ``a ∈ {0, 1}^d`` of a pixel in local order (x fastest)."""
    return [tuple((c >> a) & 1 for a in range(dim)) for c in range(2 ** dim)]


def _interpolation_from_cell(dim, position, level):
    """``W[a, c]``: weight of corner ``c`` of a level-``level`` cell in fine
    corner ``a`` of the fine pixel at ``position ∈ {0 … 2^level - 1}^d`` inside
    it (multilinear, the composite prolongation of ``level`` coarsenings)."""
    corners = _corner_index(dim)
    W = np.empty((len(corners), len(corners)))
    for ia, a in enumerate(corners):
        xi = [(p + o) / 2 ** level for p, o in zip(position, a)]
        for ic, c in enumerate(corners):
            W[ia, ic] = np.prod([x if cc else 1.0 - x
                                 for x, cc in zip(xi, c)])
    return W


def _interpolations_from_cell(positions, level, nb_components):
    """:func:`_interpolation_from_cell` for many positions at once, with the
    components: ``T[p, a nc + i, c nc + j] = W_p[a, c] δ_ij``."""
    positions = np.asarray(positions, dtype=float)
    dim = positions.shape[1]
    corners = np.array(_corner_index(dim), dtype=float)          # (2^d, d)
    xi = (positions[:, None, :] + corners[None]) / 2 ** level    # (P, a, d)
    W = np.where(corners[None, None, :, :] == 1, xi[:, :, None, :],
                 1.0 - xi[:, :, None, :]).prod(axis=-1)          # (P, a, c)
    nc = nb_components
    T = np.einsum("pac,ij->paicj", W, np.eye(nc))
    return T.reshape(len(positions), len(corners) * nc, len(corners) * nc)


def _max_scaled_eigenvalue(matrices):
    """``max λmax(diag(M)⁻¹ M)`` over a batch of symmetric PSD matrices.

    Computed as λmax of the symmetric ``D^-1/2 M D^-1/2``. A zero diagonal
    entry of a PSD matrix comes with a zero row; it is given D = 1, which adds
    an eigenvalue 0 and changes nothing else.
    """
    diagonal = np.einsum("...ii->...i", matrices)
    scale = np.abs(diagonal).max(axis=-1, keepdims=True)
    safe = np.where(diagonal > 1e-14 * scale, diagonal, 1.0)
    inv_sqrt = 1.0 / np.sqrt(safe)
    scaled = inv_sqrt[..., :, None] * matrices * inv_sqrt[..., None, :]
    scaled = 0.5 * (scaled + np.swapaxes(scaled, -1, -2))
    return float(np.linalg.eigvalsh(scaled)[..., -1].max())


class ScaledLocalMatrices:
    r"""The fine operator as a sum of scaled, fixed local matrices.

    States that ``A = Σ_pixels Σ_k s_k(pixel) K_k``: every pixel ``p``
    contributes the fixed symmetric positive semidefinite matrices ``K_k`` on
    the ``2^d`` corner nodes ``p + a`` of the pixel, scaled by ``s_k(p) ≥ 0``.
    This is the structure of every FE operator ``Bᵀ W C(x) B`` whose material
    is a scaled reference, ``C(x) = s(x) C₀`` (one family per quadrature point
    if ``s`` varies per point), plus local terms such as hourglass control.

    It makes the bound of the Chebyshev interval rigorous and independent of
    the material (mediant inequality): if ``A = Σ_e A_e`` with ``A_e ⪰ 0`` and
    ``D = Σ_e diag(A_e)``, then ``λmax(D⁻¹A) ≤ max_e λmax(diag(A_e)⁻¹ A_e)``.
    A Galerkin level ``l`` is ``Σ T_posᵀ (s K_k) T_pos``, with ``T_pos`` the
    multilinear interpolation from the corners of a level-``l`` cell to the
    fine pixel at position ``pos`` inside it, so

        ``c_l = max_{k, pos} λmax(diag(M)⁻¹ M)``,  ``M = T_posᵀ K_k T_pos``,

    bounds ``λmax(D_l⁻¹ A_l)`` for every ``s ≥ 0``. It is computed once per
    level (:meth:`bounds`) and tight over the family: ``s`` concentrated on one
    pixel attains it.

    Safeguards: :meth:`check_operator` compares the operator with the local
    sum on random patches (the multigrid calls it once, at construction), and
    :meth:`check_nonnegative` checks ``s ≥ 0`` (at every setup).

    Parameters
    ----------
    matrices : array_like
        ``(n_k, 2^d nc, 2^d nc)``. Row and column ``a nc + i``: component ``i``
        of corner ``a`` (``a = Σ_α a_α 2^α``, x fastest).
    scales : muGrid.Field
        The ``s_k`` per pixel, on the fine collection: ``n_k`` values per pixel
        (components × sub-points, in that order). The value of pixel ``p``
        belongs to the element whose lowest corner is node ``p``.
    """

    #: Most atoms (n_k × positions) a level bound is computed from (2^15: 1.5 s
    #: once, 3D level 5); deeper levels use the Gershgorin bound of their
    #: stencil alone, which is also guaranteed and there the tighter one.
    MAX_ATOMS = 2 ** 15

    def __init__(self, matrices, scales):
        matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError(f"matrices must be (n_k, m, m); got "
                             f"{matrices.shape}")
        asymmetry = np.abs(matrices - np.swapaxes(matrices, 1, 2)).max()
        if asymmetry > 1e-12 * np.abs(matrices).max():
            raise ValueError("the local matrices must be symmetric")
        lowest = min(float(np.linalg.eigvalsh(m)[0]) / max(
            float(np.abs(m).max()), 1e-300) for m in matrices)
        if lowest < -1e-10:
            raise ValueError("the local matrices must be positive "
                             f"semidefinite (relative eigenvalue {lowest:.2e})")
        self.matrices = matrices
        self.scales = scales
        self._bounds = {}

    @property
    def nb_families(self):
        return self.matrices.shape[0]

    def scale_values(self):
        """``(n_k, *local)`` view of the scales (interior pixels)."""
        values = self.scales.s
        return values.reshape((-1,) + tuple(values.shape[-self._dim():]))

    def _dim(self):
        return len(_ghosts_left(self.scales))

    # -- construction helpers ------------------------------------------------

    @classmethod
    def isotropic_stiffness(cls, operator, lambda_0, mu_0, scale):
        """``A = s(x) K(λ₀, μ₀)`` of the fused
        :class:`muGrid.IsotropicStiffnessOperator` with ``λ = s λ₀``,
        ``μ = s μ₀`` per pixel (one family). The pixel matrix is probed from
        the operator itself, on one pixel of a small periodic grid."""
        dim = len(_ghosts_left(scale))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*serial muGrid Communicator.*")
            decomposition = CartesianDecomposition(
                _serial_communicator(), [4] * dim, nb_subdivisions=[1] * dim,
                nb_ghosts_left=[1] * dim, nb_ghosts_right=[1] * dim)
        lam = decomposition.real_field("probe-lambda")
        mu = decomposition.real_field("probe-mu")
        pixel = (1,) * dim
        lam.s[(0,) + pixel] = lambda_0
        mu.s[(0,) + pixel] = mu_0
        decomposition.communicate_ghosts(lam)
        decomposition.communicate_ghosts(mu)

        def apply(u, f):
            decomposition.communicate_ghosts(u)
            operator.apply(u, lam, mu, f)

        return cls(probe_pixel_matrix(apply, decomposition, dim, pixel)[None],
                   scale)

    @classmethod
    def fem(cls, gradient, tensor, scale):
        """``A = Bᵀ W (s_q C) B`` with the gradient ``B`` of a
        :class:`muGrid.FEMGradientOperator` and a fixed tensor ``C`` of shape
        ``(nc, d, nc, d)`` mapping the gradient ``∂_l u_j`` to the flux
        ``σ_ik``; one family per quadrature point,
        ``K_q = W_q B_qᵀ C B_q``."""
        coefficients = np.asarray(gradient.coefficients)
        weights = np.asarray(gradient.quadrature_weights)
        dim = coefficients.shape[0]
        tensor = np.asarray(tensor, dtype=np.float64)
        nc = tensor.shape[0]
        corners = _corner_index(dim)
        matrices = []
        for q in range(len(weights)):
            B = np.array([[coefficients[(alpha, q, 0) + a] for a in corners]
                          for alpha in range(dim)])        # (d, corners)
            K = weights[q] * np.einsum("ka,ikjl,lb->aibj", B, tensor, B)
            matrices.append(K.reshape(len(corners) * nc, len(corners) * nc))
        return cls(np.array(matrices), scale)

    # -- the bound -------------------------------------------------------------

    def bound(self, level):
        """``c_l``, or ``None`` when level ``l`` has more than
        :data:`MAX_ATOMS` atoms."""
        if level not in self._bounds:
            dim = self._dim()
            nc = self.matrices.shape[1] // 2 ** dim
            nb_atoms = self.nb_families * 2 ** (dim * level)
            if nb_atoms > self.MAX_ATOMS:
                self._bounds[level] = None
            else:
                best = 0.0
                positions = np.array(list(itertools.product(
                    range(2 ** level), repeat=dim)), dtype=float)
                chunk = max(1, 2 ** 14 // self.nb_families)
                for start in range(0, len(positions), chunk):
                    T = _interpolations_from_cell(
                        positions[start:start + chunk], level, nc)
                    # M = Tᵀ K T for every (position, family): batched
                    # products (a 4-operand einsum would loop naively)
                    KT = np.matmul(self.matrices[None], T[:, None])
                    M = np.matmul(np.swapaxes(T, -1, -2)[:, None], KT)
                    best = max(best, _max_scaled_eigenvalue(
                        M.reshape((-1,) + M.shape[-2:])))
                self._bounds[level] = best
        return self._bounds[level]

    # -- safeguards ----------------------------------------------------------

    def check_nonnegative(self, communicator=None):
        """Raise unless every ``s_k ≥ 0`` (global minimum)."""
        values = self.scale_values()
        local = float(values.min()) if values.size else np.inf
        if communicator is not None:
            local = float(communicator.min(local))
        if not local >= 0.0:
            raise ValueError(
                f"the material scales must be >= 0 everywhere for the "
                f"multigrid's smoother bound to hold; the minimum is {local}")

    def apply_local(self, x, pixels):
        """``Σ_{p ∈ pixels} s_k(p) K_k x`` for a host array ``x`` of interior
        nodal values (pixel indices periodic within the local subdomain); used
        to check the operator."""
        dim = self._dim()
        nc = self.matrices.shape[1] // 2 ** dim
        corners = _corner_index(dim)
        local = tuple(x.shape[2:])
        y = np.zeros_like(x)
        s = _to_host(self.scale_values())
        for p in pixels:
            gather = [tuple((pp + aa) % n for pp, aa, n in zip(p, a, local))
                      for a in corners]
            values = np.concatenate([x[(slice(None), 0) + g] for g in gather])
            for k in range(self.nb_families):
                out = s[(k,) + tuple(p)] * (self.matrices[k] @ values)
                for ia, g in enumerate(gather):
                    y[(slice(None), 0) + g] += out[ia * nc:(ia + 1) * nc]
        return y

    def check_operator(self, apply, decomposition, nb_components,
                       communicator=None, nb_patches=3, seed=0, rtol=1e-8):
        """Compare ``apply`` with the local sum on random patches.

        A random vector supported on ``3^d`` nodes is applied; the result must
        equal ``Σ s_k K_k x`` from the ``4^d`` pixels touching the patch, and
        vanish elsewhere. Serial runs, or ranks whose subdomain is at least 7
        nodes wide (the patch then does not reach the ghosts), check;
        the verdict is collective.
        """
        comm = _communicator(decomposition, communicator)
        try:
            return self._check_operator(apply, decomposition, nb_components,
                                        comm, nb_patches, seed, rtol)
        finally:
            # two vectors of the fine grid, needed only here: freed, not kept
            # for the whole run (their memory goes once x and y are gone)
            collection = _collection(decomposition)
            for name in ("mg-check-x", "mg-check-y"):
                if collection.field_exists(name):
                    collection.pop_field(name)

    def _check_operator(self, apply, decomposition, nb_components, comm,
                        nb_patches, seed, rtol):
        x = _real_field(decomposition, "mg-check-x", nb_components)
        y = _real_field(decomposition, "mg-check-y", nb_components)
        local = tuple(x.s.shape[2:])
        dim = len(local)
        # A patch must not reach the ghosts unless the subdomain is the whole
        # (periodic) grid. Every rank applies the operator equally often --
        # the apply exchanges ghosts, a collective -- and ranks without room
        # for a patch apply it to zero and compare nothing.
        can_check = comm.size == 1 or min(local) >= 7
        rng = np.random.default_rng(seed + comm.rank)
        xp = _array_module(x)
        worst = 0.0
        for _ in range(nb_patches):
            values = np.zeros(x.s.shape)
            if can_check:
                centre = [int(rng.integers(3, n - 3)) if n >= 7 else
                          int(rng.integers(0, n)) for n in local]
                for off in itertools.product((-1, 0, 1), repeat=dim):
                    node = tuple((c + o) % n
                                 for c, o, n in zip(centre, off, local))
                    values[(slice(None), 0) + node] = rng.standard_normal(
                        nb_components)
            x.set_zero()
            x.s[...] = values if xp is np else xp.asarray(values)
            apply(x, y)
            if not can_check:
                continue
            got = _to_host(y.s)
            pixels = [tuple((c + o) % n for c, o, n in zip(centre, off, local))
                      for off in itertools.product((-2, -1, 0, 1), repeat=dim)]
            expected = self.apply_local(values, pixels)
            scale = max(float(np.abs(expected).max()), 1e-300)
            worst = max(worst, float(np.abs(got - expected).max()) / scale)
        worst = float(comm.max(worst))
        if worst > rtol:
            raise ValueError(
                "the operator is not Σ_pixels s_k K_k with the given local "
                f"matrices and scales (relative mismatch {worst:.2e} on a "
                "random patch). The constant smoother bound would not be "
                "valid; pass local_matrices=None for the bound computed from "
                "the operator's entries.")
        return worst


def probe_pixel_matrix(apply, decomposition, nb_components, pixel):
    """Dense ``2^d nc`` matrix of ``apply`` on the corners of ``pixel``, for
    an operator that is nonzero on that one pixel only."""
    dim = len(pixel)
    x = _real_field(decomposition, "probe-x", nb_components)
    y = _real_field(decomposition, "probe-y", nb_components)
    corners = _corner_index(dim)
    m = len(corners) * nb_components
    K = np.empty((m, m))
    for b, cb in enumerate(corners):
        for j in range(nb_components):
            x.set_zero()
            x.s[(j, 0) + tuple(p + o for p, o in zip(pixel, cb))] = 1.0
            apply(x, y)
            values = np.asarray(y.s)
            for a, ca in enumerate(corners):
                node = tuple(p + o for p, o in zip(pixel, ca))
                K[a * nb_components:(a + 1) * nb_components,
                  b * nb_components + j] = values[(slice(None), 0) + node]
    return 0.5 * (K + K.T)


# --------------------------------------------------------------------------- #
# Bounds from explicit entries (operators without local structure)
# --------------------------------------------------------------------------- #


def gershgorin_bound_from_stencil(stencil, diagonal, layout, communicator):
    """``max_i Σ_j |a_ij| / √(a_ii a_jj)``, a rigorous bound of
    ``λmax(D⁻¹A)`` (Gershgorin on ``D^-1/2 A D^-1/2``), from the explicit
    stencil. ``diagonal`` holds ``a_ii`` with ghosts exchanged; zero diagonal
    entries (rows without stiffness) are skipped."""
    nc, dim = layout.nb_components, layout.dim
    xp = _array_module(stencil)
    g = _ghosts_left(stencil)
    local = tuple(stencil.s.shape[2:])
    entries = stencil.sg[:, 0]
    diag = diagonal.sg[:, 0]

    def at(shift):
        return tuple(slice(gg + o, gg + o + n)
                     for gg, o, n in zip(g, shift, local))

    here = at((0,) * dim)
    d_here = diag[(slice(None),) + here]
    positive = d_here > 0
    inv_sqrt_here = xp.where(positive, 1.0 / xp.sqrt(xp.where(
        positive, d_here, 1.0)), 0.0)

    def inv_sqrt(shift):
        d = diag[(slice(None),) + at(shift)]
        ok = d > 0
        return xp.where(ok, 1.0 / xp.sqrt(xp.where(ok, d, 1.0)), 0.0)

    rows = xp.zeros((nc,) + local)
    for (i, j), entry in layout.centre_entry.items():
        value = xp.abs(entries[(entry,) + here])
        rows[i] += value * inv_sqrt_here[i] * inv_sqrt_here[j]
        if i != j:
            rows[j] += value * inv_sqrt_here[i] * inv_sqrt_here[j]
    for k, s in enumerate(layout.half_offsets):
        minus = tuple(-o for o in s)
        inv_plus, inv_minus = inv_sqrt(s), inv_sqrt(minus)
        for i in range(nc):
            for j in range(nc):
                forward = xp.abs(entries[(layout.block_entry(k, i, j),) + here])
                rows[i] += forward * inv_sqrt_here[i] * inv_plus[j]
                backward = xp.abs(entries[(layout.block_entry(k, j, i),)
                                          + at(minus)])
                rows[i] += backward * inv_sqrt_here[i] * inv_minus[j]
    local_max = float(rows.max()) if rows.size else 0.0
    return float(communicator.max(local_max))


# --------------------------------------------------------------------------- #
# The multigrid
# --------------------------------------------------------------------------- #


class _Level:
    """One grid of the hierarchy: decomposition, operator and work vectors.

    ``x``, ``b``: solution and right-hand side of the level problem; ``r``,
    ``d``, ``t``: residual, Chebyshev direction and scratch; ``dinv``:
    ``1/diag(A)`` per node and component. Coarse levels carry their
    ``stencil``; the fine level applies the caller's operator. All in the
    precision ``dtype`` of the cycle.
    """

    def __init__(self, index, decomposition, layout, dtype, name,
                 operator=None):
        self.index = index
        self.decomposition = decomposition
        self.layout = layout
        self.dtype = np.dtype(dtype)
        self.name = f"{name}-l{index}"
        self.operator = operator
        if operator is None:
            self.stencil = make_stencil_field(decomposition, layout,
                                              f"{self.name}-stencil", dtype)
        else:
            self.stencil = None
        self._stencil_operator = stencil_operator(layout)
        self.nb_grid_pts = tuple(int(n)
                                 for n in decomposition.nb_domain_grid_pts)
        self.x = self.b = None
        self.r = self.d = self.t = self.dinv = None
        self.bound = None
        self.bound_source = None

    def vector(self, field_name, dtype=None):
        """A nodal vector of this level (get-or-create)."""
        return _real_field(self.decomposition, f"{self.name}-{field_name}",
                           self.layout.nb_components,
                           self.dtype if dtype is None else dtype)

    def pop(self, field_name):
        """Free a field made by :meth:`vector` (or the setup stencil)."""
        collection = _collection(self.decomposition)
        name = f"{self.name}-{field_name}"
        if collection.field_exists(name):
            collection.pop_field(name)

    def allocate(self, solution=True, smoother=True):
        if solution:
            self.x = self.vector("x")
            self.b = self.vector("b")
        if smoother:
            self.r = self.vector("r")
            self.d = self.vector("d")
            self.t = self.vector("t")
            self.dinv = self.vector("dinv")

    def apply(self, x, y, stencil=None):
        """``y = A x``; exchanges the ghosts of ``x``. ``stencil`` replaces the
        level's own (the double-precision setup copy)."""
        if self.operator is not None:
            self.operator(x, y)
        else:
            self.decomposition.communicate_ghosts(x)
            self._stencil_operator.apply(
                self.stencil if stencil is None else stencil, x, y)


def stencil_diagonal(stencil, layout, out):
    """``out = diag(A)`` from the centre blocks of a stencil field."""
    for i in range(layout.nb_components):
        out.s[i] = stencil.s[layout.centre_entry[(i, i)]]


def dense_from_stencil(stencil, layout):
    """Dense matrix of a periodic stencil field on the whole (serial) grid,
    rows and columns in the order of ``field.s.ravel()``."""
    values = _to_host(stencil.s).astype(np.float64)
    nc, local = layout.nb_components, tuple(values.shape[2:])
    shape = (nc, 1) + local
    n = int(np.prod(shape))
    A = np.zeros((n, n))
    for node in np.ndindex(*local):
        rows = [np.ravel_multi_index((i, 0) + node, shape) for i in range(nc)]
        for k, s in enumerate(layout.half_offsets):
            other = tuple((c + o) % m for c, o, m in zip(node, s, local))
            cols = [np.ravel_multi_index((j, 0) + other, shape)
                    for j in range(nc)]
            block = np.array([[values[(layout.block_entry(k, i, j), 0) + node]
                               for j in range(nc)] for i in range(nc)])
            A[np.ix_(rows, cols)] += block
            A[np.ix_(cols, rows)] += block.T
        for (i, j), entry in layout.centre_entry.items():
            A[rows[i], rows[j]] += values[(entry, 0) + node]
            if i != j:
                A[rows[j], rows[i]] += values[(entry, 0) + node]
    return A


class MultigridPreconditioner(Preconditioner):
    r"""Geometric multigrid V(ν, ν)-cycle for a heterogeneous nodal operator.

    The preconditioner of muFFTTO's multigrid (Chebyshev-Jacobi smoothing,
    multilinear transfer, exact coarsest solve), built only from the action of
    the operator: the coarse operators are the Galerkin products ``R A P``
    (:func:`galerkin_by_probing`, :func:`galerkin_rap`), so the hierarchy does
    not depend on the element, the quadrature or the material model. The cycle
    is a fixed symmetric positive definite operator, as plain CG requires.

    Per level ``l`` (operator ``A_l``, ``D_l = diag(A_l)``), from ``x = 0``:

    1. ``degree`` Chebyshev steps for ``A_l x = b`` preconditioned by
       ``D_l⁻¹`` on ``[b_l/α, b_l]`` (Saad, Alg. 12.1);
    2. ``b_{l+1} = R (b - A_l x)``; the coarse correction by recursion, or on
       the coarsest grid by the pseudo-inverse (periodic, so singular);
    3. ``x += P x_{l+1}``, and ``degree`` Chebyshev steps again.

    ``b_l ≥ λmax(D_l⁻¹ A_l)`` is a guaranteed bound, never an estimate:
    Chebyshev amplifies modes above ``b_l`` (degree 2: ``|p| > 1`` beyond
    1.033 ``b_l``), so an underestimate breaks the cycle.

    * With ``local_matrices`` (:class:`ScaledLocalMatrices`, i.e.
      ``A = Σ s_k K_k`` with fixed ``K_k`` and ``s_k ≥ 0``): ``b_l = c_l``, a
      constant per level, independent of the material, computed once. The
      structure is checked against the operator at construction and
      ``s ≥ 0`` at every :meth:`setup`.
    * Otherwise, and on levels deeper than ``ScaledLocalMatrices.MAX_ATOMS``
      allows: ``b_l = max_i Σ_j |a_ij|/√(a_ii a_jj)`` (Gershgorin) from the
      explicit entries, recomputed at every setup. On the fine level the
      entries are probed (``nc ∏ k_α`` applies).

    Precision: the setup (probing, RAP, bounds, coarsest pseudo-inverse) is
    done in double precision whatever the cycle's. With
    ``precision="single"`` the stencils, work vectors and pseudo-inverse are
    stored in float32 and the cycle runs in float32 between the caller's
    float64 vectors. The fine level runs in float32 when ``operator_single``
    is given, else in double precision.

    Parameters
    ----------
    operator : callable
        ``operator(x, y)``: ``y = A x`` on float64 nodal fields of
        ``decomposition``, exchanging the ghosts of ``x`` itself (the
        ``hessp`` of :func:`~muGrid.Solvers.conjugate_gradients`).
    decomposition : muGrid.CartesianDecomposition or muGrid.FFTEngine
        The fine grid; the fields of :meth:`apply` live on its collection.
        One ghost layer per side at least.
    nb_components : int
        Unknowns per node (1, 2 or 3), one nodal point per pixel.
    local_matrices : ScaledLocalMatrices, optional
        The structure that makes the bound constant (see above).
    diagonal : callable, optional
        ``diagonal(field)`` writes ``diag(A)`` into a float64 field (e.g. the
        stiffness operator's ``assemble_diagonal``). Default: from
        ``local_matrices``, else probed.
    degree : int
        Chebyshev steps per smoothing (2: the V(2, 2) cycle).
    alpha : float
        Smoothing interval ``[b/α, b]``.
    n_coarsest : int
        Coarsen while every axis stays even and at least this wide.
    precision : {"double", "single"}
        Precision of the cycle (see above).
    operator_single : callable, optional
        ``operator(x, y)`` on float32 fields; the fine level of a single-
        precision cycle then runs in float32 as well.
    communicator : muGrid.Communicator, optional
        Communicator of the run. Default: the FFT engine's, else serial
        (a CartesianDecomposition does not keep its communicator).
    timer : muTimer.Timer, optional
        Records "setup" stages and "vcycle".
    name : str
        Prefix of the work fields.
    """

    def __init__(self, operator, decomposition, nb_components,
                 local_matrices=None, diagonal=None, degree=2, alpha=30.0,
                 n_coarsest=4, n_agglomerate=16, precision="double",
                 operator_single=None, communicator=None, timer=None,
                 name="mg"):
        if precision not in ("double", "single"):
            raise ValueError(f"precision must be 'double' or 'single', got "
                             f"{precision!r}")
        self.operator = operator
        self.decomposition = decomposition
        self.nb_components = nc = int(nb_components)
        self.local_matrices = local_matrices
        self.diagonal = diagonal
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.n_coarsest = int(n_coarsest)
        self.n_agglomerate = int(n_agglomerate)
        self.precision = precision
        self._timer = timer
        self._name = name
        self.comm = _communicator(decomposition, communicator)
        nb_grid_pts = tuple(int(n) for n in decomposition.nb_domain_grid_pts)
        self.dim = dim = len(nb_grid_pts)
        if self.n_coarsest < 3:
            raise ValueError("n_coarsest must be at least 3")
        self.layout = StencilLayout(dim, nc)
        self.transfer = GridTransfer(dim)
        dtype = np.float32 if precision == "single" else np.float64
        self.dtype = np.dtype(dtype)
        device = decomposition.device

        # The fine level of the cycle: the caller's operator in double, or in
        # single precision with its own b and x when operator_single is given.
        fine_single = precision == "single" and operator_single is not None
        self.levels = [_Level(0, decomposition, self.layout,
                              dtype if fine_single else np.float64, name,
                              operator=operator_single if fine_single
                              else operator)]

        # Distributed levels: every level on the same communicator and the
        # caller's subdivisions, nested (each rank's coarse subdomain is half
        # its fine one) while every rank keeps an even number, at least 4, of
        # nodes per axis and the grid stays at least n_agglomerate wide.
        distributed = self.comm.size > 1
        subdivisions = [int(n) for n in getattr(
            decomposition, "nb_subdivisions", [1] * dim)]
        level_pts = nb_grid_pts
        local = tuple(int(n) for n in decomposition.nb_subdomain_grid_pts)
        while _coarsenable(level_pts, self.n_coarsest):
            if distributed:
                # The first coarsening is distributed even below
                # n_agglomerate: the fine operator is only known through its
                # action, so the level gathered on rank 0 must be a stencil.
                nested = (all(n % 2 == 0 and n >= 4 for n in local) and
                          (min(level_pts) // 2 >= self.n_agglomerate
                           or len(self.levels) == 1))
                if not self.comm.all(nested):
                    break
            level_pts = tuple(n // 2 for n in level_pts)
            local = tuple(n // 2 for n in local)
            coarse = CartesianDecomposition(
                self.comm, list(level_pts), nb_subdivisions=subdivisions,
                nb_ghosts_left=[1] * dim, nb_ghosts_right=[1] * dim,
                device=device)
            self.levels.append(_Level(len(self.levels), coarse, self.layout,
                                      dtype, name))
        if len(self.levels) < 2:
            raise ValueError(
                f"{nb_grid_pts} cannot be coarsened towards n_coarsest="
                f"{self.n_coarsest}"
                + (" with every rank keeping an even number of at least 4 "
                   "nodes per axis" if distributed else "")
                + f": every axis must be even and at least "
                f"{2 * self.n_coarsest} wide")

        # Agglomeration: the last distributed level is gathered on rank 0 and
        # the rest of the cycle -- that grid and all coarser ones -- runs there
        # as a serial hierarchy (other ranks wait for its result).
        self.tail = None
        if distributed:
            self.tail = []
            if self.comm.rank == 0:
                serial = _serial_communicator()
                level_pts = self.levels[-1].nb_grid_pts
                while True:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message=".*serial muGrid Communicator.*")
                        tail_decomposition = CartesianDecomposition(
                            serial, list(level_pts),
                            nb_subdivisions=[1] * dim,
                            nb_ghosts_left=[1] * dim,
                            nb_ghosts_right=[1] * dim, device=device)
                    self.tail.append(_Level(
                        len(self.levels) - 1 + len(self.tail),
                        tail_decomposition, self.layout, dtype,
                        f"{name}-tail"))
                    if not _coarsenable(level_pts, self.n_coarsest):
                        break
                    level_pts = tuple(n // 2 for n in level_pts)
                for level in self.tail:
                    level.allocate(smoother=level is not self.tail[-1])

        for level in self.levels:
            level.allocate(solution=level.index > 0 or fine_single,
                           smoother=level is not self.levels[-1])
        #: whether the fine level has its own b and x (a float32 fine level)
        #: or works on the caller's r and z
        self._own_fine_vectors = fine_single
        # A double fine level above a single coarse one: the transfer needs
        # equal precisions, so restriction and prolongation go through a
        # double copy of the level-1 vector.
        self._bridge = (self.levels[1].vector("bridge", np.float64)
                        if self.levels[0].dtype != self.levels[1].dtype
                        else None)
        self._coarse_pinv = None
        if local_matrices is not None:
            local_matrices.check_operator(operator, decomposition, nc,
                                          self.comm)

    def _timed(self, label):
        return self._timer(label) if self._timer is not None else nullcontext()

    # -- setup -----------------------------------------------------------------

    def setup(self):
        """Everything that depends on the operator: Galerkin coarse operators,
        Jacobi diagonals, smoother bounds and the coarsest pseudo-inverse.
        Call after every change of the operator (material update).

        The coarse stencils are measured straight into the levels' own
        fields, in the cycle's precision (see :meth:`_setup_stencils`); the
        diagonal of the fine level is computed in double precision. Fields
        used only by the setup are freed at its end."""
        if self.local_matrices is not None:
            self.local_matrices.check_nonnegative(self.comm)
        # double-precision fine level for the setup: the cycle's own, or a
        # transient twin applying the double operator
        fine = self.levels[0]
        setup_fine = fine if fine.dtype == np.float64 else _Level(
            0, self.decomposition, self.layout, np.float64,
            f"{self._name}-setup", operator=self.operator)
        stencils = self._setup_stencils(self.levels[1:])
        diagonal_0 = (setup_fine.dinv if setup_fine is fine
                      else setup_fine.vector("dinv"))

        with self._timed("setup: diagonal"):
            self._fine_diagonal(setup_fine, diagonal_0)
        with self._timed("setup: galerkin"):
            # A float32 fine level probes with its float32 operator: on
            # GPUs with weak double throughput the double operator costs ~10x
            # more, and probing dominates the setup. The blocks are summed in
            # double either way (R A P is accumulated in the stencil field).
            # The probe vectors on the fine grid are the fine level's own
            # work vectors d and t (free while the setup runs).
            galerkin_by_probing(fine.operator, self.transfer,
                                self.decomposition,
                                self.levels[1].decomposition, self.layout,
                                stencils[1], name=f"{self._name}-probe",
                                dtype=fine.dtype,
                                fine_vectors=(fine.d, fine.t))
            self._rap_chain(self.levels[1:], stencils)
        with self._timed("setup: bounds"):
            fine.decomposition.communicate_ghosts(diagonal_0)
            fine.bound, fine.bound_source = self._bound(
                fine, setup_fine, None, diagonal_0)
            _invert_in_place(diagonal_0)
            if diagonal_0 is not fine.dinv:
                _copy_cast(diagonal_0, fine.dinv)
            self._smoother_setup(self.levels[1:-1], stencils, setup_fine)
        with self._timed("setup: coarsest"):
            if self.tail is None:
                self._factorise_coarsest(self.levels[-1],
                                         stencils[self.levels[-1].index])
            else:
                gathered = self._gather(stencils[self.levels[-1].index],
                                        self.levels[-1])
                if self.comm.rank == 0:
                    self._setup_tail(gathered)
        if setup_fine is not fine:
            for field_name in ("dinv", "r", "d", "t"):
                setup_fine.pop(field_name)
        self._free_probe_fields()

    def _setup_stencils(self, levels):
        """The stencils the setup measures into: the levels' own, in the
        cycle's precision. A single-precision cycle loses nothing by this: a
        float32 fine level probes with its float32 operator (the measured
        entries are float32 values), RAP accumulates in double and rounds
        each entry once, the Gershgorin bound promotes the entries to double,
        and the coarsest level is factorised in double on the host. Transient
        double copies of all coarse stencils were the largest part of the
        setup's memory (about 175 bytes per fine voxel in 3D elasticity)."""
        return {level.index: level.stencil for level in levels}

    def _rap_chain(self, levels, stencils):
        for upper, lower in zip(levels[:-1], levels[1:]):
            galerkin_rap(stencils[upper.index], upper.decomposition,
                         stencils[lower.index], lower.decomposition,
                         self.layout, name=f"{self._name}-rap")

    def _smoother_setup(self, levels, stencils, setup_fine=None, comm=None):
        """Diagonal, bound and inverse diagonal of smoothed coarse levels."""
        for level in levels:
            diagonal = (level.dinv if self.dtype == np.float64
                        else level.vector("dinv64", np.float64))
            stencil_diagonal(stencils[level.index], self.layout, diagonal)
            level.decomposition.communicate_ghosts(diagonal)
            level.bound, level.bound_source = self._bound(
                level, setup_fine, stencils[level.index], diagonal, comm)
            _invert_in_place(diagonal)
            if diagonal is not level.dinv:
                _copy_cast(diagonal, level.dinv)
                level.pop("dinv64")

    def _setup_tail(self, gathered):
        """Rank 0: the serial hierarchy below the last distributed level,
        starting from that level's gathered stencil."""
        tail = self.tail
        stencils = self._setup_stencils(tail)
        first = stencils[tail[0].index]
        xp = _array_module(first)
        first.s[...] = gathered if xp is np else xp.asarray(gathered)
        tail[0].decomposition.communicate_ghosts(first)
        self._rap_chain(tail, stencils)
        # rank 0 alone: its reductions must not involve the other ranks
        self._smoother_setup(tail[:-1], stencils,
                             comm=_serial_communicator())
        self._factorise_coarsest(tail[-1], stencils[tail[-1].index])

    def _free_probe_fields(self):
        """The fine probe vectors and the RAP block are setup-only."""
        for decomposition, names in (
                (self.decomposition, ("-probe-Pe", "-probe-APe")),
                (self.levels[1].decomposition,
                 ("-probe-e", "-probe-RAPe"))):
            # (one name per field: the probe vectors take the fine level's
            # precision, which is fixed for the multigrid)
            collection = _collection(decomposition)
            for suffix in names:
                if collection.field_exists(self._name + suffix):
                    collection.pop_field(self._name + suffix)
        for level in self.levels[1:] + (self.tail or []):
            collection = _collection(level.decomposition)
            if collection.field_exists(f"{self._name}-rap-block"):
                collection.pop_field(f"{self._name}-rap-block")

    # -- agglomeration -------------------------------------------------------------

    def _gather(self, field, level):
        """The global host array of a distributed field, on rank 0 (None
        elsewhere). Host copies only: the MPI need not be device-aware."""
        decomposition = level.decomposition
        location = tuple(int(o) for o in decomposition.subdomain_locations)
        pieces = self.comm.gather((location, _to_host(field.s)), root=0)
        if self.comm.rank != 0:
            return None
        lead = pieces[0][1].shape[:2]
        full = np.empty(lead + level.nb_grid_pts,
                        dtype=pieces[0][1].dtype)
        for start, piece in pieces:
            full[(slice(None), slice(None)) + tuple(
                slice(o, o + n) for o, n in zip(start, piece.shape[2:]))] = piece
        return full

    def _scatter(self, full, field, level):
        """Rank 0's global host array into the distributed field."""
        full = self.comm.bcast(full if self.comm.rank == 0 else None, root=0)
        decomposition = level.decomposition
        location = tuple(int(o) for o in decomposition.subdomain_locations)
        local = tuple(field.s.shape[2:])
        piece = full[(slice(None), slice(None)) + tuple(
            slice(o, o + n) for o, n in zip(location, local))]
        xp = _array_module(field)
        field.s[...] = piece if xp is np else xp.asarray(piece)

    def _fine_diagonal(self, setup_fine, out):
        if self.diagonal is not None:
            self.diagonal(out)
        elif self.local_matrices is not None:
            self._diagonal_from_local_matrices(out)
        else:
            self._probe_diagonal(setup_fine, out)

    def _diagonal_from_local_matrices(self, out):
        """``diag(A) = Σ_a Σ_k K_k[a, a] s_k(n - a)``: node ``n`` is corner
        ``a`` of pixel ``n - a``, read through the ghosts of the scales."""
        lm = self.local_matrices
        nc, dim = self.nb_components, self.dim
        self.decomposition.communicate_ghosts(lm.scales)
        with_ghosts = lm.scales.sg
        s = with_ghosts.reshape((-1,) + tuple(with_ghosts.shape[-dim:]))
        g = _ghosts_left(lm.scales)
        local = tuple(out.s.shape[2:])
        values = out.s[:, 0]
        values[...] = 0
        for a, corner in enumerate(_corner_index(dim)):
            window = tuple(slice(gg - c, gg - c + n)
                           for gg, c, n in zip(g, corner, local))
            for i in range(nc):
                for k in range(lm.nb_families):
                    values[i] += float(lm.matrices[k, a * nc + i,
                                                   a * nc + i]) * s[(k,) + window]

    def _probe_diagonal(self, level, out):
        """``diag(A)`` by Dirac combs: nodes two apart do not interact."""
        comb, response = level.vector("d"), level.vector("t")
        location = tuple(int(o) for o in
                         level.decomposition.subdomain_locations)
        for i in range(self.nb_components):
            for offset in itertools.product((0, 1), repeat=self.dim):
                # nodes of global parity `offset` (the same comb on every rank)
                nodes = (i, 0) + tuple(slice((o - loc) % 2, None, 2)
                                       for o, loc in zip(offset, location))
                comb.set_zero()
                comb.s[nodes] = 1.0
                level.apply(comb, response)
                out.s[nodes] = response.s[nodes]

    def _bound(self, level, setup_fine, stencil, diagonal, comm=None):
        """A guaranteed upper bound of λmax(D⁻¹A) on ``level``, and its
        source. On coarse levels both bounds are cheap, and the smaller of two
        guaranteed bounds is still one: the material-independent c_l is the
        sup over all s ≥ 0 and grows with depth, the Gershgorin bound of the
        actual stencil is often tighter there."""
        c = (self.local_matrices.bound(level.index)
             if self.local_matrices is not None else None)
        if level.index == 0:
            if c is not None:
                return c, "local matrices"
            return (self._probed_fine_gershgorin(setup_fine, diagonal),
                    "gershgorin (probed)")
        g = gershgorin_bound_from_stencil(
            stencil, diagonal, self.layout,
            self.comm if comm is None else comm)
        if c is not None and c <= g:
            return c, "local matrices"
        return g, "gershgorin"

    def _probed_fine_gershgorin(self, level, diag):
        """Gershgorin bound of the fine operator from probed entries, without
        storing them: per colour and component the response is one block
        column per node; its scaled absolute values add to the row sums.
        ``diag`` holds ``diag(A)`` with ghosts exchanged."""
        nc, dim = self.nb_components, self.dim
        decomposition = level.decomposition
        pts = level.nb_grid_pts
        location = tuple(int(o) for o in decomposition.subdomain_locations)
        local = tuple(diag.s.shape[2:])
        colours = [axis_colouring(n) for n in pts]
        g = _ghosts_left(diag)
        xp = _array_module(diag)

        def inv_sqrt(values):
            ok = values > 0
            return xp.where(ok, 1.0 / xp.sqrt(xp.where(ok, values, 1.0)), 0.0)

        inv_here = inv_sqrt(diag.s[:, 0])
        rows = xp.zeros((nc,) + local)
        probe, response = level.vector("d"), level.vector("t")
        weight = level.vector("r")
        local_colour = [colours[a][(location[a] + np.arange(local[a]))
                                   % pts[a]] for a in range(dim)]
        for colour in itertools.product(*(range(c.max() + 1)
                                          for c in colours)):
            mask = np.ones(local, dtype=bool)
            for a in range(dim):
                shape = [1] * dim
                shape[a] = local[a]
                mask = mask & (local_colour[a] == colour[a]).reshape(shape)
            mask = xp.asarray(mask)
            for j in range(nc):
                probe.set_zero()
                probe.s[j, 0][mask] = 1.0
                level.apply(probe, response)
                # d_j^-1/2 at the one neighbour m of every node n that has
                # colour c: the field mask * d_j^-1/2, ghosts exchanged, summed
                # over the 3^d neighbours (at most one is nonzero)
                weight.set_zero()
                weight.s[0, 0][...] = xp.where(mask, inv_sqrt(diag.s[j, 0]),
                                               0.0)
                decomposition.communicate_ghosts(weight)
                spread = xp.zeros(local)
                for t in itertools.product((-1, 0, 1), repeat=dim):
                    spread = spread + weight.sg[(0, 0) + tuple(
                        slice(gg + o, gg + o + n)
                        for gg, o, n in zip(g, t, local))]
                rows += xp.abs(response.s[:, 0]) * inv_here * spread[None]
        local_max = float(rows.max()) if rows.size else 0.0
        return float(self.comm.max(local_max))

    def _factorise_coarsest(self, level, stencil):
        """Dense pseudo-inverse of the coarsest operator (periodic: the
        constants are its null space), from its stencil, in double precision
        on the host.

        The cut-off is relative to the largest eigenvalue and follows the
        precision of the stencil: 1e-12 for a double stencil, 1e-5 for a
        float32 one (also when the fine level probed in double), whose
        rounding leaves the null space at about 1e-7 instead of 0, possibly
        negative: it must be cut, not inverted. The smallest nonzero
        eigenvalue of a 4^d grid is O(0.1) of the largest."""
        A = dense_from_stencil(stencil, self.layout)
        A = 0.5 * (A + A.T)
        rcond = 1e-12 if np.dtype(stencil.dtype) == np.float64 else 1e-5
        pinv = np.linalg.pinv(A, rcond=rcond, hermitian=True)
        xp = _array_module(level.x)
        self._coarse_pinv = xp.asarray(pinv.astype(level.dtype))

    # -- the cycle ---------------------------------------------------------------

    def apply(self, r, z):
        """``z = M r``: one V-cycle from ``z = 0``."""
        fine = self.levels[0]
        with self._timed("vcycle"):
            if self._own_fine_vectors:           # its own (float32) b and x
                _copy_cast(r, fine.b)
                self._cycle(self.levels, 0)
                _copy_cast(fine.x, z)
            else:                                # the cycle works on r and z
                fine.b, fine.x = r, z
                self._cycle(self.levels, 0)

    def _solve_coarsest(self, level):
        values = self._coarse_pinv @ level.b.s.reshape(-1)
        level.x.s[...] = values.reshape(level.x.s.shape)

    def _agglomerate(self, level):
        """The last distributed level: gather b on rank 0, run the serial
        tail of the cycle there, scatter x."""
        full = self._gather(level.b, level)
        solution = None
        if self.comm.rank == 0:
            first = self.tail[0]
            xp = _array_module(first.b)
            first.b.s[...] = full if xp is np else xp.asarray(full)
            self._cycle(self.tail, 0)
            solution = _to_host(first.x.s)
        self._scatter(solution, level.x, level)

    def _cycle(self, levels, index):
        level = levels[index]
        if index == len(levels) - 1:
            if levels is self.levels and self.tail is not None:
                self._agglomerate(level)
            else:
                self._solve_coarsest(level)
            return
        coarse = levels[index + 1]
        bridge = self._bridge if (levels is self.levels and index == 0) \
            else None
        self._smooth(level, x_is_zero=True)
        level.apply(level.x, level.t)                   # r = b - A x
        linalg.copy(level.b, level.r)
        linalg.axpy(-1.0, level.t, level.r)
        level.decomposition.communicate_ghosts(level.r)
        if bridge is None:
            self.transfer.restrict(level.r, coarse.b)
        else:
            self.transfer.restrict(level.r, bridge)
            _copy_cast(bridge, coarse.b)
        self._cycle(levels, index + 1)
        if bridge is None:
            coarse.decomposition.communicate_ghosts(coarse.x)
            self.transfer.prolong(coarse.x, level.t)    # x += P x_H
        else:
            _copy_cast(coarse.x, bridge)
            coarse.decomposition.communicate_ghosts(bridge)
            self.transfer.prolong(bridge, level.t)
        linalg.axpy(1.0, level.t, level.x)
        self._smooth(level, x_is_zero=False)

    def _smooth(self, level, x_is_zero):
        """``degree`` Chebyshev steps for ``A x = b`` preconditioned by
        ``D⁻¹`` on ``[b/α, b]`` (Saad, Alg. 12.1)."""
        lmax = level.bound
        lmin = lmax / self.alpha
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        x, b, r, d, t = level.x, level.b, level.r, level.d, level.t
        linalg.copy(b, r)                               # r = b - A x
        if not x_is_zero:
            level.apply(x, t)
            linalg.axpy(-1.0, t, r)
        self._jacobi(level, r, d)                       # d = D⁻¹ r / θ
        linalg.scal(1.0 / theta, d)
        for k in range(self.degree):
            if k == 0 and x_is_zero:
                linalg.copy(d, x)
            else:
                linalg.axpy(1.0, d, x)                  # x += d
            if k == self.degree - 1:
                break
            level.apply(d, t)                           # r -= A d
            linalg.axpy(-1.0, t, r)
            rho_new = 1.0 / (2.0 * sigma - rho)
            self._jacobi(level, r, t)                   # d = ρ'ρ d + 2ρ'/δ D⁻¹r
            linalg.axpby(2.0 * rho_new / delta, t, rho_new * rho, d)
            rho = rho_new

    @staticmethod
    def _jacobi(level, r, z):
        """``z = D⁻¹ r``, exact on the interior.

        On a GPU the product runs over the whole contiguous buffers (see
        :func:`_contiguous_view`): an elementwise kernel on the strided
        ghost-free views ``.s`` computes a multi-dimensional index per
        element and took a sixth of a V-cycle at 128³. The ghost entries of
        ``z`` then hold ``dinv · r`` of the ghosts; they are finite (``dinv``
        is inverted on the whole buffer) and every reader of ``z`` exchanges
        its ghosts first."""
        xp = _array_module(r)
        dim = len(level.nb_grid_pts)
        views = [_contiguous_view(field, dim) for field in (level.dinv, r, z)]
        if all(view is not None for view in views):
            xp.multiply(views[0], views[1], out=views[2])
        else:
            xp.multiply(level.dinv.s, r.s, out=z.s)

    # -- diagnostics -------------------------------------------------------------

    def level_summary(self):
        """``[(grid, bound, source)]`` per smoothed level."""
        levels = self.levels[:-1] + (self.tail[:-1] if self.tail else [])
        return [(level.nb_grid_pts, level.bound, level.bound_source)
                for level in levels]


def _coarsenable(nb_grid_pts, n_coarsest):
    """Whether a grid can be halved with every axis staying >= n_coarsest."""
    return all(n % 2 == 0 and n // 2 >= n_coarsest for n in nb_grid_pts)


def _serial_communicator():
    """A one-rank communicator (rank 0's serial tail under MPI), without the
    warning muGrid gives for a serial communicator under an MPI launcher."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=".*serial muGrid Communicator.*")
        return Communicator()


def _invert_in_place(field):
    """``field = 1 / field`` on the whole buffer; zero entries (nodes without
    stiffness) become 1, which leaves those nodes alone."""
    values = field.sg
    xp = _array_module(field)
    values[values == 0] = 1.0
    xp.divide(1.0, values, out=values)


def _contiguous_view(field, dim):
    """The whole buffer of a device field (ghosts included) as a C-contiguous
    array, or None if there is no such view (host fields keep the components
    of a node together).

    On a device muGrid stores a field component by component with the grid
    points innermost, x fastest: reversing the spatial axes of ``field.sg``
    gives a C-contiguous array ``[components..., sub-points, z, y, x]``.
    Elementwise kernels on it run at memory bandwidth; use it for pointwise
    operations whose result on the ghosts does not matter."""
    values = field.sg
    if _array_module(field) is np:
        return None
    lead = values.ndim - dim
    view = values.transpose(*range(lead), *reversed(range(lead, values.ndim)))
    return view if view.flags.c_contiguous else None


def _copy_cast(source, target):
    """``target = source`` for fields of possibly different precision (whole
    buffers, ghosts included). On a device the cast runs over the contiguous
    buffers (:func:`_contiguous_view`): on the component-first strided view
    ``.sg`` the elementwise kernel computes an index per element, which made
    the two casts per V-cycle 7 ms of 43 at 256³ on an L40S."""
    if np.dtype(source.dtype) == np.dtype(target.dtype):
        linalg.copy(source, target)
        return
    dim = len(_ghosts_left(source))
    source_view = _contiguous_view(source, dim)
    target_view = _contiguous_view(target, dim)
    if source_view is not None and target_view is not None:
        target_view[...] = source_view
    else:
        target.sg[...] = source.sg
