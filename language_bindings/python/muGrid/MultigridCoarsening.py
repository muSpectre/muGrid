"""
Coarsening rules for :class:`muGrid.Multigrid.MultigridPreconditioner`.

By default the multigrid's coarse operators are Galerkin products ``R A P``,
measured from the fine operator's action and stored as stencils: no element,
quadrature or material model is needed. A *coarsening rule* replaces them by
re-discretised operators: the same kind of operator as the fine one, on the
coarse grid, with the material averaged onto it. That is the multigrid of
muFFTTO's large-grid topology optimisation. It costs no probing and no RAP at a
material update, applies a fused kernel instead of a 27-point stencil on every
coarse level, and keeps only the averaged material per coarse voxel.

A rule is the only place that knows the element. The multigrid asks it, per
coarse level, for an object with

* ``apply(x, y)``: ``y = A_l x`` (exchanging the ghosts of ``x`` itself);
* ``coarsen_from(finer)``: put the material of the next finer level
  (``None`` for the fine one) onto this level, at every setup;
* ``local_matrices``: a :class:`~muGrid.Multigrid.ScaledLocalMatrices` of
  this level's operator, ``A_l = Σ_cells Σ_k s_k K_k``, from which the
  multigrid takes the diagonal, the guaranteed smoother bound and the
  coarsest dense matrix;
* ``material_fields()``: the fields holding the material (gathered on rank 0
  for the serial tail of a distributed hierarchy), and
  ``set_material(values)`` to fill them from such gathered arrays.

The bound stays guaranteed: a re-discretised level is again a sum of scaled
copies of one element matrix with non-negative factors, so the mediant
inequality of :class:`~muGrid.Multigrid.ScaledLocalMatrices` applies with
that element matrix.
"""

import numpy as np

from . import _muGrid
from .Multigrid import ScaledLocalMatrices, _array_module, _to_host
from .Wrappers import IsotropicStiffnessOperator


def _mean_of_children(finer, coarse, dim):
    """``coarse[c] = mean over o ∈ {0,1}^d of finer[2c + o]`` for the
    interiors of two pixel fields of nested grids (each rank's coarse
    subdomain is half its fine one, so all children are local). Summed in
    double in a fixed order, then stored in the coarse field's precision."""
    xp = _array_module(coarse)
    fine_values = finer.s.reshape((-1,) + tuple(finer.s.shape[-dim:]))
    coarse_values = coarse.s.reshape((-1,) + tuple(coarse.s.shape[-dim:]))
    total = xp.zeros(coarse_values.shape, dtype=np.float64)
    for offset in np.ndindex(*(2,) * dim):
        total += fine_values[(slice(None),) + tuple(
            slice(o, None, 2) for o in offset)]
    total /= 2 ** dim
    coarse_values[...] = total


class IsotropicRediscretisation:
    r"""Re-discretised coarse levels for the fused isotropic stiffness of
    trilinear (3D) or bilinear (2D) elements with one material value per
    voxel, :class:`muGrid.IsotropicStiffnessOperator` with fields
    ``λ(x), μ(x)``.

    Level ``l`` applies the same operator at the grid spacing ``2^l h``, with
    ``λ_l, μ_l`` the means of the ``2^d`` child voxels of level ``l - 1``
    (arithmetic means, as muFFTTO's multigrid). For a homogeneous material the
    re-discretised and the Galerkin operator coincide (multilinear
    interpolation is exact for Q1); for a heterogeneous one the coarse levels
    are softer than Galerkin's, which may cost CG iterations.

    Parameters
    ----------
    element : muGrid.FEMElement
        The element of the fine operator. Only ``FEMElement.q1``: for P1 the
        multilinear transfer is not nested, so re-discretised P1 levels are
        not Galerkin even for a homogeneous material.
    grid_spacing : sequence of float
        The fine grid spacing per axis, as given to the fine operator.
    lame : tuple of muGrid.Field
        ``(λ, μ)``: the fine operator's material fields, one value per pixel,
        on the multigrid's fine decomposition, with their ghosts exchanged.
        Read at every setup.
    lambda_over_mu : float, optional
        ``λ / μ`` if it is the same everywhere (a scaled material
        ``s(x) C₀``): the levels then keep ``μ_l`` as the one family of
        their local matrices, which gives the tight bound. Default: two
        families ``(λ, μ)``, a bound valid for any ``λ, μ ≥ 0``.
    """

    def __init__(self, element, grid_spacing, lame, lambda_over_mu=None):
        if element != _muGrid.FEMElement.q1:
            raise ValueError(
                "re-discretised coarse levels need trilinear/bilinear (q1) "
                "elements: for P1 the multilinear transfer is not nested")
        self.element = element
        self.grid_spacing = [float(h) for h in grid_spacing]
        self.dim = len(self.grid_spacing)
        self.lame = tuple(lame)
        if len(self.lame) != 2:
            raise ValueError("lame must be the pair (lambda, mu) of fields")
        self.lambda_over_mu = (None if lambda_over_mu is None
                               else float(lambda_over_mu))
        if self.lambda_over_mu is not None and self.lambda_over_mu < 0:
            raise ValueError("lambda_over_mu must be >= 0")

    def operator(self, index):
        """The fused operator of level ``index`` (spacing ``2^index h``)."""
        return IsotropicStiffnessOperator(
            self.dim, [2 ** index * h for h in self.grid_spacing],
            self.element)

    def level(self, index, decomposition, dtype, name):
        """The coarse operator of level ``index`` on ``decomposition``."""
        return RediscretisedLevel(self, index, decomposition, dtype, name)

    def check(self, operator, decomposition, nb_components, communicator,
              rtol=1e-8, seed=0):
        """Raise unless the rule's level-0 operator (the fine material at
        spacing ``h``) equals ``operator`` on a random vector: catches a wrong
        grid spacing, element or material field before the coarse levels are
        built from them. Two fine vectors, freed afterwards."""
        from .Multigrid import _collection, _real_field

        names = ("mg-rule-check-x", "mg-rule-check-y", "mg-rule-check-z")
        collection = _collection(decomposition)
        try:
            x, y, z = (_real_field(decomposition, n, nb_components)
                       for n in names)
            rng = np.random.default_rng(seed + communicator.rank)
            values = rng.standard_normal(x.s.shape)
            xp = _array_module(x)
            x.s[...] = values if xp is np else xp.asarray(values)
            operator(x, y)
            decomposition.communicate_ghosts(x)
            self.operator(0).apply(x, *self.lame, z)
            got, expected = _to_host(z.s), _to_host(y.s)
            scale = max(float(np.abs(expected).max()), 1e-300)
            local = float(np.abs(got - expected).max()) / scale
        finally:
            for n in names:
                if collection.field_exists(n):
                    collection.pop_field(n)
        worst = float(communicator.max(local))
        if worst > rtol:
            raise ValueError(
                "the coarsening rule does not reproduce the fine operator "
                f"(relative mismatch {worst:.2e}): check grid_spacing, the "
                "element and the (lambda, mu) fields")


class RediscretisedLevel:
    """One coarse level of :class:`IsotropicRediscretisation`: the fused
    operator at this level's spacing and its averaged material ``λ_l, μ_l``
    (fields of the level's decomposition, in the cycle's precision)."""

    def __init__(self, rule, index, decomposition, dtype, name):
        self.rule = rule
        self.index = index
        self.decomposition = decomposition
        self.dtype = np.dtype(dtype)
        self._operator = rule.operator(index)
        # one component, so that the multigrid's gather sees (1, 1, *grid)
        self.lam = decomposition.real_field(f"{name}-lambda", (1,),
                                            dtype=dtype)
        self.mu = decomposition.real_field(f"{name}-mu", (1,), dtype=dtype)
        ratio = rule.lambda_over_mu
        if ratio is not None:
            # λ = ratio μ everywhere: one family scaled by μ
            self.local_matrices = ScaledLocalMatrices.isotropic_stiffness(
                self._operator, ratio, 1.0, self.mu)
        else:
            # λ and μ independent: two families, the scales in one field
            self._scales = decomposition.real_field(f"{name}-scales", (2,),
                                                    dtype=dtype)
            k_lambda = ScaledLocalMatrices.isotropic_stiffness(
                self._operator, 1.0, 0.0, self.mu).matrices[0]
            k_mu = ScaledLocalMatrices.isotropic_stiffness(
                self._operator, 0.0, 1.0, self.mu).matrices[0]
            self.local_matrices = ScaledLocalMatrices(
                np.array([k_lambda, k_mu]), self._scales)

    def apply(self, x, y):
        """``y = A_l x``; exchanges the ghosts of ``x``."""
        self.decomposition.communicate_ghosts(x)
        self._operator.apply(x, self.lam, self.mu, y)

    def material_fields(self):
        return self.lam, self.mu

    def coarsen_from(self, finer):
        """Means of the child voxels of ``finer`` (a RediscretisedLevel, or
        None for the fine material of the rule)."""
        sources = self.rule.lame if finer is None else finer.material_fields()
        for source, target in zip(sources, (self.lam, self.mu)):
            _mean_of_children(source, target, self.rule.dim)
        self._finish()

    def set_material(self, lam, mu):
        """Fill the material from host arrays of this level's subdomain,
        shaped like the fields' interior (the serial tail under MPI)."""
        for values, target in ((lam, self.lam), (mu, self.mu)):
            xp = _array_module(target)
            target.s[...] = values if xp is np else xp.asarray(values)
        self._finish()

    def _finish(self):
        for field in (self.lam, self.mu):
            self.decomposition.communicate_ghosts(field)
        if self.rule.lambda_over_mu is None:
            self._scales.s[0] = self.lam.s[0]
            self._scales.s[1] = self.mu.s[0]
            self.decomposition.communicate_ghosts(self._scales)
