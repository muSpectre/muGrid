#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_laminate_homogenization_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    02 Jan 2026

@brief   Tests for isotropic stiffness kernel using laminate homogenization.

This test verifies that the homogenized elastic properties of laminate
microstructures exactly match the analytical Voigt and Reuss bounds.

For a two-phase laminate:
- Voigt bound (uniform strain): applies when loading perpendicular to layers
- Reuss bound (uniform stress): applies when loading parallel to layers

Copyright 2026 Lars Pastewka

uGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

uGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with uGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np
import pytest

import muGrid
from muGrid import Timer
from muGrid.Solvers import conjugate_gradients

# =============================================================================
# Helper functions
# =============================================================================


def voigt_index_2d(i, j):
    """Convert 2D tensor indices to Voigt notation index."""
    if i == j:
        return i
    return 2


def voigt_index_3d(i, j):
    """Convert 3D tensor indices to Voigt notation index.

    Voigt ordering: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
    """
    if i == j:
        return i
    pair = tuple(sorted([i, j]))
    mapping = {(1, 2): 3, (0, 2): 4, (0, 1): 5}
    return mapping[pair]


def voigt_index(dim, i, j):
    """Convert tensor indices to Voigt notation index for given dimension."""
    if dim == 2:
        return voigt_index_2d(i, j)
    else:
        return voigt_index_3d(i, j)


def isotropic_stiffness_2d(E, nu):
    """
    Create 2D plane strain isotropic stiffness tensor in Voigt notation.
    Returns C[3, 3] where [xx, yy, xy] ordering is used.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = np.zeros((3, 3))
    C[0, 0] = lam + 2 * mu  # C_xxxx
    C[1, 1] = lam + 2 * mu  # C_yyyy
    C[2, 2] = mu  # C_xyxy
    C[0, 1] = C[1, 0] = lam  # C_xxyy
    return C


def isotropic_stiffness_3d(E, nu):
    """
    Create 3D isotropic stiffness tensor in Voigt notation.
    Returns C[6, 6] where [xx, yy, zz, yz, xz, xy] ordering is used.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = np.zeros((6, 6))
    C[0, 0] = lam + 2 * mu
    C[1, 1] = lam + 2 * mu
    C[2, 2] = lam + 2 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam
    return C


def compute_voigt_average(C1, C2, f1):
    """Compute Voigt (arithmetic) average of stiffness tensors."""
    f2 = 1.0 - f1
    return f1 * C1 + f2 * C2


def compute_reuss_average(C1, C2, f1):
    """Compute Reuss (harmonic) average of stiffness tensors."""
    f2 = 1.0 - f1
    S1 = np.linalg.inv(C1)
    S2 = np.linalg.inv(C2)
    S_eff = f1 * S1 + f2 * S2
    return np.linalg.inv(S_eff)


def laminate_stiffness_2d(E1, E2, nu, f1, layer_dir):
    """
    Compute exact laminate stiffness tensor for 2D plane strain.

    For a laminate with layers normal to `layer_dir`, the effective stiffness
    is NOT simply Voigt or Reuss average due to Poisson coupling.

    Parameters
    ----------
    E1, E2 : float
        Young's moduli of the two phases.
    nu : float
        Poisson's ratio (same for both phases).
    f1 : float
        Volume fraction of phase 1.
    layer_dir : int
        Direction normal to layers (0=x, 1=y).

    Returns
    -------
    C_eff : ndarray
        Effective stiffness tensor in Voigt notation (3x3).
    """
    f2 = 1.0 - f1

    # Lame parameters
    lam1 = E1 * nu / ((1 + nu) * (1 - 2 * nu))
    mu1 = E1 / (2 * (1 + nu))
    lam2 = E2 * nu / ((1 + nu) * (1 - 2 * nu))
    mu2 = E2 / (2 * (1 + nu))

    # Plane strain modulus M = lambda + 2*mu
    M1 = lam1 + 2 * mu1
    M2 = lam2 + 2 * mu2

    # Effective stiffness components
    # For layer normal in direction d:
    # - C_dd follows Reuss (stress continuity perpendicular to layers)
    # - C_tt follows modified Voigt with Poisson correction (parallel to layers)
    # - C_dt has off-diagonal coupling

    # Reuss for perpendicular direction (d)
    C_dd = 1.0 / (f1 / M1 + f2 / M2)

    # Modified formula for parallel direction (t)
    # C_tt = Voigt - f1*f2*(lambda1 - lambda2)^2 / (M1*f2 + M2*f1)
    M_voigt = f1 * M1 + f2 * M2
    correction = f1 * f2 * (lam1 - lam2) ** 2 / (M1 * f2 + M2 * f1)
    C_tt = M_voigt - correction

    # Off-diagonal coupling C_dt = C_td
    # Analytical: C_dt = weighted average of lambda with correction
    lam_voigt = f1 * lam1 + f2 * lam2
    lam_correction = f1 * f2 * (lam1 - lam2) * (M1 - M2) / (M1 * f2 + M2 * f1)
    C_dt = lam_voigt - lam_correction

    # Shear modulus: Reuss average (stress continuity for shear)
    mu_reuss = 1.0 / (f1 / mu1 + f2 / mu2)

    # Build stiffness tensor
    C_eff = np.zeros((3, 3))
    t_dir = 1 - layer_dir  # tangent direction

    C_eff[layer_dir, layer_dir] = C_dd
    C_eff[t_dir, t_dir] = C_tt
    C_eff[layer_dir, t_dir] = C_dt
    C_eff[t_dir, layer_dir] = C_dt
    C_eff[2, 2] = mu_reuss

    return C_eff


def laminate_stiffness_3d(E1, E2, nu, f1, layer_dir):
    """
    Compute exact laminate stiffness tensor for 3D.

    For a laminate with layers normal to `layer_dir`, the effective stiffness
    accounts for stress/strain continuity constraints.

    Parameters
    ----------
    E1, E2 : float
        Young's moduli of the two phases.
    nu : float
        Poisson's ratio (same for both phases).
    f1 : float
        Volume fraction of phase 1.
    layer_dir : int
        Direction normal to layers (0=x, 1=y, 2=z).

    Returns
    -------
    C_eff : ndarray
        Effective stiffness tensor in Voigt notation (6x6).
    """
    f2 = 1.0 - f1

    # Lame parameters
    lam1 = E1 * nu / ((1 + nu) * (1 - 2 * nu))
    mu1 = E1 / (2 * (1 + nu))
    lam2 = E2 * nu / ((1 + nu) * (1 - 2 * nu))
    mu2 = E2 / (2 * (1 + nu))

    # Plane strain modulus M = lambda + 2*mu
    M1 = lam1 + 2 * mu1
    M2 = lam2 + 2 * mu2

    # For layer normal in direction d and tangent directions t1, t2:
    # - C_dd: Reuss (stress continuity perpendicular to layers)
    # - C_t1t1, C_t2t2: Modified with Poisson correction
    # - Coupling terms adjusted

    # Reuss for perpendicular direction
    C_dd = 1.0 / (f1 / M1 + f2 / M2)

    # Modified for parallel directions
    M_voigt = f1 * M1 + f2 * M2
    correction = f1 * f2 * (lam1 - lam2) ** 2 / (M1 * f2 + M2 * f1)
    C_tt = M_voigt - correction

    # Coupling: lambda between d and t
    lam_voigt = f1 * lam1 + f2 * lam2
    lam_correction = f1 * f2 * (lam1 - lam2) * (M1 - M2) / (M1 * f2 + M2 * f1)
    C_dt = lam_voigt - lam_correction

    # Coupling: lambda between t1 and t2 (both parallel to layers)
    # This also has Poisson correction due to equilibrium constraint in layer normal
    # The correction is the same as for C_tt
    C_t1t2 = lam_voigt - correction

    # Shear moduli
    # Shear in plane containing d: Reuss
    mu_reuss = 1.0 / (f1 / mu1 + f2 / mu2)
    # Shear in plane parallel to layers: Voigt
    mu_voigt = f1 * mu1 + f2 * mu2

    # Build stiffness tensor in Voigt notation
    # Ordering: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
    C_eff = np.zeros((6, 6))

    # Get tangent directions
    tangent_dirs = [i for i in range(3) if i != layer_dir]
    t1, t2 = tangent_dirs

    # Normal components
    C_eff[layer_dir, layer_dir] = C_dd
    C_eff[t1, t1] = C_tt
    C_eff[t2, t2] = C_tt

    # Coupling between d and tangent
    C_eff[layer_dir, t1] = C_dt
    C_eff[t1, layer_dir] = C_dt
    C_eff[layer_dir, t2] = C_dt
    C_eff[t2, layer_dir] = C_dt

    # Coupling between tangent directions
    C_eff[t1, t2] = C_t1t2
    C_eff[t2, t1] = C_t1t2

    # Shear components (Voigt indices 3, 4, 5 correspond to yz, xz, xy)
    # Shear in plane containing layer_dir uses Reuss
    # Shear in plane parallel to layers uses Voigt
    for shear_idx in [3, 4, 5]:
        # Determine which plane this shear is in
        # 3=yz (plane perpendicular to x), 4=xz (perp to y), 5=xy (perp to z)
        perp_to = {3: 0, 4: 1, 5: 2}[shear_idx]
        if perp_to == layer_dir:
            # Shear parallel to layers
            C_eff[shear_idx, shear_idx] = mu_voigt
        else:
            # Shear involving layer normal direction
            C_eff[shear_idx, shear_idx] = mu_reuss

    return C_eff


# =============================================================================
# Homogenization solver
# =============================================================================


class LaminateHomogenization:
    """
    Compute homogenized stiffness of a laminate microstructure.

    The laminate consists of alternating layers of two isotropic materials.
    """

    def __init__(
        self,
        nb_grid_pts,
        E1,
        E2,
        nu,
        layer_direction,
        volume_fraction_1=0.5,
        tol=1e-6,
        maxiter=1000,
    ):
        """
        Initialize laminate homogenization.

        Parameters
        ----------
        nb_grid_pts : tuple
            Number of grid points in each direction.
        E1 : float
            Young's modulus of phase 1.
        E2 : float
            Young's modulus of phase 2.
        nu : float
            Poisson's ratio (same for both phases).
        layer_direction : int
            Direction normal to layers (0=x, 1=y, 2=z).
        volume_fraction_1 : float
            Volume fraction of phase 1 (default 0.5).
        tol : float
            CG tolerance.
        maxiter : int
            Maximum CG iterations.
        """
        self.nb_grid_pts = nb_grid_pts
        self.dim = len(nb_grid_pts)
        self.E1 = E1
        self.E2 = E2
        self.nu = nu
        self.layer_direction = layer_direction
        self.volume_fraction_1 = volume_fraction_1
        self.tol = tol
        self.maxiter = maxiter

        # Communicator
        try:
            from mpi4py import MPI

            self.comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            self.comm = muGrid.Communicator()

        # Compute Lame parameters
        self.lam1 = E1 * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu1 = E1 / (2 * (1 + nu))
        self.lam2 = E2 * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu2 = E2 / (2 * (1 + nu))

        # Domain size (unit cube)
        self.domain_size = tuple(1.0 for _ in range(self.dim))
        self.grid_spacing = [L / n for L, n in zip(self.domain_size, nb_grid_pts)]

        # Create gradient operator
        self.gradient_op = muGrid.FEMGradientOperator(self.dim, list(self.grid_spacing))
        self.nb_quad = self.gradient_op.nb_quad_pts
        self.quad_weights = self.gradient_op.quadrature_weights

        # Create decomposition with ghost cells
        self.decomposition = muGrid.CartesianDecomposition(
            self.comm,
            nb_grid_pts,
            nb_subdivisions=(1,) * self.dim,
            nb_ghosts_left=(1,) * self.dim,
            nb_ghosts_right=(1,) * self.dim,
            nb_sub_pts={"quad": self.nb_quad},
        )

        # Create element decomposition for material fields
        self.element_decomposition = muGrid.CartesianDecomposition(
            self.comm,
            nb_grid_pts,
            nb_subdivisions=(1,) * self.dim,
            nb_ghosts_left=(1,) * self.dim,
            nb_ghosts_right=(1,) * self.dim,
        )

        # Create stiffness operator
        if self.dim == 2:
            self.stiffness_op = muGrid.IsotropicStiffnessOperator2D(
                list(self.grid_spacing)
            )
        else:
            self.stiffness_op = muGrid.IsotropicStiffnessOperator3D(
                list(self.grid_spacing)
            )

        # Create fields
        self._create_fields()
        self._setup_material()

    def _create_fields(self):
        """Create required fields."""
        # Displacement and force fields
        self.u_field = self.decomposition.real_field("displacement", (self.dim,))
        self.rhs_field = self.decomposition.real_field("rhs", (self.dim,))
        self.force_field = self.decomposition.real_field("force", (self.dim,))

        # Material fields
        self.lambda_field = self.element_decomposition.real_field("lambda")
        self.mu_field = self.element_decomposition.real_field("mu")

        # Gradient and stress fields for post-processing
        self.gradient_field = self.decomposition.real_field(
            "gradient", (self.dim, self.dim), "quad"
        )
        self.stress_field = self.decomposition.real_field(
            "stress", (self.dim, self.dim), "quad"
        )

        # Stiffness tensor field for generic operator
        self.nb_voigt = 3 if self.dim == 2 else 6
        self.C_field = np.zeros(
            (self.nb_voigt, self.nb_voigt, self.nb_quad) + tuple(self.nb_grid_pts)
        )

    def _setup_material(self):
        """Set up laminate material distribution."""
        # Create coordinate grid for nodes
        coords = [
            np.linspace(0, L, n, endpoint=False)
            for L, n in zip(self.domain_size, self.nb_grid_pts)
        ]
        grid = np.meshgrid(*coords, indexing="ij")

        # Determine phase based on layer direction
        # Phase 1 where coordinate < volume_fraction_1
        layer_coord = grid[self.layer_direction]
        phase = (layer_coord < self.volume_fraction_1).astype(float)

        # Set material properties
        self.lambda_field.p[...] = self.lam1 * phase + self.lam2 * (1 - phase)
        self.mu_field.p[...] = self.mu1 * phase + self.mu2 * (1 - phase)

        # Fill ghost cells
        self.element_decomposition.communicate_ghosts(self.lambda_field)
        self.element_decomposition.communicate_ghosts(self.mu_field)

        # Also set up stiffness tensor for post-processing
        C1 = (
            isotropic_stiffness_2d(self.E1, self.nu)
            if self.dim == 2
            else isotropic_stiffness_3d(self.E1, self.nu)
        )
        C2 = (
            isotropic_stiffness_2d(self.E2, self.nu)
            if self.dim == 2
            else isotropic_stiffness_3d(self.E2, self.nu)
        )

        for q in range(self.nb_quad):
            for i in range(self.nb_voigt):
                for j in range(self.nb_voigt):
                    self.C_field[i, j, q, ...] = C1[i, j] * phase + C2[i, j] * (
                        1 - phase
                    )

    def _apply_stiffness(self, u_in, f_out):
        """Apply stiffness operator K = B^T C B."""
        self.decomposition.communicate_ghosts(u_in)
        self.stiffness_op.apply(u_in, self.lambda_field, self.mu_field, f_out)

    def _strain_to_voigt(self, strain):
        """Convert strain tensor to Voigt notation."""
        if self.dim == 2:
            eps_voigt = np.zeros((3, self.nb_quad) + tuple(self.nb_grid_pts))
            eps_voigt[0, ...] = strain[0, 0, ...]
            eps_voigt[1, ...] = strain[1, 1, ...]
            eps_voigt[2, ...] = 2 * strain[0, 1, ...]
        else:
            eps_voigt = np.zeros((6, self.nb_quad) + tuple(self.nb_grid_pts))
            eps_voigt[0, ...] = strain[0, 0, ...]
            eps_voigt[1, ...] = strain[1, 1, ...]
            eps_voigt[2, ...] = strain[2, 2, ...]
            eps_voigt[3, ...] = 2 * strain[1, 2, ...]
            eps_voigt[4, ...] = 2 * strain[0, 2, ...]
            eps_voigt[5, ...] = 2 * strain[0, 1, ...]
        return eps_voigt

    def _voigt_to_stress(self, sig_voigt, stress):
        """Convert stress from Voigt notation to tensor."""
        if self.dim == 2:
            stress[0, 0, ...] = sig_voigt[0, ...]
            stress[1, 1, ...] = sig_voigt[1, ...]
            stress[0, 1, ...] = sig_voigt[2, ...]
            stress[1, 0, ...] = sig_voigt[2, ...]
        else:
            stress[0, 0, ...] = sig_voigt[0, ...]
            stress[1, 1, ...] = sig_voigt[1, ...]
            stress[2, 2, ...] = sig_voigt[2, ...]
            stress[1, 2, ...] = sig_voigt[3, ...]
            stress[2, 1, ...] = sig_voigt[3, ...]
            stress[0, 2, ...] = sig_voigt[4, ...]
            stress[2, 0, ...] = sig_voigt[4, ...]
            stress[0, 1, ...] = sig_voigt[5, ...]
            stress[1, 0, ...] = sig_voigt[5, ...]

    def _compute_stress(self, strain, stress):
        """Compute stress from strain using Voigt notation."""
        eps_voigt = self._strain_to_voigt(strain)
        sig_voigt = np.einsum("ijq...,jq...->iq...", self.C_field, eps_voigt)
        self._voigt_to_stress(sig_voigt, stress)

    def _compute_divergence(self, stress, f_vec):
        """Compute divergence of stress tensor."""
        self.stress_field.s[...] = stress
        self.decomposition.communicate_ghosts(self.stress_field)
        f_vec.pg[...] = 0.0
        self.gradient_op.transpose(self.stress_field, f_vec, list(self.quad_weights))

    def _compute_rhs(self, E_macro, rhs_out):
        """Compute RHS: f = -B^T C E_macro."""
        strain_shape = (self.dim, self.dim, self.nb_quad) + tuple(self.nb_grid_pts)
        eps_macro = np.zeros(strain_shape)
        for i in range(self.dim):
            for j in range(self.dim):
                eps_macro[i, j, ...] = E_macro[i, j]

        sig_macro = np.zeros_like(eps_macro)
        self._compute_stress(eps_macro, sig_macro)
        self._compute_divergence(sig_macro, rhs_out)
        rhs_out.s[...] *= -1.0

    def compute_homogenized_stiffness(self):
        """
        Compute the full homogenized stiffness tensor.

        Returns
        -------
        C_eff : ndarray
            Homogenized stiffness tensor in Voigt notation.
        """
        timer = Timer()
        C_eff = np.zeros((self.nb_voigt, self.nb_voigt))

        # Strain cases
        if self.dim == 2:
            strain_cases = [(0, 0), (1, 1), (0, 1)]
        else:
            strain_cases = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for case_idx, (i, j) in enumerate(strain_cases):
            E_macro = np.zeros((self.dim, self.dim))
            if i == j:
                # Normal strain: tensor strain = 1.0
                E_macro[i, j] = 1.0
            else:
                # Shear strain: tensor strain = 0.5 so engineering shear = 1.0
                # In Voigt notation, C relates engineering shear (γ = 2ε) to stress
                E_macro[i, j] = 0.5
                E_macro[j, i] = 0.5

            voigt_col = voigt_index(self.dim, i, j)

            # Compute RHS
            self._compute_rhs(E_macro, self.rhs_field)

            # Initialize displacement
            self.u_field.s[...] = 0.0

            # Solve
            conjugate_gradients(
                self.comm,
                self.decomposition,
                self.rhs_field,
                self.u_field,
                hessp=self._apply_stiffness,
                tol=self.tol,
                maxiter=self.maxiter,
                timer=timer,
            )

            # Compute strain from solution
            self.decomposition.communicate_ghosts(self.u_field)
            self.gradient_op.apply(self.u_field, self.gradient_field)
            grad = self.gradient_field.s

            # Symmetric strain
            strain_shape = (self.dim, self.dim, self.nb_quad) + tuple(self.nb_grid_pts)
            strain = np.zeros(strain_shape)
            for ii in range(self.dim):
                for jj in range(self.dim):
                    strain[ii, jj, ...] = 0.5 * (grad[ii, jj, ...] + grad[jj, ii, ...])

            # Add macroscopic strain
            for ii in range(self.dim):
                for jj in range(self.dim):
                    strain[ii, jj, ...] += E_macro[ii, jj]

            # Compute stress
            stress = np.zeros_like(strain)
            self._compute_stress(strain, stress)

            # Average stress
            sig_avg = np.zeros((self.dim, self.dim))
            for k in range(self.dim):
                for L in range(self.dim):
                    local_sum = 0.0
                    for q in range(self.nb_quad):
                        local_sum += self.quad_weights[q] * np.sum(stress[k, L, q, ...])
                    sig_avg[k, L] = self.comm.sum(local_sum)

            total_volume = np.prod(self.domain_size)
            sig_avg /= total_volume

            # Store in C_eff
            for k in range(self.dim):
                for L in range(self.dim):
                    voigt_row = voigt_index(self.dim, k, L)
                    if k <= L:
                        C_eff[voigt_row, voigt_col] = sig_avg[k, L]

        return C_eff


# =============================================================================
# Tests
# =============================================================================


class TestLaminateExactStiffness:
    """
    Test that computed laminate stiffness matches exact analytical formulas.

    For a laminate, the effective stiffness is NOT simply Voigt or Reuss average
    due to Poisson coupling effects. The exact analytical formulas account for
    stress/strain continuity constraints at layer interfaces.
    """

    @pytest.mark.parametrize("layer_dir", [0, 1])
    def test_exact_stiffness_2d(self, layer_dir):
        """Test full stiffness tensor against exact formula in 2D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (16, 16)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=layer_dir, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        # Compute exact analytical laminate stiffness
        C_analytical = laminate_stiffness_2d(E1, E2, nu, f1, layer_dir)

        np.testing.assert_allclose(
            C_eff,
            C_analytical,
            rtol=1e-4,
            atol=1e-8,  # Handle tiny numerical errors near zero
            err_msg=f"Exact laminate stiffness failed for 2D, layer_dir={layer_dir}",
        )

    @pytest.mark.parametrize("layer_dir", [0, 1, 2])
    def test_exact_stiffness_3d(self, layer_dir):
        """Test full stiffness tensor against exact formula in 3D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=layer_dir, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        # Compute exact analytical laminate stiffness
        C_analytical = laminate_stiffness_3d(E1, E2, nu, f1, layer_dir)

        np.testing.assert_allclose(
            C_eff,
            C_analytical,
            rtol=1e-4,
            atol=1e-8,  # Handle tiny numerical errors near zero
            err_msg=f"Exact laminate stiffness failed for 3D, layer_dir={layer_dir}",
        )


class TestLaminateReussBound:
    """
    Test that laminate with loading perpendicular to layers gives Reuss bound.

    For a laminate with layers normal to direction d, applying strain in
    direction d (perpendicular to layers) requires uniform stress at interfaces,
    giving Reuss bound.

    Physics:
    - Layers with normals in direction d stack perpendicular to d
    - Loading perpendicular to layers means stress must be continuous (equilibrium)
    - This gives Reuss (uniform stress / harmonic) average
    """

    @pytest.mark.parametrize("layer_dir", [0, 1])
    def test_reuss_bound_2d(self, layer_dir):
        """Test Reuss bound in 2D.

        For layers normal to layer_dir, loading in layer_dir (perpendicular to layers)
        should give Reuss bound.
        """
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (16, 16)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=layer_dir, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        C1 = isotropic_stiffness_2d(E1, nu)
        C2 = isotropic_stiffness_2d(E2, nu)
        C_reuss = compute_reuss_average(C1, C2, f1)

        # Loading perpendicular to layers should give Reuss bound
        np.testing.assert_allclose(
            C_eff[layer_dir, layer_dir],
            C_reuss[layer_dir, layer_dir],
            rtol=1e-4,
            err_msg=f"Reuss bound failed for 2D laminate, layer_dir={layer_dir}",
        )

    @pytest.mark.parametrize("layer_dir", [0, 1, 2])
    def test_reuss_bound_3d(self, layer_dir):
        """Test Reuss bound in 3D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=layer_dir, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        C1 = isotropic_stiffness_3d(E1, nu)
        C2 = isotropic_stiffness_3d(E2, nu)
        C_reuss = compute_reuss_average(C1, C2, f1)

        np.testing.assert_allclose(
            C_eff[layer_dir, layer_dir],
            C_reuss[layer_dir, layer_dir],
            rtol=1e-4,
            err_msg=f"Reuss bound failed for 3D laminate, layer_dir={layer_dir}",
        )


class TestLaminateShearModulus:
    """
    Test shear modulus of laminate.

    For isotropic phases, shear modulus follows specific mixing rules
    depending on shear plane orientation relative to layers.

    For a laminate with layers normal to x:
    - Shear in xy or xz plane (involves x direction) → Reuss average
    - Shear in yz plane (parallel to layers) → Voigt average
    """

    def test_shear_involving_layer_normal_2d(self):
        """Test shear modulus for shear involving the layer normal direction in 2D.

        For layers normal to x (layer_dir=0), shear in xy plane involves the
        layer normal direction, so we expect Reuss (harmonic) average.
        """
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (16, 16)

        # Layers normal to x (layer_dir=0)
        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        # Shear moduli
        mu1 = E1 / (2 * (1 + nu))
        mu2 = E2 / (2 * (1 + nu))

        # For shear in xy plane with layers normal to x:
        # Shear stress continuity at interfaces → Reuss averaging
        mu_reuss = 1.0 / (f1 / mu1 + (1 - f1) / mu2)

        # C[2,2] is the shear modulus in 2D Voigt notation
        np.testing.assert_allclose(
            C_eff[2, 2],
            mu_reuss,
            rtol=1e-4,
            err_msg="Shear modulus (Reuss) failed for laminate with layer_dir=0",
        )


class TestLaminateSymmetry:
    """Test that homogenized stiffness tensor has correct symmetry."""

    def test_major_symmetry_2d(self):
        """Test major symmetry C_ijkl = C_klij in 2D."""
        E1, E2 = 1.0, 5.0
        nu = 0.3
        nb_grid_pts = (16, 16)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        # Check symmetry with tolerance for numerical errors
        np.testing.assert_allclose(
            C_eff,
            C_eff.T,
            rtol=1e-6,
            atol=1e-8,
            err_msg="C_eff is not symmetric in 2D",
        )

    def test_major_symmetry_3d(self):
        """Test major symmetry C_ijkl = C_klij in 3D."""
        E1, E2 = 1.0, 5.0
        nu = 0.3
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        np.testing.assert_allclose(
            C_eff,
            C_eff.T,
            rtol=1e-6,
            atol=1e-8,
            err_msg="C_eff is not symmetric in 3D",
        )


class TestLaminatePositiveDefinite:
    """Test that homogenized stiffness tensor is positive definite."""

    def test_positive_definite_2d(self):
        """Test positive definiteness in 2D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        nb_grid_pts = (16, 16)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        eigenvalues = np.linalg.eigvalsh(C_eff)
        assert np.all(
            eigenvalues > 0
        ), f"C_eff not positive definite, eigenvalues: {eigenvalues}"

    def test_positive_definite_3d(self):
        """Test positive definiteness in 3D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        eigenvalues = np.linalg.eigvalsh(C_eff)
        assert np.all(
            eigenvalues > 0
        ), f"C_eff not positive definite, eigenvalues: {eigenvalues}"


class TestLaminateBoundsOrdering:
    """Test that effective properties lie between Voigt and Reuss bounds."""

    def test_bounds_ordering_2d(self):
        """Test bounds ordering in 2D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (16, 16)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        C1 = isotropic_stiffness_2d(E1, nu)
        C2 = isotropic_stiffness_2d(E2, nu)
        C_voigt = compute_voigt_average(C1, C2, f1)
        C_reuss = compute_reuss_average(C1, C2, f1)

        # All diagonal components should be between Reuss and Voigt
        # Use small tolerance for numerical precision
        tol = 1e-8
        for i in range(3):
            assert C_reuss[i, i] - tol <= C_eff[i, i] <= C_voigt[i, i] + tol, (
                f"Bounds violated for C[{i},{i}]: "
                f"Reuss={C_reuss[i, i]}, Eff={C_eff[i, i]}, Voigt={C_voigt[i, i]}"
            )

    def test_bounds_ordering_3d(self):
        """Test bounds ordering in 3D."""
        E1, E2 = 1.0, 10.0
        nu = 0.3
        f1 = 0.5
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E1, E2, nu, layer_direction=0, volume_fraction_1=f1
        )
        C_eff = homog.compute_homogenized_stiffness()

        C1 = isotropic_stiffness_3d(E1, nu)
        C2 = isotropic_stiffness_3d(E2, nu)
        C_voigt = compute_voigt_average(C1, C2, f1)
        C_reuss = compute_reuss_average(C1, C2, f1)

        for i in range(6):
            assert C_reuss[i, i] - 1e-10 <= C_eff[i, i] <= C_voigt[i, i] + 1e-10, (
                f"Bounds violated for C[{i},{i}]: "
                f"Reuss={C_reuss[i, i]}, Eff={C_eff[i, i]}, Voigt={C_voigt[i, i]}"
            )


class TestHomogeneousMaterial:
    """Test that homogeneous material gives exact input properties."""

    def test_homogeneous_2d(self):
        """Test homogeneous material in 2D."""
        E = 10.0
        nu = 0.3
        nb_grid_pts = (16, 16)

        # Same E for both phases = homogeneous
        homog = LaminateHomogenization(
            nb_grid_pts, E, E, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        C_analytical = isotropic_stiffness_2d(E, nu)

        np.testing.assert_allclose(
            C_eff,
            C_analytical,
            rtol=1e-6,
            err_msg="Homogeneous material does not match analytical",
        )

    def test_homogeneous_3d(self):
        """Test homogeneous material in 3D."""
        E = 10.0
        nu = 0.3
        nb_grid_pts = (8, 8, 8)

        homog = LaminateHomogenization(
            nb_grid_pts, E, E, nu, layer_direction=0, volume_fraction_1=0.5
        )
        C_eff = homog.compute_homogenized_stiffness()

        C_analytical = isotropic_stiffness_3d(E, nu)

        np.testing.assert_allclose(
            C_eff,
            C_analytical,
            rtol=1e-6,
            err_msg="Homogeneous material does not match analytical",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
