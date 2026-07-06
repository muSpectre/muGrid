/**
 * @file   isotropic_stiffness_operator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   31 Dec 2025
 *
 * @brief  Host implementation of fused isotropic stiffness operator
 *
 * Copyright © 2025 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "isotropic_stiffness.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#include <cassert>
#include <cstddef>
#include <vector>

namespace muGrid {

    // Forward declarations of the (templated) host kernels defined at the
    // bottom of this file, so the operator's impl methods above can call them.
    namespace isotropic_stiffness_kernels {
        template <typename T>
        void isotropic_stiffness_3d_host(
            const T *, const T *, const T *, T *, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, Index_t,
            const T *, const T *, T, bool);
        template <typename T>
        void isotropic_stiffness_3d_host_uniform(
            const T *, T, T, T *, Index_t, Index_t, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, const T *,
            const T *, T, bool);
        template <typename T>
        void isotropic_stiffness_3d_host_macro_rhs(
            const T *, const T *, T *, Index_t, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, const T *,
            const T *, T, bool);
        template <typename T>
        void isotropic_stiffness_3d_host_average(
            const T *, const T *, const T *, Index_t, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, const T *,
            const T *, Real, Real *);
        template <typename T>
        void isotropic_stiffness_3d_host_sensitivity(
            const T *, const T *, T *, T *, Index_t, Index_t, Index_t, Index_t,
            Index_t, Index_t, Index_t, Index_t, Index_t, Index_t, const T *,
            const T *, const T *, const T *);
    }  // namespace isotropic_stiffness_kernels

    // ============================================================================
    // 3D Implementation
    // ============================================================================

    // Shape-function gradients and quadrature weights come from the element
    // traits (fem_element.hh) via the operator's runtime element data.

    template <>
    void IsotropicStiffnessOperator<3>::precompute_matrices() {
        // Initialize matrices to zero
        std::fill(G_matrix.begin(), G_matrix.end(), 0.0);
        std::fill(V_matrix.begin(), V_matrix.end(), 0.0);

        const Real hx = grid_spacing[0];
        const Real hy = grid_spacing[1];
        const Real hz = grid_spacing[2];
        const Real scale[3] = {1.0 / hx, 1.0 / hy, 1.0 / hz};

        // Selected element's reference shape-function gradient B[q][d][n].
        auto B = [this](Index_t q, Index_t d, Index_t n) {
            return elem_B[(q * NB_DOFS_PER_NODE + d) * NB_NODES + n];
        };

        // G = Σ_q w_q B_q^T I_sym B_q (shear/μ); V = Σ_q w_q (B_q^T m)(m^T B_q)
        // (volumetric/λ), m = [1,1,1,0,0,0]^T.
        for (Index_t q = 0; q < elem_nb_quad; ++q) {
            Real w = elem_Wfrac[q] * hx * hy * hz;  // includes element volume
            for (Index_t I = 0; I < NB_NODES; ++I) {
                for (Index_t J = 0; J < NB_NODES; ++J) {
                    for (Index_t a = 0; a < NB_DOFS_PER_NODE; ++a) {
                        for (Index_t b = 0; b < NB_DOFS_PER_NODE; ++b) {
                            Real g_contrib = 0.0;
                            if (a == b) {
                                g_contrib += B(q, a, I) * scale[a] *
                                             B(q, a, J) * scale[a];
                            }
                            Real dNI_dx = B(q, 0, I) * scale[0];
                            Real dNI_dy = B(q, 1, I) * scale[1];
                            Real dNI_dz = B(q, 2, I) * scale[2];
                            Real dNJ_dx = B(q, 0, J) * scale[0];
                            Real dNJ_dy = B(q, 1, J) * scale[1];
                            Real dNJ_dz = B(q, 2, J) * scale[2];
                            // Engineering-shear coupling (2ε), factor 1/2.
                            if (a == 0 && b == 0) {
                                g_contrib +=
                                    0.5 * dNI_dy * dNJ_dy + 0.5 * dNI_dz * dNJ_dz;
                            } else if (a == 1 && b == 1) {
                                g_contrib +=
                                    0.5 * dNI_dx * dNJ_dx + 0.5 * dNI_dz * dNJ_dz;
                            } else if (a == 2 && b == 2) {
                                g_contrib +=
                                    0.5 * dNI_dx * dNJ_dx + 0.5 * dNI_dy * dNJ_dy;
                            } else if (a == 0 && b == 1) {
                                g_contrib += 0.5 * dNI_dy * dNJ_dx;
                            } else if (a == 1 && b == 0) {
                                g_contrib += 0.5 * dNI_dx * dNJ_dy;
                            } else if (a == 0 && b == 2) {
                                g_contrib += 0.5 * dNI_dz * dNJ_dx;
                            } else if (a == 2 && b == 0) {
                                g_contrib += 0.5 * dNI_dx * dNJ_dz;
                            } else if (a == 1 && b == 2) {
                                g_contrib += 0.5 * dNI_dz * dNJ_dy;
                            } else if (a == 2 && b == 1) {
                                g_contrib += 0.5 * dNI_dy * dNJ_dz;
                            }
                            // V: (B^T m)(m^T B), trace(ε) = εxx + εyy + εzz.
                            Real BtmI = (a == 0 ? dNI_dx : 0.0) +
                                        (a == 1 ? dNI_dy : 0.0) +
                                        (a == 2 ? dNI_dz : 0.0);
                            Real BtmJ = (b == 0 ? dNJ_dx : 0.0) +
                                        (b == 1 ? dNJ_dy : 0.0) +
                                        (b == 2 ? dNJ_dz : 0.0);
                            Index_t idx =
                                (I * NB_DOFS_PER_NODE + a) * NB_ELEMENT_DOFS +
                                (J * NB_DOFS_PER_NODE + b);
                            G_matrix[idx] += w * g_contrib;
                            V_matrix[idx] += w * BtmI * BtmJ;
                        }
                    }
                }
            }
        }

        // Element-averaged gradient operator: Dbar[j*NB_NODES+n] =
        // scale[j] Σ_q Wfrac_q B[q][j][n] (fractional weights, summing to 1).
        std::fill(Dbar_matrix.begin(), Dbar_matrix.end(), 0.0);
        for (Index_t j = 0; j < NB_DOFS_PER_NODE; ++j) {
            for (Index_t n = 0; n < NB_NODES; ++n) {
                Real s = 0.0;
                for (Index_t q = 0; q < elem_nb_quad; ++q) {
                    s += elem_Wfrac[q] * B(q, j, n);
                }
                Dbar_matrix[j * NB_NODES + n] = scale[j] * s;
            }
        }
    }

    template <>
    template <typename T>
    void IsotropicStiffnessOperator<3>::apply_impl(
        const TypedFieldBase<T> & displacement,
        const TypedFieldBase<T> & lambda, const TypedFieldBase<T> & mu,
        T alpha, TypedFieldBase<T> & force, bool increment) const {

        // Validate field collections and ghosts; the dimension-generic checks
        // live in internal::validate_stiffness_fields (isotropic_stiffness.hh).
        const auto info = internal::validate_stiffness_fields<3>(
            displacement.get_collection(), lambda.get_collection());
        const auto * disp_global_fc = info.disp_fc;
        const auto * mat_global_fc = info.mat_fc;

        // Computable region (node field == material field, guaranteed above).
        const Index_t nnx = info.nb_computable[0];
        const Index_t nny = info.nb_computable[1];
        const Index_t nnz = info.nb_computable[2];
        const Index_t nelx = nnx;
        const Index_t nely = nny;
        const Index_t nelz = nnz;

        // Stencil offset: one ghost cell on each side (see validation helper).
        constexpr Index_t STENCIL_LEFT = 1;

        // Node dimensions (for displacement/force fields with ghosts)
        auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];
        Index_t ny = nb_nodes[1];

        // Offset to first computable node (based on stencil requirements, not ghost size)
        // This allows computing in ghost regions where stencil has valid data
        Index_t disp_offset_x = STENCIL_LEFT;
        Index_t disp_offset_y = STENCIL_LEFT;
        Index_t disp_offset_z = STENCIL_LEFT;
        Index_t mat_offset_x = STENCIL_LEFT;
        Index_t mat_offset_y = STENCIL_LEFT;
        Index_t mat_offset_z = STENCIL_LEFT;

        // Compute strides (AoS layout for host)
        Index_t disp_stride_d = 1;
        Index_t disp_stride_x = NB_DOFS_PER_NODE;
        Index_t disp_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t disp_stride_z = NB_DOFS_PER_NODE * nx * ny;

        // Material field strides (with ghosts if present)
        auto mat_nb_pts_with_ghosts =
            mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t mat_nx = mat_nb_pts_with_ghosts[0];
        Index_t mat_ny = mat_nb_pts_with_ghosts[1];
        Index_t mat_stride_x = 1;
        Index_t mat_stride_y = mat_nx;
        Index_t mat_stride_z = mat_nx * mat_ny;

        Index_t force_stride_d = 1;
        Index_t force_stride_x = NB_DOFS_PER_NODE;
        Index_t force_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t force_stride_z = NB_DOFS_PER_NODE * nx * ny;

        // Offset data pointers to point to first computable node
        Index_t disp_offset = disp_offset_x * disp_stride_x +
                              disp_offset_y * disp_stride_y +
                              disp_offset_z * disp_stride_z;
        Index_t force_offset = disp_offset_x * force_stride_x +
                               disp_offset_y * force_stride_y +
                               disp_offset_z * force_stride_z;
        Index_t mat_offset = mat_offset_x * mat_stride_x +
                             mat_offset_y * mat_stride_y +
                             mat_offset_z * mat_stride_z;

        const T * disp_data = displacement.data() + disp_offset;
        const T * lambda_data = lambda.data() + mat_offset;
        const T * mu_data = mu.data() + mat_offset;
        T * force_data = force.data() + force_offset;

        std::array<T, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> G_s, V_s;
        const T * G_ptr = geometry_as<T>(G_matrix, G_s);
        const T * V_ptr = geometry_as<T>(V_matrix, V_s);

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host<T>(
            disp_data, lambda_data, mu_data, force_data, nnx, nny,
            nnz,               // Number of interior nodes
            nelx, nely, nelz,  // Number of elements
            disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
            mat_stride_x, mat_stride_y, mat_stride_z, force_stride_x,
            force_stride_y, force_stride_z, force_stride_d, G_ptr,
            V_ptr, alpha, increment);
    }

    template <>
    template <typename T>
    void IsotropicStiffnessOperator<3>::apply_uniform_impl(
        const TypedFieldBase<T> & displacement, T lambda, T mu,
        T alpha, TypedFieldBase<T> & force, bool increment) const {

        // Uniform Lamé scalars: no material field, so the only geometry comes
        // from the displacement/force collection. Mirrors apply_impl<3> minus
        // all material-field discovery and validation.
        auto & disp_coll = displacement.get_collection();
        auto * disp_global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&disp_coll);
        if (!disp_global_fc) {
            throw RuntimeError(
                "IsotropicStiffnessOperator3D requires GlobalFieldCollection");
        }

        auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
        auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
        if (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 ||
            nb_ghosts_left[2] < 1 || nb_ghosts_right[0] < 1 ||
            nb_ghosts_right[1] < 1 || nb_ghosts_right[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on both sides of "
                               "displacement/force fields");
        }

        constexpr Index_t STENCIL_LEFT = 1;
        constexpr Index_t STENCIL_RIGHT = 1;

        auto nb_with_ghosts =
            disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nnx = nb_with_ghosts[0] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nny = nb_with_ghosts[1] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nnz = nb_with_ghosts[2] - STENCIL_LEFT - STENCIL_RIGHT;

        Index_t nx = nb_with_ghosts[0];
        Index_t ny = nb_with_ghosts[1];

        Index_t disp_stride_d = 1;
        Index_t disp_stride_x = NB_DOFS_PER_NODE;
        Index_t disp_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t disp_stride_z = NB_DOFS_PER_NODE * nx * ny;
        Index_t force_stride_d = 1;
        Index_t force_stride_x = NB_DOFS_PER_NODE;
        Index_t force_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t force_stride_z = NB_DOFS_PER_NODE * nx * ny;

        Index_t disp_offset = STENCIL_LEFT * disp_stride_x +
                              STENCIL_LEFT * disp_stride_y +
                              STENCIL_LEFT * disp_stride_z;
        Index_t force_offset = STENCIL_LEFT * force_stride_x +
                               STENCIL_LEFT * force_stride_y +
                               STENCIL_LEFT * force_stride_z;

        std::array<T, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> G_s, V_s;
        const T * G_ptr = geometry_as<T>(G_matrix, G_s);
        const T * V_ptr = geometry_as<T>(V_matrix, V_s);

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host_uniform<T>(
            displacement.data() + disp_offset, lambda, mu,
            force.data() + force_offset, nnx, nny, nnz, disp_stride_x,
            disp_stride_y, disp_stride_z, disp_stride_d, force_stride_x,
            force_stride_y, force_stride_z, force_stride_d, G_ptr,
            V_ptr, alpha, increment);
    }

    template <>
    template <typename T>
    void IsotropicStiffnessOperator<3>::apply_macro_rhs_impl(
        const TypedFieldBase<T> & lambda, const TypedFieldBase<T> & mu,
        const std::array<Real, 9> & E_macro,
        TypedFieldBase<T> & force) const {

        // The force field plays the role of the node field for geometry; the
        // displacement is the constant affine pattern folded into Gu, Vu.
        const auto info = internal::validate_stiffness_fields<3>(
            force.get_collection(), lambda.get_collection());
        const auto * node_global_fc = info.disp_fc;
        const auto * mat_global_fc = info.mat_fc;

        const Index_t nnx = info.nb_computable[0];
        const Index_t nny = info.nb_computable[1];
        const Index_t nnz = info.nb_computable[2];
        constexpr Index_t STENCIL_LEFT = 1;

        auto nb_nodes = node_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];
        Index_t ny = nb_nodes[1];
        auto mat_nb = mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t mat_nx = mat_nb[0];
        Index_t mat_ny = mat_nb[1];

        Index_t mat_stride_x = 1;
        Index_t mat_stride_y = mat_nx;
        Index_t mat_stride_z = mat_nx * mat_ny;
        Index_t force_stride_d = 1;
        Index_t force_stride_x = NB_DOFS_PER_NODE;
        Index_t force_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t force_stride_z = NB_DOFS_PER_NODE * nx * ny;

        Index_t force_offset = STENCIL_LEFT * force_stride_x +
                               STENCIL_LEFT * force_stride_y +
                               STENCIL_LEFT * force_stride_z;
        Index_t mat_offset = STENCIL_LEFT * mat_stride_x +
                             STENCIL_LEFT * mat_stride_y +
                             STENCIL_LEFT * mat_stride_z;

        ElementMatrix Gu{}, Vu{};
        this->macro_rhs_vectors(E_macro, Gu, Vu);
        std::array<T, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> Gu_s, Vu_s;
        const T * Gu_ptr = geometry_as<T>(Gu, Gu_s);
        const T * Vu_ptr = geometry_as<T>(Vu, Vu_s);

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host_macro_rhs<T>(
            lambda.data() + mat_offset, mu.data() + mat_offset,
            force.data() + force_offset, nnx, nny, nnz, mat_stride_x,
            mat_stride_y, mat_stride_z, force_stride_x, force_stride_y,
            force_stride_z, force_stride_d, Gu_ptr, Vu_ptr,
            static_cast<T>(1), false);
    }

    template <>
    template <typename T>
    void IsotropicStiffnessOperator<3>::assemble_diagonal_impl(
        const TypedFieldBase<T> & lambda, const TypedFieldBase<T> & mu,
        TypedFieldBase<T> & diagonal) const {

        // diag(K) = Σ_e (2μ_e diag(G) + λ_e diag(V)); this is the macro-RHS
        // gather with the constant per-element vectors Gu, Vu replaced by the
        // diagonals of the element matrices G, V. The diagonal field plays the
        // role of the node (force) field.
        const auto info = internal::validate_stiffness_fields<3>(
            diagonal.get_collection(), lambda.get_collection());
        const auto * node_global_fc = info.disp_fc;
        const auto * mat_global_fc = info.mat_fc;

        const Index_t nnx = info.nb_computable[0];
        const Index_t nny = info.nb_computable[1];
        const Index_t nnz = info.nb_computable[2];
        constexpr Index_t STENCIL_LEFT = 1;

        auto nb_nodes = node_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];
        Index_t ny = nb_nodes[1];
        auto mat_nb = mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t mat_nx = mat_nb[0];
        Index_t mat_ny = mat_nb[1];

        Index_t mat_stride_x = 1;
        Index_t mat_stride_y = mat_nx;
        Index_t mat_stride_z = mat_nx * mat_ny;
        Index_t force_stride_d = 1;
        Index_t force_stride_x = NB_DOFS_PER_NODE;
        Index_t force_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t force_stride_z = NB_DOFS_PER_NODE * nx * ny;

        Index_t force_offset = STENCIL_LEFT * force_stride_x +
                               STENCIL_LEFT * force_stride_y +
                               STENCIL_LEFT * force_stride_z;
        Index_t mat_offset = STENCIL_LEFT * mat_stride_x +
                             STENCIL_LEFT * mat_stride_y +
                             STENCIL_LEFT * mat_stride_z;

        // Diagonals of the element matrices (only the first NB_ELEMENT_DOFS
        // entries are read by the macro-RHS kernel, indexed by local DOF row).
        ElementMatrix Gd{}, Vd{};
        for (Index_t r = 0; r < NB_ELEMENT_DOFS; ++r) {
            Gd[r] = G_matrix[r * NB_ELEMENT_DOFS + r];
            Vd[r] = V_matrix[r * NB_ELEMENT_DOFS + r];
        }
        std::array<T, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> Gd_s, Vd_s;
        const T * Gd_ptr = geometry_as<T>(Gd, Gd_s);
        const T * Vd_ptr = geometry_as<T>(Vd, Vd_s);

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host_macro_rhs<T>(
            lambda.data() + mat_offset, mu.data() + mat_offset,
            diagonal.data() + force_offset, nnx, nny, nnz, mat_stride_x,
            mat_stride_y, mat_stride_z, force_stride_x, force_stride_y,
            force_stride_z, force_stride_d, Gd_ptr, Vd_ptr,
            static_cast<T>(1), false);
    }

    template <>
    template <typename T>
    void IsotropicStiffnessOperator<3>::compute_sensitivity_impl(
        const TypedFieldBase<T> & forward_disp,
        const std::array<Real, 9> & forward_macro,
        const TypedFieldBase<T> & costate_disp,
        const std::array<Real, 9> & costate_macro,
        TypedFieldBase<T> & g_shear, TypedFieldBase<T> & g_vol) const {

        // See the 2D counterpart: g_shear = aₑᵀ G bₑ, g_vol = aₑᵀ V bₑ over the
        // owned elements, per-pixel output on the material collection.
        const auto info = internal::validate_stiffness_fields<3>(
            forward_disp.get_collection(), g_shear.get_collection());
        const auto * disp_fc = info.disp_fc;
        const auto * out_fc = info.mat_fc;

        auto disp_gl = disp_fc->get_nb_ghosts_left();
        auto disp_gr = disp_fc->get_nb_ghosts_right();
        auto disp_wg = disp_fc->get_nb_subdomain_grid_pts_with_ghosts();
        auto out_gl = out_fc->get_nb_ghosts_left();
        auto out_wg = out_fc->get_nb_subdomain_grid_pts_with_ghosts();

        const Index_t nelx = disp_wg[0] - disp_gl[0] - disp_gr[0];
        const Index_t nely = disp_wg[1] - disp_gl[1] - disp_gr[1];
        const Index_t nelz = disp_wg[2] - disp_gl[2] - disp_gr[2];

        Index_t nx = disp_wg[0], ny = disp_wg[1];
        Index_t out_nx = out_wg[0], out_ny = out_wg[1];

        Index_t disp_stride_d = 1;
        Index_t disp_stride_x = NB_DOFS_PER_NODE;
        Index_t disp_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t disp_stride_z = NB_DOFS_PER_NODE * nx * ny;
        Index_t out_stride_x = 1;
        Index_t out_stride_y = out_nx;
        Index_t out_stride_z = out_nx * out_ny;

        Index_t disp_offset = disp_gl[0] * disp_stride_x +
                              disp_gl[1] * disp_stride_y +
                              disp_gl[2] * disp_stride_z;
        Index_t out_offset = out_gl[0] * out_stride_x +
                             out_gl[1] * out_stride_y +
                             out_gl[2] * out_stride_z;

        std::array<Real, NB_ELEMENT_DOFS> ustar_f{}, ustar_c{};
        this->affine_element_dofs(forward_macro, ustar_f);
        this->affine_element_dofs(costate_macro, ustar_c);

        std::array<T, NB_ELEMENT_DOFS * NB_ELEMENT_DOFS> G_s, V_s;
        std::array<T, NB_ELEMENT_DOFS> uf_s, uc_s;
        const T * G_ptr = geometry_as<T>(G_matrix, G_s);
        const T * V_ptr = geometry_as<T>(V_matrix, V_s);
        const T * uf_ptr = geometry_as<T>(ustar_f, uf_s);
        const T * uc_ptr = geometry_as<T>(ustar_c, uc_s);

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host_sensitivity<T>(
            forward_disp.data() + disp_offset,
            costate_disp.data() + disp_offset, g_shear.data() + out_offset,
            g_vol.data() + out_offset, nelx, nely, nelz, disp_stride_x,
            disp_stride_y, disp_stride_z, disp_stride_d, out_stride_x,
            out_stride_y, out_stride_z, G_ptr, V_ptr, uf_ptr, uc_ptr);
    }

    template <>
    template <typename T>
    std::array<Real, 9> IsotropicStiffnessOperator<3>::average_stress_impl(
        const TypedFieldBase<T> & displacement,
        const TypedFieldBase<T> & lambda, const TypedFieldBase<T> & mu,
        const std::array<Real, 9> & E_macro) const {

        const auto info = internal::validate_stiffness_fields<3>(
            displacement.get_collection(), lambda.get_collection());
        const auto * disp_global_fc = info.disp_fc;
        const auto * mat_global_fc = info.mat_fc;

        // Integrate over exactly the *owned* voxels (one element per voxel),
        // not the stencil-computable region: the latter (with_ghosts - 2) can
        // include ghost/padding elements -- e.g. the FFTEngine real-space
        // collection pads one axis beyond the requested ghosts -- which carry
        // nonzero periodic material and would double-count into the integral.
        auto disp_gl = disp_global_fc->get_nb_ghosts_left();
        auto disp_gr = disp_global_fc->get_nb_ghosts_right();
        auto disp_wg = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        auto mat_gl = mat_global_fc->get_nb_ghosts_left();
        auto mat_wg = mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();

        const Index_t nelx = disp_wg[0] - disp_gl[0] - disp_gr[0];
        const Index_t nely = disp_wg[1] - disp_gl[1] - disp_gr[1];
        const Index_t nelz = disp_wg[2] - disp_gl[2] - disp_gr[2];

        Index_t nx = disp_wg[0];
        Index_t ny = disp_wg[1];
        Index_t mat_nx = mat_wg[0];
        Index_t mat_ny = mat_wg[1];

        Index_t disp_stride_d = 1;
        Index_t disp_stride_x = NB_DOFS_PER_NODE;
        Index_t disp_stride_y = NB_DOFS_PER_NODE * nx;
        Index_t disp_stride_z = NB_DOFS_PER_NODE * nx * ny;
        Index_t mat_stride_x = 1;
        Index_t mat_stride_y = mat_nx;
        Index_t mat_stride_z = mat_nx * mat_ny;

        Index_t disp_offset = disp_gl[0] * disp_stride_x +
                              disp_gl[1] * disp_stride_y +
                              disp_gl[2] * disp_stride_z;
        Index_t mat_offset = mat_gl[0] * mat_stride_x +
                             mat_gl[1] * mat_stride_y +
                             mat_gl[2] * mat_stride_z;

        const Real vol_elem =
            grid_spacing[0] * grid_spacing[1] * grid_spacing[2];

        std::array<T, NB_DOFS_PER_NODE * NB_NODES> Dbar_s;
        std::array<T, 9> E_s;
        const T * Dbar_ptr = geometry_as<T>(Dbar_matrix, Dbar_s);
        const T * E_ptr = geometry_as<T>(E_macro, E_s);

        std::array<Real, 9> accum{};
        isotropic_stiffness_kernels::isotropic_stiffness_3d_host_average<T>(
            displacement.data() + disp_offset, lambda.data() + mat_offset,
            mu.data() + mat_offset, nelx, nely, nelz, disp_stride_x,
            disp_stride_y, disp_stride_z, disp_stride_d, mat_stride_x,
            mat_stride_y, mat_stride_z, Dbar_ptr, E_ptr,
            vol_elem, accum.data());
        return accum;
    }

    // ============================================================================
    // Host Kernel Implementations
    // ============================================================================

    namespace isotropic_stiffness_kernels {

        // Node offsets for 3D [node][dim]
        [[maybe_unused]] static const Index_t NODE_OFFSET_3D[8][3] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        // Vectorized shared kernel body for the per-pixel (Uniform == false)
        // and spatially-uniform (Uniform == true) stiffness apply. See the 2D
        // counterpart for the full rationale: with the compile-time AoS
        // x-strides the entry points hard-code (component stride 1, node
        // stride NB_DOFS, material stride 1), every address within one grid
        // row is affine in ix with a small constant offset, so the per-node
        // computation is straight-line code the compiler vectorizes across
        // ix. The row contraction is factored as
        // f += 2μ·(G_row·u) + λ·(V_row·u), keeping the geometry coefficients
        // loop-invariant broadcasts.
        template <typename T, bool Uniform, bool Increment>
        static void isotropic_stiffness_3d_row_kernel(
            const T * MUGRID_RESTRICT displacement,
            const T * MUGRID_RESTRICT lambda,
            const T * MUGRID_RESTRICT mu, T * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nnz, Index_t disp_stride_y,
            Index_t disp_stride_z, Index_t mat_stride_y, Index_t mat_stride_z,
            Index_t force_stride_y, Index_t force_stride_z,
            const T * MUGRID_RESTRICT G, const T * MUGRID_RESTRICT V, T alpha,
            T lam_u, T mu_u) {

            constexpr Index_t NB_DOFS = 3;
            constexpr Index_t NB_ELEM_DOFS = 24;

            // Signed strides: ghost rows/columns sit at negative offsets.
            using SIndex_t = std::ptrdiff_t;
            const SIndex_t dy = static_cast<SIndex_t>(disp_stride_y);
            const SIndex_t dz = static_cast<SIndex_t>(disp_stride_z);
            const SIndex_t my = static_cast<SIndex_t>(mat_stride_y);
            const SIndex_t mz = static_cast<SIndex_t>(mat_stride_z);
            const SIndex_t fy = static_cast<SIndex_t>(force_stride_y);
            const SIndex_t fz = static_cast<SIndex_t>(force_stride_z);

            // One element's contribution to its local node `ln` (rows
            // r0 = 3·ln .. r0+2 of K_e = 2μG + λV applied to the gathered
            // element displacement). The element's 8 nodes live on 4 nodal
            // rows (y-offset × z-offset), each holding 2 x-consecutive nodes,
            // i.e. 2·NB_DOFS consecutive values per row starting at
            // xb = NB_DOFS·(ix + element x-offset).
            const auto elem_contrib =
                [G, V](const T * MUGRID_RESTRICT r00,
                       const T * MUGRID_RESTRICT r10,
                       const T * MUGRID_RESTRICT r01,
                       const T * MUGRID_RESTRICT r11, SIndex_t xb, Index_t r0,
                       T two_mu, T lam, T & f0, T & f1, T & f2) {
                    // Contract the three K_e rows of this node directly
                    // against the four nodal rows (u[j] = row[xb + d] with
                    // j = 6·row_index + d) — no local u[] array, so nothing
                    // the vectorizer would have to keep in a per-lane stack
                    // slot.
                    const T * MUGRID_RESTRICT Gr = G + r0 * NB_ELEM_DOFS;
                    const T * MUGRID_RESTRICT Vr = V + r0 * NB_ELEM_DOFS;
                    T g0{0}, v0{0}, g1{0}, v1{0}, g2{0}, v2{0};
                    for (Index_t d = 0; d < 2 * NB_DOFS; ++d) {
                        const T u0 = r00[xb + d];
                        const T u1 = r10[xb + d];
                        const T u2 = r01[xb + d];
                        const T u3 = r11[xb + d];
                        g0 += Gr[d] * u0 + Gr[6 + d] * u1 + Gr[12 + d] * u2 +
                              Gr[18 + d] * u3;
                        v0 += Vr[d] * u0 + Vr[6 + d] * u1 + Vr[12 + d] * u2 +
                              Vr[18 + d] * u3;
                        g1 += Gr[24 + d] * u0 + Gr[30 + d] * u1 +
                              Gr[36 + d] * u2 + Gr[42 + d] * u3;
                        v1 += Vr[24 + d] * u0 + Vr[30 + d] * u1 +
                              Vr[36 + d] * u2 + Vr[42 + d] * u3;
                        g2 += Gr[48 + d] * u0 + Gr[54 + d] * u1 +
                              Gr[60 + d] * u2 + Gr[66 + d] * u3;
                        v2 += Vr[48 + d] * u0 + Vr[54 + d] * u1 +
                              Vr[60 + d] * u2 + Vr[66 + d] * u3;
                    }
                    f0 += two_mu * g0 + lam * v0;
                    f1 += two_mu * g1 + lam * v1;
                    f2 += two_mu * g2 + lam * v2;
                };

            // Row-local force accumulator: the eight incident elements are
            // applied in eight separate sweeps over the x-row, each
            // accumulating into this L1-resident buffer, and the buffer is
            // flushed to the force field once per row. One sweep's loop body
            // (a 4-row gather plus three 24-term row contractions) is small
            // enough for the vectorizer; the single-loop form with all eight
            // elements inline is not (clang bails on the ~1300-op body).
            std::vector<T> facc_storage(NB_DOFS *
                                        static_cast<std::size_t>(nnx));
            T * MUGRID_RESTRICT facc = facc_storage.data();

            // One element sweep: for every node of the row, the contribution
            // of the incident element at x-offset `coff` whose nodal rows are
            // r00/r10/r01/r11 and whose rows of K_e start at r0 = 3·ln. The
            // accumulator is an explicit restrict parameter (not a capture):
            // a by-reference capture hides the pointer behind the closure
            // struct and the vectorizer then "cannot identify array bounds".
            const auto elem_sweep = [&elem_contrib, lam_u, mu_u](
                                        const T * MUGRID_RESTRICT r00,
                                        const T * MUGRID_RESTRICT r10,
                                        const T * MUGRID_RESTRICT r01,
                                        const T * MUGRID_RESTRICT r11,
                                        const T * MUGRID_RESTRICT lrow,
                                        const T * MUGRID_RESTRICT mrow,
                                        T * MUGRID_RESTRICT facc_p,
                                        SIndex_t coff, Index_t r0,
                                        Index_t nx_count) {
                // assume_safety: the only store target is the private facc
                // buffer (never aliased by the field rows), so the runtime
                // alias checks the vectorizer would otherwise emit over the
                // seven live pointers are provably unnecessary — and their
                // number is what makes the plain vectorize(enable) transform
                // bail out.
                #if defined(_MSC_VER)
                #pragma loop(ivdep)
                #elif defined(__clang__)
                #pragma clang loop vectorize(assume_safety) interleave(enable)
                #elif defined(__GNUC__) && !defined(__NVCOMPILER)
                #pragma GCC ivdep
                #endif
                for (SIndex_t ix = 0; ix < static_cast<SIndex_t>(nx_count);
                     ++ix) {
                    T lam, mu_val;
                    if constexpr (Uniform) {
                        lam = lam_u;
                        mu_val = mu_u;
                    } else {
                        lam = lrow[ix + coff];
                        mu_val = mrow[ix + coff];
                    }
                    T f0{0}, f1{0}, f2{0};
                    elem_contrib(r00, r10, r01, r11, NB_DOFS * (ix + coff), r0,
                                 static_cast<T>(2) * mu_val, lam, f0, f1, f2);
                    facc_p[NB_DOFS * ix + 0] += f0;
                    facc_p[NB_DOFS * ix + 1] += f1;
                    facc_p[NB_DOFS * ix + 2] += f2;
                }
            };

            for (SIndex_t iz = 0; iz < static_cast<SIndex_t>(nnz); ++iz) {
                for (SIndex_t iy = 0; iy < static_cast<SIndex_t>(nny); ++iy) {
                    // Nodal rows (iy-1+a, iz-1+b), a, b in {0,1,2}, and
                    // material rows (iy-1+a, iz-1+b), a, b in {0,1}; the
                    // out-of-range rows are ghost rows guaranteed by the
                    // stencil requirement.
                    const T * drow[3][3];
                    for (Index_t a = 0; a < 3; ++a) {
                        for (Index_t b = 0; b < 3; ++b) {
                            drow[a][b] = displacement + (iy - 1 + a) * dy +
                                         (iz - 1 + b) * dz;
                        }
                    }
                    const T * lrow[2][2]{{nullptr, nullptr},
                                         {nullptr, nullptr}};
                    const T * mrow[2][2]{{nullptr, nullptr},
                                         {nullptr, nullptr}};
                    if constexpr (!Uniform) {
                        for (Index_t a = 0; a < 2; ++a) {
                            for (Index_t b = 0; b < 2; ++b) {
                                lrow[a][b] = lambda + (iy - 1 + a) * my +
                                             (iz - 1 + b) * mz;
                                mrow[a][b] = mu + (iy - 1 + a) * my +
                                             (iz - 1 + b) * mz;
                            }
                        }
                    }
                    T * MUGRID_RESTRICT f_row = force + iy * fy + iz * fz;

                    for (SIndex_t i = 0;
                         i < static_cast<SIndex_t>(NB_DOFS * nnx); ++i) {
                        facc[i] = T{0};
                    }

                    // The eight incident elements at offsets
                    // (eox, eoy, eoz) in {-1,0}^3; the node (ix, iy, iz) is
                    // local node (eox?1:0) + 2·(eoy?1:0) + 4·(eoz?1:0) of
                    // each. Element (·, eoy, eoz) reads nodal rows
                    // drow[eoy+1 + {0,1}][eoz+1 + {0,1}] and material row
                    // [eoy+1][eoz+1] at column ix+eox.
                    const auto sweep_pair = [&](Index_t a, Index_t b,
                                                Index_t r0_left,
                                                Index_t r0_right) {
                        elem_sweep(drow[a][b], drow[a + 1][b], drow[a][b + 1],
                                   drow[a + 1][b + 1], lrow[a][b], mrow[a][b],
                                   facc, -1, r0_left, nnx);
                        elem_sweep(drow[a][b], drow[a + 1][b], drow[a][b + 1],
                                   drow[a + 1][b + 1], lrow[a][b], mrow[a][b],
                                   facc, 0, r0_right, nnx);
                    };
                    sweep_pair(0, 0, 21, 18);  // (·,-1,-1): ln 7 / ln 6
                    sweep_pair(1, 0, 15, 12);  // (·, 0,-1): ln 5 / ln 4
                    sweep_pair(0, 1, 9, 6);    // (·,-1, 0): ln 3 / ln 2
                    sweep_pair(1, 1, 3, 0);    // (·, 0, 0): ln 1 / ln 0

                    #if defined(_MSC_VER)
                    #pragma loop(ivdep)
                    #elif defined(__clang__)
                    #pragma clang loop vectorize(enable) interleave(enable)
                    #elif defined(__GNUC__) && !defined(__NVCOMPILER)
                    #pragma GCC ivdep
                    #endif
                    for (SIndex_t i = 0;
                         i < static_cast<SIndex_t>(NB_DOFS * nnx); ++i) {
                        if constexpr (Increment) {
                            f_row[i] += alpha * facc[i];
                        } else {
                            f_row[i] = alpha * facc[i];
                        }
                    }
                }
            }
        }

        // Shared dispatcher for the per-pixel (Uniform == false) and the
        // spatially-uniform (Uniform == true) stiffness apply: resolves the
        // runtime `increment` flag to a compile-time branch and asserts the
        // compile-time AoS x-strides the entry points hard-code; only the
        // row/plane pitches reach the kernel.
        template <typename T, bool Uniform>
        static void isotropic_stiffness_3d_host_tmpl(
            const T * MUGRID_RESTRICT displacement,
            const T * MUGRID_RESTRICT lambda,
            const T * MUGRID_RESTRICT mu, T * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nnz, Index_t disp_stride_x,
            Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
            Index_t force_stride_x, Index_t force_stride_y,
            Index_t force_stride_z, Index_t force_stride_d, const T * G,
            const T * V, T alpha, bool increment, T lam_u,
            T mu_u) {
            [[maybe_unused]] constexpr Index_t NB_DOFS = 3;
            assert(disp_stride_d == 1 && force_stride_d == 1);
            assert(disp_stride_x == NB_DOFS && force_stride_x == NB_DOFS);
            assert(Uniform || mat_stride_x == 1);
            (void)disp_stride_x;
            (void)disp_stride_d;
            (void)mat_stride_x;
            (void)force_stride_x;
            (void)force_stride_d;
            if (increment) {
                isotropic_stiffness_3d_row_kernel<T, Uniform, true>(
                    displacement, lambda, mu, force, nnx, nny, nnz,
                    disp_stride_y, disp_stride_z, mat_stride_y, mat_stride_z,
                    force_stride_y, force_stride_z, G, V, alpha, lam_u, mu_u);
            } else {
                isotropic_stiffness_3d_row_kernel<T, Uniform, false>(
                    displacement, lambda, mu, force, nnx, nny, nnz,
                    disp_stride_y, disp_stride_z, mat_stride_y, mat_stride_z,
                    force_stride_y, force_stride_z, G, V, alpha, lam_u, mu_u);
            }
        }

        // Per-pixel material entry point (unchanged public signature).
        template <typename T>
        void isotropic_stiffness_3d_host(
            const T * MUGRID_RESTRICT displacement,
            const T * MUGRID_RESTRICT lambda,
            const T * MUGRID_RESTRICT mu, T * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nnz, Index_t /*nelx*/,
            Index_t /*nely*/, Index_t /*nelz*/, Index_t disp_stride_x,
            Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
            Index_t force_stride_x, Index_t force_stride_y,
            Index_t force_stride_z, Index_t force_stride_d, const T * G,
            const T * V, T alpha, bool increment) {
            isotropic_stiffness_3d_host_tmpl<T, false>(
                displacement, lambda, mu, force, nnx, nny, nnz, disp_stride_x,
                disp_stride_y, disp_stride_z, disp_stride_d, mat_stride_x,
                mat_stride_y, mat_stride_z, force_stride_x, force_stride_y,
                force_stride_z, force_stride_d, G, V, alpha, increment, T(0),
                T(0));
        }

        // Uniform Lamé scalars: no material pointers, no material strides.
        template <typename T>
        void isotropic_stiffness_3d_host_uniform(
            const T * MUGRID_RESTRICT displacement, T lambda, T mu,
            T * MUGRID_RESTRICT force, Index_t nnx, Index_t nny, Index_t nnz,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_z,
            Index_t disp_stride_d, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_z,
            Index_t force_stride_d, const T * G, const T * V, T alpha,
            bool increment) {
            isotropic_stiffness_3d_host_tmpl<T, true>(
                displacement, nullptr, nullptr, force, nnx, nny, nnz,
                disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d, 0,
                0, 0, force_stride_x, force_stride_y, force_stride_z,
                force_stride_d, G, V, alpha, increment, lambda, mu);
        }

        // Macro RHS: assemble force = Bᵀ C E_macro = K @ u* with the affine
        // u* folded into the constant per-element vectors Gu = G u*, Vu = V u*.
        template <typename T>
        void isotropic_stiffness_3d_host_macro_rhs(
            const T * MUGRID_RESTRICT lambda, const T * MUGRID_RESTRICT mu,
            T * MUGRID_RESTRICT force, Index_t nnx, Index_t nny, Index_t nnz,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
            Index_t force_stride_x, Index_t force_stride_y,
            Index_t force_stride_z, Index_t force_stride_d, const T * Gu,
            const T * Vu, T alpha, bool increment) {

            constexpr Index_t NB_DOFS = 3;
            using SIndex_t = std::ptrdiff_t;
            SIndex_t s_mat_stride_x = static_cast<SIndex_t>(mat_stride_x);
            SIndex_t s_mat_stride_y = static_cast<SIndex_t>(mat_stride_y);
            SIndex_t s_mat_stride_z = static_cast<SIndex_t>(mat_stride_z);
            SIndex_t s_force_stride_x = static_cast<SIndex_t>(force_stride_x);
            SIndex_t s_force_stride_y = static_cast<SIndex_t>(force_stride_y);
            SIndex_t s_force_stride_z = static_cast<SIndex_t>(force_stride_z);
            SIndex_t s_force_stride_d = static_cast<SIndex_t>(force_stride_d);

            static const SIndex_t ELEM_OFFSETS[8][4] = {
                {-1, -1, -1, 7}, {0, -1, -1, 6}, {-1, 0, -1, 5},
                {0, 0, -1, 4},   {-1, -1, 0, 3}, {0, -1, 0, 2},
                {-1, 0, 0, 1},   {0, 0, 0, 0}};

            for (SIndex_t iz = 0; iz < static_cast<SIndex_t>(nnz); ++iz) {
                for (SIndex_t iy = 0; iy < static_cast<SIndex_t>(nny); ++iy) {
                    for (SIndex_t ix = 0; ix < static_cast<SIndex_t>(nnx);
                         ++ix) {
                        T f[NB_DOFS] = {0, 0, 0};
                        for (Index_t elem = 0; elem < 8; ++elem) {
                            SIndex_t ex = ix + ELEM_OFFSETS[elem][0];
                            SIndex_t ey = iy + ELEM_OFFSETS[elem][1];
                            SIndex_t ez = iz + ELEM_OFFSETS[elem][2];
                            Index_t local_node = ELEM_OFFSETS[elem][3];
                            SIndex_t mat_idx = ex * s_mat_stride_x +
                                               ey * s_mat_stride_y +
                                               ez * s_mat_stride_z;
                            T lam = lambda[mat_idx];
                            T mu_val = mu[mat_idx];
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                Index_t row = local_node * NB_DOFS + d;
                                f[d] += static_cast<T>(2) * mu_val * Gu[row] +
                                        lam * Vu[row];
                            }
                        }
                        SIndex_t base = ix * s_force_stride_x +
                                        iy * s_force_stride_y +
                                        iz * s_force_stride_z;
                        if (increment) {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * s_force_stride_d] +=
                                    alpha * f[d];
                            }
                        } else {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * s_force_stride_d] =
                                    alpha * f[d];
                            }
                        }
                    }
                }
            }
        }

        // Stress average: loop over local elements, compute the element-
        // averaged strain ḡ = Dbar u, form σ = C(λ,μ):(E_macro + sym ḡ), and
        // accumulate the local volume integral vol_elem Σ_e σ_e into accum_out.
        template <typename T>
        void isotropic_stiffness_3d_host_average(
            const T * MUGRID_RESTRICT displacement,
            const T * MUGRID_RESTRICT lambda, const T * MUGRID_RESTRICT mu,
            Index_t nelx, Index_t nely, Index_t nelz, Index_t disp_stride_x,
            Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t mat_stride_z,
            const T * Dbar, const T * E_macro, Real vol_elem,
            Real * accum_out) {

            constexpr Index_t NB_NODES = 8;
            constexpr Index_t NB_DOFS = 3;
            constexpr Index_t DIM = 3;
            using SIndex_t = std::ptrdiff_t;
            SIndex_t s_disp_stride_x = static_cast<SIndex_t>(disp_stride_x);
            SIndex_t s_disp_stride_y = static_cast<SIndex_t>(disp_stride_y);
            SIndex_t s_disp_stride_z = static_cast<SIndex_t>(disp_stride_z);
            SIndex_t s_disp_stride_d = static_cast<SIndex_t>(disp_stride_d);
            SIndex_t s_mat_stride_x = static_cast<SIndex_t>(mat_stride_x);
            SIndex_t s_mat_stride_y = static_cast<SIndex_t>(mat_stride_y);
            SIndex_t s_mat_stride_z = static_cast<SIndex_t>(mat_stride_z);

            static const SIndex_t NODE_OFFSET[8][3] = {
                {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

            // Per-element stress in working precision T; reduction in double.
            Real acc[DIM * DIM] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            for (SIndex_t ez = 0; ez < static_cast<SIndex_t>(nelz); ++ez) {
                for (SIndex_t ey = 0; ey < static_cast<SIndex_t>(nely); ++ey) {
                    for (SIndex_t ex = 0; ex < static_cast<SIndex_t>(nelx);
                         ++ex) {
                        T u[NB_NODES * NB_DOFS];
                        for (Index_t node = 0; node < NB_NODES; ++node) {
                            SIndex_t nx_pos = ex + NODE_OFFSET[node][0];
                            SIndex_t ny_pos = ey + NODE_OFFSET[node][1];
                            SIndex_t nz_pos = ez + NODE_OFFSET[node][2];
                            SIndex_t disp_idx = nx_pos * s_disp_stride_x +
                                                ny_pos * s_disp_stride_y +
                                                nz_pos * s_disp_stride_z;
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                u[node * NB_DOFS + d] =
                                    displacement[disp_idx + d * s_disp_stride_d];
                            }
                        }
                        SIndex_t mat_idx = ex * s_mat_stride_x +
                                           ey * s_mat_stride_y +
                                           ez * s_mat_stride_z;
                        T lam = lambda[mat_idx];
                        T mu_val = mu[mat_idx];

                        T g[DIM][DIM];
                        for (Index_t i = 0; i < DIM; ++i) {
                            for (Index_t j = 0; j < DIM; ++j) {
                                T s = 0;
                                for (Index_t n = 0; n < NB_NODES; ++n) {
                                    s += Dbar[j * NB_NODES + n] *
                                         u[n * NB_DOFS + i];
                                }
                                g[i][j] = s;
                            }
                        }
                        T E[DIM][DIM];
                        for (Index_t i = 0; i < DIM; ++i) {
                            for (Index_t j = 0; j < DIM; ++j) {
                                E[i][j] = E_macro[i * DIM + j] +
                                          static_cast<T>(0.5) *
                                              (g[i][j] + g[j][i]);
                            }
                        }
                        T trE = E[0][0] + E[1][1] + E[2][2];
                        for (Index_t i = 0; i < DIM; ++i) {
                            for (Index_t j = 0; j < DIM; ++j) {
                                T sig = static_cast<T>(2) * mu_val * E[i][j] +
                                        (i == j ? lam * trE : T(0));
                                acc[i * DIM + j] += static_cast<Real>(sig);
                            }
                        }
                    }
                }
            }
            for (Index_t k = 0; k < DIM * DIM; ++k) {
                accum_out[k] = acc[k] * vol_elem;
            }
        }

        // Sensitivity contraction (see the 2D counterpart): per owned element,
        // a = u* + gather(fwd), b = u* + gather(cos), write g_shear = aᵀ G b,
        // g_vol = aᵀ V b to the per-pixel outputs.
        template <typename T>
        void isotropic_stiffness_3d_host_sensitivity(
            const T * MUGRID_RESTRICT fwd, const T * MUGRID_RESTRICT cos,
            T * MUGRID_RESTRICT g_shear, T * MUGRID_RESTRICT g_vol,
            Index_t nelx, Index_t nely, Index_t nelz, Index_t disp_stride_x,
            Index_t disp_stride_y, Index_t disp_stride_z, Index_t disp_stride_d,
            Index_t out_stride_x, Index_t out_stride_y, Index_t out_stride_z,
            const T * G, const T * V, const T * ustar_f, const T * ustar_c) {

            constexpr Index_t NB_NODES = 8;
            constexpr Index_t NB_DOFS = 3;
            constexpr Index_t NB_ELEM_DOFS = NB_NODES * NB_DOFS;
            using SIndex_t = std::ptrdiff_t;
            SIndex_t s_disp_stride_x = static_cast<SIndex_t>(disp_stride_x);
            SIndex_t s_disp_stride_y = static_cast<SIndex_t>(disp_stride_y);
            SIndex_t s_disp_stride_z = static_cast<SIndex_t>(disp_stride_z);
            SIndex_t s_disp_stride_d = static_cast<SIndex_t>(disp_stride_d);
            SIndex_t s_out_stride_x = static_cast<SIndex_t>(out_stride_x);
            SIndex_t s_out_stride_y = static_cast<SIndex_t>(out_stride_y);
            SIndex_t s_out_stride_z = static_cast<SIndex_t>(out_stride_z);

            static const SIndex_t NODE_OFFSET[8][3] = {
                {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

            for (SIndex_t ez = 0; ez < static_cast<SIndex_t>(nelz); ++ez) {
                for (SIndex_t ey = 0; ey < static_cast<SIndex_t>(nely); ++ey) {
                    for (SIndex_t ex = 0; ex < static_cast<SIndex_t>(nelx);
                         ++ex) {
                        T a[NB_ELEM_DOFS], b[NB_ELEM_DOFS];
                        for (Index_t node = 0; node < NB_NODES; ++node) {
                            SIndex_t nx_pos = ex + NODE_OFFSET[node][0];
                            SIndex_t ny_pos = ey + NODE_OFFSET[node][1];
                            SIndex_t nz_pos = ez + NODE_OFFSET[node][2];
                            SIndex_t disp_idx = nx_pos * s_disp_stride_x +
                                                ny_pos * s_disp_stride_y +
                                                nz_pos * s_disp_stride_z;
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                Index_t r = node * NB_DOFS + d;
                                SIndex_t k = disp_idx + d * s_disp_stride_d;
                                a[r] = ustar_f[r] + fwd[k];
                                b[r] = ustar_c[r] + cos[k];
                            }
                        }
                        T gs = 0, gv = 0;
                        for (Index_t r = 0; r < NB_ELEM_DOFS; ++r) {
                            T Gr = 0, Vr = 0;
                            for (Index_t c = 0; c < NB_ELEM_DOFS; ++c) {
                                Gr += G[r * NB_ELEM_DOFS + c] * b[c];
                                Vr += V[r * NB_ELEM_DOFS + c] * b[c];
                            }
                            gs += a[r] * Gr;
                            gv += a[r] * Vr;
                        }
                        SIndex_t out_idx = ex * s_out_stride_x +
                                           ey * s_out_stride_y +
                                           ez * s_out_stride_z;
                        g_shear[out_idx] = gs;
                        g_vol[out_idx] = gv;
                    }
                }
            }
        }

    }  // namespace isotropic_stiffness_kernels

    // Explicit instantiations of the per-precision host impls.
#define MUGRID_INSTANTIATE_STIFFNESS_3D(T)                                     \
    template void IsotropicStiffnessOperator<3>::apply_impl<T>(                \
        const TypedFieldBase<T> &, const TypedFieldBase<T> &,                  \
        const TypedFieldBase<T> &, T, TypedFieldBase<T> &, bool) const;        \
    template void IsotropicStiffnessOperator<3>::apply_uniform_impl<T>(        \
        const TypedFieldBase<T> &, T, T, T, TypedFieldBase<T> &, bool) const;  \
    template void IsotropicStiffnessOperator<3>::apply_macro_rhs_impl<T>(      \
        const TypedFieldBase<T> &, const TypedFieldBase<T> &,                  \
        const std::array<Real, 9> &, TypedFieldBase<T> &) const;               \
    template std::array<Real, 9>                                              \
    IsotropicStiffnessOperator<3>::average_stress_impl<T>(                     \
        const TypedFieldBase<T> &, const TypedFieldBase<T> &,                  \
        const TypedFieldBase<T> &, const std::array<Real, 9> &) const;         \
    template void IsotropicStiffnessOperator<3>::assemble_diagonal_impl<T>(    \
        const TypedFieldBase<T> &, const TypedFieldBase<T> &,                  \
        TypedFieldBase<T> &) const;                                            \
    template void IsotropicStiffnessOperator<3>::compute_sensitivity_impl<T>(  \
        const TypedFieldBase<T> &, const std::array<Real, 9> &,                \
        const TypedFieldBase<T> &, const std::array<Real, 9> &,                \
        TypedFieldBase<T> &, TypedFieldBase<T> &) const;
    MUGRID_INSTANTIATE_STIFFNESS_3D(Real)
    MUGRID_INSTANTIATE_STIFFNESS_3D(Real32)
#undef MUGRID_INSTANTIATE_STIFFNESS_3D

}  // namespace muGrid
