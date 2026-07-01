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
        static const Index_t NODE_OFFSET_3D[8][3] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        // Shared kernel body for the per-pixel (Uniform == false) and the
        // spatially-uniform (Uniform == true) stiffness apply. `if constexpr`
        // picks the material source at compile time, so the heterogeneous path
        // is byte-for-byte the original kernel and the uniform path reads λ, μ
        // from registers (lambda/mu/mat strides are unused there).
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

            constexpr Index_t NB_NODES = 8;
            constexpr Index_t NB_DOFS = 3;
            constexpr Index_t NB_ELEM_DOFS = NB_NODES * NB_DOFS;

            // Use signed versions of strides to avoid unsigned overflow
            // when accessing ghost cells at negative indices
            using SIndex_t = std::ptrdiff_t;
            SIndex_t s_disp_stride_x = static_cast<SIndex_t>(disp_stride_x);
            SIndex_t s_disp_stride_y = static_cast<SIndex_t>(disp_stride_y);
            SIndex_t s_disp_stride_z = static_cast<SIndex_t>(disp_stride_z);
            SIndex_t s_disp_stride_d = static_cast<SIndex_t>(disp_stride_d);
            [[maybe_unused]] SIndex_t s_mat_stride_x =
                static_cast<SIndex_t>(mat_stride_x);
            [[maybe_unused]] SIndex_t s_mat_stride_y =
                static_cast<SIndex_t>(mat_stride_y);
            [[maybe_unused]] SIndex_t s_mat_stride_z =
                static_cast<SIndex_t>(mat_stride_z);
            SIndex_t s_force_stride_x = static_cast<SIndex_t>(force_stride_x);
            SIndex_t s_force_stride_y = static_cast<SIndex_t>(force_stride_y);
            SIndex_t s_force_stride_z = static_cast<SIndex_t>(force_stride_z);
            SIndex_t s_force_stride_d = static_cast<SIndex_t>(force_stride_d);

            // Neighboring element offsets and corresponding local node index
            // Element at (ix + eox, iy + eoy, iz + eoz) has this node as local node
            static const SIndex_t ELEM_OFFSETS[8][4] = {
                {-1, -1, -1, 7},  // Element (ix-1, iy-1, iz-1): local node 7 (corner 1,1,1)
                { 0, -1, -1, 6},  // Element (ix,   iy-1, iz-1): local node 6 (corner 0,1,1)
                {-1,  0, -1, 5},  // Element (ix-1, iy,   iz-1): local node 5 (corner 1,0,1)
                { 0,  0, -1, 4},  // Element (ix,   iy,   iz-1): local node 4 (corner 0,0,1)
                {-1, -1,  0, 3},  // Element (ix-1, iy-1, iz  ): local node 3 (corner 1,1,0)
                { 0, -1,  0, 2},  // Element (ix,   iy-1, iz  ): local node 2 (corner 0,1,0)
                {-1,  0,  0, 1},  // Element (ix-1, iy,   iz  ): local node 1 (corner 1,0,0)
                { 0,  0,  0, 0}   // Element (ix,   iy,   iz  ): local node 0 (corner 0,0,0)
            };

            // Node offsets within element [node][dim]
            static const SIndex_t NODE_OFFSET[8][3] = {
                {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

            // Iteration bounds: iterate over all computable nodes based on stencil
            // requirements. The computable region is determined by where the stencil
            // has valid input data, not by the ghost region size. This allows
            // computing in ghost regions beyond the minimum stencil requirement.
            SIndex_t ix_start = 0;
            SIndex_t iy_start = 0;
            SIndex_t iz_start = 0;
            SIndex_t ix_end = static_cast<SIndex_t>(nnx);
            SIndex_t iy_end = static_cast<SIndex_t>(nny);
            SIndex_t iz_end = static_cast<SIndex_t>(nnz);

            // Gather pattern: loop over all computable nodes, gather from neighboring
            // elements. Ghost cells handle periodicity and MPI boundaries.
            for (SIndex_t iz = iz_start; iz < iz_end; ++iz) {
                for (SIndex_t iy = iy_start; iy < iy_end; ++iy) {
                    for (SIndex_t ix = ix_start; ix < ix_end; ++ix) {
                        // Accumulate force for this node
                        T f[NB_DOFS] = {0, 0, 0};

                        // Loop over neighboring elements (all 8 elements guaranteed
                        // to exist for nodes in this iteration range)
                        for (Index_t elem = 0; elem < 8; ++elem) {
                            // Element indices (can be -1 for periodic BC accessing ghost cells)
                            SIndex_t ex = ix + ELEM_OFFSETS[elem][0];
                            SIndex_t ey = iy + ELEM_OFFSETS[elem][1];
                            SIndex_t ez = iz + ELEM_OFFSETS[elem][2];
                            Index_t local_node = ELEM_OFFSETS[elem][3];

                            // Get material parameters
                            T lam, mu_val;
                            if constexpr (Uniform) {
                                lam = lam_u;
                                mu_val = mu_u;
                            } else {
                                SIndex_t mat_idx = ex * s_mat_stride_x +
                                                   ey * s_mat_stride_y +
                                                   ez * s_mat_stride_z;
                                lam = lambda[mat_idx];
                                mu_val = mu[mat_idx];
                            }

                            // Gather displacements from all 8 nodes of this element
                            T u[NB_ELEM_DOFS];
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

                            // Compute only the rows that correspond to this node
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                Index_t row = local_node * NB_DOFS + d;
                                T contrib = 0;
                                for (Index_t j = 0; j < NB_ELEM_DOFS; ++j) {
                                    contrib +=
                                        (static_cast<T>(2) * mu_val *
                                             G[row * NB_ELEM_DOFS + j] +
                                         lam * V[row * NB_ELEM_DOFS + j]) *
                                        u[j];
                                }
                                f[d] += contrib;
                            }
                        }

                        // Write force for this node
                        SIndex_t base = ix * s_force_stride_x +
                                        iy * s_force_stride_y +
                                        iz * s_force_stride_z;
                        if (increment) {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * s_force_stride_d] += alpha * f[d];
                            }
                        } else {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * s_force_stride_d] = alpha * f[d];
                            }
                        }
                    }
                }
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
        const TypedFieldBase<T> &, const std::array<Real, 9> &) const;
    MUGRID_INSTANTIATE_STIFFNESS_3D(Real)
    MUGRID_INSTANTIATE_STIFFNESS_3D(Real32)
#undef MUGRID_INSTANTIATE_STIFFNESS_3D

}  // namespace muGrid
