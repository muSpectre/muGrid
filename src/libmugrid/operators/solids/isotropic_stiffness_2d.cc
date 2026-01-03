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

#include "isotropic_stiffness_2d.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

namespace muGrid {

    // ============================================================================
    // 2D Implementation
    // ============================================================================

    // 2D shape function gradients for 2 triangles per pixel
    // B_2D[quad][dim][node] - gradient of shape function for node w.r.t. dim
    // Nodes: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    // Triangle 0: nodes 0,1,2 (lower-left)
    // Triangle 1: nodes 1,2,3 (upper-right)
    static const Real B_2D[2][2][4] = {
        // Triangle 0 (lower-left)
        {{-1.0, 1.0, 0.0, 0.0},   // d/dx
         {-1.0, 0.0, 1.0, 0.0}},  // d/dy
        // Triangle 1 (upper-right)
        {{0.0, 0.0, -1.0, 1.0},  // d/dx
         {0.0, -1.0, 0.0, 1.0}}  // d/dy
    };

    // Quadrature weights for 2D (area of each triangle / pixel area)
    static const Real W_2D[2] = {0.5, 0.5};

    IsotropicStiffnessOperator2D::IsotropicStiffnessOperator2D(
        const std::vector<Real> & grid_spacing)
        : grid_spacing{grid_spacing} {
        if (grid_spacing.size() != 2) {
            throw RuntimeError("2D operator requires 2D grid spacing");
        }
        this->precompute_matrices();
    }

    void IsotropicStiffnessOperator2D::precompute_matrices() {
        // Initialize matrices to zero
        std::fill(G_matrix.begin(), G_matrix.end(), 0.0);
        std::fill(V_matrix.begin(), V_matrix.end(), 0.0);

        const Real hx = grid_spacing[0];
        const Real hy = grid_spacing[1];
        const Real inv_hx = 1.0 / hx;
        const Real inv_hy = 1.0 / hy;

        // Scale factors for each derivative direction
        const Real scale[2] = {inv_hx, inv_hy};

        // Compute G = Σ_q w_q B_q^T I B_q
        // and V = Σ_q w_q (B_q^T m)(m^T B_q) where m = [1,1,0]^T for 2D Voigt
        for (Index_t q = 0; q < NB_QUAD; ++q) {
            Real w = W_2D[q] * hx * hy;  // Weight includes element area

            // For each pair of nodes (I, J) and DOFs (a, b)
            // G[I*2+a, J*2+b] += w * Σ_d B[q,d,I] * scale[d] * B[q,d,J] *
            // scale[d] * δ_ab Plus cross terms for shear
            for (Index_t I = 0; I < NB_NODES; ++I) {
                for (Index_t J = 0; J < NB_NODES; ++J) {
                    // G contribution: B^T I B where I is identity in strain
                    // space For 2D: strain = [εxx, εyy, 2εxy] B maps
                    // displacement to strain

                    // Compute B^T B for this node pair
                    // B is 3×8 (strain × DOFs), B^T B is 8×8
                    for (Index_t a = 0; a < NB_DOFS_PER_NODE; ++a) {
                        for (Index_t b = 0; b < NB_DOFS_PER_NODE; ++b) {
                            Real g_contrib = 0.0;
                            Real v_contrib = 0.0;

                            // Diagonal strain terms (εxx, εyy)
                            // B[εxx, u_x] = dN/dx, B[εyy, u_y] = dN/dy
                            if (a == b) {
                                // Normal strain contribution
                                g_contrib += B_2D[q][a][I] * scale[a] *
                                             B_2D[q][a][J] * scale[a];
                            }

                            // Shear strain term (2εxy = du_y/dx + du_x/dy)
                            // B[2εxy, u_x] = dN/dy, B[2εxy, u_y] = dN/dx
                            // Contributes to G but with factor 0.5 due to
                            // engineering strain
                            if (a == 0 && b == 0) {
                                // u_x - u_x coupling via shear
                                g_contrib += 0.5 * B_2D[q][1][I] * scale[1] *
                                             B_2D[q][1][J] * scale[1];
                            } else if (a == 1 && b == 1) {
                                // u_y - u_y coupling via shear
                                g_contrib += 0.5 * B_2D[q][0][I] * scale[0] *
                                             B_2D[q][0][J] * scale[0];
                            } else if (a == 0 && b == 1) {
                                // u_x - u_y coupling via shear
                                g_contrib += 0.5 * B_2D[q][1][I] * scale[1] *
                                             B_2D[q][0][J] * scale[0];
                            } else if (a == 1 && b == 0) {
                                // u_y - u_x coupling via shear
                                g_contrib += 0.5 * B_2D[q][0][I] * scale[0] *
                                             B_2D[q][1][J] * scale[1];
                            }

                            // V contribution: (B^T m)(m^T B)
                            // m = [1, 1, 0]^T selects trace(ε) = εxx + εyy
                            // B^T m gives displacement contributions to trace
                            // For node I, DOF a: (B^T m)[I,a] = δ_{a,0}*dN_I/dx
                            // + δ_{a,1}*dN_I/dy
                            Real BtmI =
                                (a == 0 ? B_2D[q][0][I] * scale[0] : 0.0) +
                                (a == 1 ? B_2D[q][1][I] * scale[1] : 0.0);
                            Real BtmJ =
                                (b == 0 ? B_2D[q][0][J] * scale[0] : 0.0) +
                                (b == 1 ? B_2D[q][1][J] * scale[1] : 0.0);
                            v_contrib = BtmI * BtmJ;

                            Index_t idx =
                                (I * NB_DOFS_PER_NODE + a) * NB_ELEMENT_DOFS +
                                (J * NB_DOFS_PER_NODE + b);
                            G_matrix[idx] += w * g_contrib;
                            V_matrix[idx] += w * v_contrib;
                        }
                    }
                }
            }
        }
    }

    void IsotropicStiffnessOperator2D::apply(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        TypedFieldBase<Real> & force) const {
        apply_impl(displacement, lambda, mu, 1.0, force, false);
    }

    void IsotropicStiffnessOperator2D::apply_increment(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        Real alpha, TypedFieldBase<Real> & force) const {
        apply_impl(displacement, lambda, mu, alpha, force, true);
    }

    void IsotropicStiffnessOperator2D::apply_impl(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        Real alpha, TypedFieldBase<Real> & force, bool increment) const {

        // Get field collections and validate
        auto & disp_coll = displacement.get_collection();
        auto * disp_global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&disp_coll);
        if (!disp_global_fc) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D requires GlobalFieldCollection");
        }

        // Get dimensions from material field collection (defines number of
        // elements)
        auto & mat_coll = lambda.get_collection();
        auto * mat_global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&mat_coll);
        if (!mat_global_fc) {
            throw RuntimeError("IsotropicStiffnessOperator2D material fields "
                               "require GlobalFieldCollection");
        }

        // Validate ghost configuration for displacement/force fields
        auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
        auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
        if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator2D requires at least "
                               "1 ghost cell on the right side of "
                               "displacement/force fields");
        }

        // Material field dimensions = number of elements (interior, without
        // ghosts)
        auto nb_elements =
            mat_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
        Index_t nelx = nb_elements[0];
        Index_t nely = nb_elements[1];

        // Get number of interior nodes
        auto nb_interior =
            disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
        Index_t nnx = nb_interior[0];
        Index_t nny = nb_interior[1];

        // Determine periodic/non-periodic based on material vs node dimensions
        // Periodic: nelx == nnx (one element wraps around)
        // Non-periodic: nelx == nnx - 1 (boundary nodes lack some neighbors)
        bool periodic_x = (nelx == nnx);
        bool periodic_y = (nely == nny);
        bool non_periodic_x = (nelx == nnx - 1);
        bool non_periodic_y = (nely == nny - 1);

        // Validate material field dimensions
        if (!periodic_x && !non_periodic_x) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D: material field x-dimension (" +
                std::to_string(nelx) + ") must be either nnx (" +
                std::to_string(nnx) + ") for periodic or nnx-1 (" +
                std::to_string(nnx - 1) + ") for non-periodic");
        }
        if (!periodic_y && !non_periodic_y) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D: material field y-dimension (" +
                std::to_string(nely) + ") must be either nny (" +
                std::to_string(nny) + ") for periodic or nny-1 (" +
                std::to_string(nny - 1) + ") for non-periodic");
        }

        bool periodic = periodic_x && periodic_y;

        // Validate ghost configuration based on periodic/non-periodic mode
        if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1)) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D with periodic BC requires at "
                "least 1 ghost cell on the left side of displacement/force "
                "fields");
        }

        // Get material field ghost configuration
        auto mat_nb_ghosts_left = mat_global_fc->get_nb_ghosts_left();

        // For periodic case, material field needs ghost cells too
        if (periodic && (mat_nb_ghosts_left[0] < 1 || mat_nb_ghosts_left[1] < 1)) {
            throw RuntimeError("IsotropicStiffnessOperator2D with periodic BC "
                               "requires at least 1 ghost cell on the left "
                               "side of material fields (lambda, mu)");
        }

        // Node dimensions (for displacement/force fields with ghosts)
        auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];

        // Ghost offsets (interior starts at this offset in the ghosted array)
        Index_t disp_ghost_offset_x = nb_ghosts_left[0];
        Index_t disp_ghost_offset_y = nb_ghosts_left[1];
        Index_t mat_ghost_offset_x = mat_nb_ghosts_left[0];
        Index_t mat_ghost_offset_y = mat_nb_ghosts_left[1];

        // Compute strides (AoS layout for host)
        Index_t disp_stride_d = 1;
        Index_t disp_stride_x = NB_DOFS_PER_NODE;
        Index_t disp_stride_y = NB_DOFS_PER_NODE * nx;

        // Material field strides (with ghosts if present)
        auto mat_nb_pts_with_ghosts =
            mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t mat_nx = mat_nb_pts_with_ghosts[0];
        Index_t mat_stride_x = 1;
        Index_t mat_stride_y = mat_nx;

        Index_t force_stride_d = 1;
        Index_t force_stride_x = NB_DOFS_PER_NODE;
        Index_t force_stride_y = NB_DOFS_PER_NODE * nx;

        // Offset data pointers to account for left ghosts
        // Interior data starts at position (ghost_offset_x, ghost_offset_y)
        Index_t disp_offset = disp_ghost_offset_x * disp_stride_x +
                              disp_ghost_offset_y * disp_stride_y;
        Index_t force_offset = disp_ghost_offset_x * force_stride_x +
                               disp_ghost_offset_y * force_stride_y;
        Index_t mat_offset = mat_ghost_offset_x * mat_stride_x +
                             mat_ghost_offset_y * mat_stride_y;

        const Real * disp_data = displacement.data() + disp_offset;
        const Real * lambda_data = lambda.data() + mat_offset;
        const Real * mu_data = mu.data() + mat_offset;
        Real * force_data = force.data() + force_offset;

        isotropic_stiffness_kernels::isotropic_stiffness_2d_host(
            disp_data, lambda_data, mu_data, force_data, nnx,
            nny,         // Number of interior nodes
            nelx, nely,  // Number of elements
            disp_stride_x, disp_stride_y, disp_stride_d, mat_stride_x,
            mat_stride_y, force_stride_x, force_stride_y, force_stride_d,
            G_matrix.data(), V_matrix.data(), alpha, increment);
    }

    // ============================================================================
    // Host Kernel Implementations
    // ============================================================================

    namespace isotropic_stiffness_kernels {

        // Node offsets for 2D [node][dim]
        static const Index_t NODE_OFFSET_2D[4][2] = {
            {0, 0}, {1, 0}, {0, 1}, {1, 1}};

        void isotropic_stiffness_2d_host(
            const Real * MUGRID_RESTRICT displacement,
            const Real * MUGRID_RESTRICT lambda,
            const Real * MUGRID_RESTRICT mu, Real * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nelx, Index_t nely,
            Index_t disp_stride_x, Index_t disp_stride_y, Index_t disp_stride_d,
            Index_t mat_stride_x, Index_t mat_stride_y, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_d, const Real * G,
            const Real * V, Real alpha, bool increment) {

            constexpr Index_t NB_NODES = 4;
            constexpr Index_t NB_DOFS = 2;
            constexpr Index_t NB_ELEM_DOFS = NB_NODES * NB_DOFS;

            // Use signed versions of strides to avoid unsigned overflow
            // when accessing ghost cells at negative indices
            using SIndex_t = std::ptrdiff_t;
            SIndex_t s_disp_stride_x = static_cast<SIndex_t>(disp_stride_x);
            SIndex_t s_disp_stride_y = static_cast<SIndex_t>(disp_stride_y);
            SIndex_t s_disp_stride_d = static_cast<SIndex_t>(disp_stride_d);
            SIndex_t s_mat_stride_x = static_cast<SIndex_t>(mat_stride_x);
            SIndex_t s_mat_stride_y = static_cast<SIndex_t>(mat_stride_y);
            SIndex_t s_force_stride_x = static_cast<SIndex_t>(force_stride_x);
            SIndex_t s_force_stride_y = static_cast<SIndex_t>(force_stride_y);
            SIndex_t s_force_stride_d = static_cast<SIndex_t>(force_stride_d);

            // Neighboring element offsets and corresponding local node index
            // Element at (ix + eox, iy + eoy) has this node as local node
            // `local_node`. Offsets are signed because they can be -1.
            static const SIndex_t ELEM_OFFSETS[4][3] = {
                {-1, -1, 3},  // Element (ix-1, iy-1): this node is local node 3
                { 0, -1, 2},  // Element (ix,   iy-1): this node is local node 2
                {-1,  0, 1},  // Element (ix-1, iy  ): this node is local node 1
                { 0,  0, 0}   // Element (ix,   iy  ): this node is local node 0
            };

            // Node offsets within element [node][dim]
            static const SIndex_t NODE_OFFSET[4][2] = {
                {0, 0}, {1, 0}, {0, 1}, {1, 1}};

            // Compute iteration bounds
            // For periodic BC (nelx == nnx): iterate over all nodes, all elements valid
            // For non-periodic BC (nelx == nnx-1): boundary node forces are irrelevant
            //   (overwritten by ghost communication), so skip bounds checking
            SIndex_t ix_start = (nelx == nnx) ? 0 : 1;
            SIndex_t iy_start = (nely == nny) ? 0 : 1;
            SIndex_t ix_end = (nelx == nnx) ? static_cast<SIndex_t>(nnx) : static_cast<SIndex_t>(nnx) - 1;
            SIndex_t iy_end = (nely == nny) ? static_cast<SIndex_t>(nny) : static_cast<SIndex_t>(nny) - 1;

            // Gather pattern: loop over interior NODES, gather from neighboring
            // elements. Ghost cells handle periodicity and MPI boundaries.
            // For non-periodic BC, boundary nodes are skipped (their forces are
            // overwritten by ghost communication anyway).
            for (SIndex_t iy = iy_start; iy < iy_end; ++iy) {
                for (SIndex_t ix = ix_start; ix < ix_end; ++ix) {
                    // Accumulate force for this node
                    Real f[NB_DOFS] = {0.0, 0.0};

                    // Loop over neighboring elements (all 4 elements guaranteed
                    // to exist for nodes in this iteration range)
                    for (Index_t elem = 0; elem < 4; ++elem) {
                        // Element indices (can be -1 for periodic BC accessing ghost cells)
                        SIndex_t ex = ix + ELEM_OFFSETS[elem][0];
                        SIndex_t ey = iy + ELEM_OFFSETS[elem][1];
                        Index_t local_node = ELEM_OFFSETS[elem][2];

                        // Get material parameters for this element
                        SIndex_t mat_idx = ex * s_mat_stride_x + ey * s_mat_stride_y;
                        Real lam = lambda[mat_idx];
                        Real mu_val = mu[mat_idx];

                        // Gather displacements from all 4 nodes of this element
                        Real u[NB_ELEM_DOFS];
                        for (Index_t node = 0; node < NB_NODES; ++node) {
                            SIndex_t nx_pos = ex + NODE_OFFSET[node][0];
                            SIndex_t ny_pos = ey + NODE_OFFSET[node][1];
                            SIndex_t disp_idx = nx_pos * s_disp_stride_x +
                                                ny_pos * s_disp_stride_y;
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                u[node * NB_DOFS + d] =
                                    displacement[disp_idx + d * s_disp_stride_d];
                            }
                        }

                        // Compute only the rows of K @ u that correspond to
                        // this node
                        for (Index_t d = 0; d < NB_DOFS; ++d) {
                            Index_t row = local_node * NB_DOFS + d;
                            Real contrib = 0.0;
                            for (Index_t j = 0; j < NB_ELEM_DOFS; ++j) {
                                contrib +=
                                    (2.0 * mu_val * G[row * NB_ELEM_DOFS + j] +
                                     lam * V[row * NB_ELEM_DOFS + j]) *
                                    u[j];
                            }
                            f[d] += contrib;
                        }
                    }

                    // Write force for this node
                    SIndex_t base = ix * s_force_stride_x + iy * s_force_stride_y;
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

    }  // namespace isotropic_stiffness_kernels

}  // namespace muGrid
