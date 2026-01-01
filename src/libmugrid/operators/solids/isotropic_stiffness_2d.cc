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
                               "1 ghost cell on the "
                               "right side of displacement/force fields "
                               "(nb_ghosts_right >= (1, 1))");
        }

        // Get material field ghost configuration
        auto mat_nb_ghosts_left = mat_global_fc->get_nb_ghosts_left();
        auto mat_nb_ghosts_right = mat_global_fc->get_nb_ghosts_right();

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

        // Determine if periodic BC based on material field size
        // Periodic: nelx == nnx (N elements for N nodes, wraps around)
        // Non-periodic: nelx == nnx - 1 (N-1 elements for N nodes)
        bool periodic_x = (nelx == nnx);
        bool periodic_y = (nely == nny);
        bool periodic = periodic_x && periodic_y;

        // Validate material field dimensions
        if (!periodic_x && nelx != nnx - 1) {
            throw RuntimeError(
                "Material field x-dimension (" + std::to_string(nelx) +
                ") must equal interior nodes (" + std::to_string(nnx) +
                ") for periodic BC, or interior nodes - 1 (" +
                std::to_string(nnx - 1) + ") for non-periodic BC");
        }
        if (!periodic_y && nely != nny - 1) {
            throw RuntimeError(
                "Material field y-dimension (" + std::to_string(nely) +
                ") must equal interior nodes (" + std::to_string(nny) +
                ") for periodic BC, or interior nodes - 1 (" +
                std::to_string(nny - 1) + ") for non-periodic BC");
        }
        if (periodic_x != periodic_y) {
            throw RuntimeError("Mixed periodic/non-periodic BC not supported. "
                               "Material dimensions: (" +
                               std::to_string(nelx) + ", " +
                               std::to_string(nely) + "), interior nodes: (" +
                               std::to_string(nnx) + ", " +
                               std::to_string(nny) + ")");
        }

        // For periodic BC, need left ghosts on displacement/force fields
        if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1)) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D with periodic BC requires at "
                "least 1 "
                "ghost cell on the left side of displacement/force fields");
        }

        // For periodic BC, material field should also have left ghosts
        // (filled via communicate_ghosts once before calling apply)
        if (periodic &&
            (mat_nb_ghosts_left[0] < 1 || mat_nb_ghosts_left[1] < 1)) {
            throw RuntimeError(
                "IsotropicStiffnessOperator2D with periodic BC requires at "
                "least 1 "
                "ghost cell on the left side of material fields (lambda, mu). "
                "Call communicate_ghosts on material fields once before "
                "apply.");
        }

        // Node dimensions (for displacement/force fields with ghosts)
        auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];
        Index_t ny = nb_nodes[1];

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
            G_matrix.data(), V_matrix.data(), alpha, increment, periodic);
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
            const Real * V, Real alpha, bool increment, bool periodic) {

            constexpr Index_t NB_NODES = 4;
            constexpr Index_t NB_DOFS = 2;
            constexpr Index_t NB_ELEM_DOFS = NB_NODES * NB_DOFS;

            // Neighboring element offsets and corresponding local node index
            // Element at (ix + eox, iy + eoy) has this node as local node
            // `local_node`
            static const Index_t ELEM_OFFSETS[4][3] = {
                {-1, -1, 3},  // Element (ix-1, iy-1): this node is local node 3
                {0, -1, 2},   // Element (ix,   iy-1): this node is local node 2
                {-1, 0, 1},   // Element (ix-1, iy  ): this node is local node 1
                {0, 0, 0}     // Element (ix,   iy  ): this node is local node 0
            };

            // Gather pattern: loop over interior NODES, gather from neighboring
            // elements
            for (Index_t iy = 0; iy < nny; ++iy) {
                for (Index_t ix = 0; ix < nnx; ++ix) {
                    // Accumulate force for this node
                    Real f[NB_DOFS] = {0.0, 0.0};

                    // Loop over neighboring elements
                    for (Index_t elem = 0; elem < 4; ++elem) {
                        Index_t ex = ix + ELEM_OFFSETS[elem][0];
                        Index_t ey = iy + ELEM_OFFSETS[elem][1];
                        Index_t local_node = ELEM_OFFSETS[elem][2];

                        // Handle boundary: skip out-of-bounds elements for
                        // non-periodic BC For periodic BC, ghost cells contain
                        // wrap-around data, so no skip needed
                        if (!periodic &&
                            (ex < 0 || ex >= nelx || ey < 0 || ey >= nely)) {
                            continue;
                        }

                        // Get material parameters for this element
                        // For periodic BC with ghost cells: ex=-1 accesses left
                        // ghost (contains last element)
                        Real lam =
                            lambda[ex * mat_stride_x + ey * mat_stride_y];
                        Real mu_val = mu[ex * mat_stride_x + ey * mat_stride_y];

                        // Gather displacements from all 4 nodes of this element
                        Real u[NB_ELEM_DOFS];
                        for (Index_t node = 0; node < NB_NODES; ++node) {
                            // Node position = element position + node offset
                            Index_t nx_pos = ex + NODE_OFFSET_2D[node][0];
                            Index_t ny_pos = ey + NODE_OFFSET_2D[node][1];
                            // For periodic, node positions can go up to nelx (=
                            // nnx for periodic) which accesses the right ghost
                            // containing the periodic copy
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                u[node * NB_DOFS + d] =
                                    displacement[nx_pos * disp_stride_x +
                                                 ny_pos * disp_stride_y +
                                                 d * disp_stride_d];
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
                    Index_t base = ix * force_stride_x + iy * force_stride_y;
                    if (increment) {
                        for (Index_t d = 0; d < NB_DOFS; ++d) {
                            force[base + d * force_stride_d] += alpha * f[d];
                        }
                    } else {
                        for (Index_t d = 0; d < NB_DOFS; ++d) {
                            force[base + d * force_stride_d] = alpha * f[d];
                        }
                    }
                }
            }
        }

    }  // namespace isotropic_stiffness_kernels

}  // namespace muGrid
