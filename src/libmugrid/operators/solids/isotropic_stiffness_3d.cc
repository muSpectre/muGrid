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

#include "isotropic_stiffness_3d.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

namespace muGrid {

    // ============================================================================
    // 3D Implementation
    // ============================================================================

    // 3D shape function gradients for 5 tetrahedra per voxel
    // B_3D[quad][dim][node]
    // Nodes: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(1,1,0),
    //        4=(0,0,1), 5=(1,0,1), 6=(0,1,1), 7=(1,1,1)
    static const Real B_3D[5][3][8] = {
        // Tet 0: Central tetrahedron (nodes 1,2,4,7)
        {{0.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5},   // d/dx
         {0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5},   // d/dy
         {0.0, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5}},  // d/dz
        // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
        {{-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
         {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}},
        // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
        {{0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
         {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0}},
        // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
        {{0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0},
         {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0},
         {0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},
        // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
        {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0},
         {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0},
         {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0}}};

    // Quadrature weights for 3D (volume of each tet / voxel volume)
    static const Real W_3D[5] = {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
                                 1.0 / 6.0};

    IsotropicStiffnessOperator3D::IsotropicStiffnessOperator3D(
        const std::vector<Real> & grid_spacing)
        : grid_spacing{grid_spacing} {
        if (grid_spacing.size() != 3) {
            throw RuntimeError("3D operator requires 3D grid spacing");
        }
        this->precompute_matrices();
    }

    void IsotropicStiffnessOperator3D::precompute_matrices() {
        // Initialize matrices to zero
        std::fill(G_matrix.begin(), G_matrix.end(), 0.0);
        std::fill(V_matrix.begin(), V_matrix.end(), 0.0);

        const Real hx = grid_spacing[0];
        const Real hy = grid_spacing[1];
        const Real hz = grid_spacing[2];
        const Real inv_hx = 1.0 / hx;
        const Real inv_hy = 1.0 / hy;
        const Real inv_hz = 1.0 / hz;

        // Scale factors for each derivative direction
        const Real scale[3] = {inv_hx, inv_hy, inv_hz};

        // Compute G and V matrices
        // G = Σ_q w_q B_q^T I B_q (for shear modulus μ)
        // V = Σ_q w_q (B_q^T m)(m^T B_q) (for Lamé λ)
        // where m = [1, 1, 1, 0, 0, 0]^T (Voigt trace selector)

        for (Index_t q = 0; q < NB_QUAD; ++q) {
            Real w = W_3D[q] * hx * hy * hz;  // Weight includes element volume

            for (Index_t I = 0; I < NB_NODES; ++I) {
                for (Index_t J = 0; J < NB_NODES; ++J) {
                    for (Index_t a = 0; a < NB_DOFS_PER_NODE; ++a) {
                        for (Index_t b = 0; b < NB_DOFS_PER_NODE; ++b) {
                            Real g_contrib = 0.0;
                            Real v_contrib = 0.0;

                            // G contribution from diagonal strains (εxx, εyy,
                            // εzz)
                            if (a == b) {
                                g_contrib += B_3D[q][a][I] * scale[a] *
                                             B_3D[q][a][J] * scale[a];
                            }

                            // G contribution from shear strains
                            // 2εyz = du_z/dy + du_y/dz
                            // 2εxz = du_z/dx + du_x/dz
                            // 2εxy = du_y/dx + du_x/dy
                            // For shear strain ε_ab, the B vector has:
                            //   B[u_a] = dN/dx_b and B[u_b] = dN/dx_a
                            // So the stiffness contribution is:
                            //   K[u_a,I; u_a,J] += μ * dN_I/dx_b * dN_J/dx_b
                            //   K[u_b,I; u_b,J] += μ * dN_I/dx_a * dN_J/dx_a
                            //   K[u_a,I; u_b,J] += μ * dN_I/dx_b * dN_J/dx_a
                            //   K[u_b,I; u_a,J] += μ * dN_I/dx_a * dN_J/dx_b
                            if (a == 0 && b == 0) {
                                // u_x - u_x: shear εxy (dN/dy) and εxz (dN/dz)
                                g_contrib += 0.5 * B_3D[q][1][I] * scale[1] *
                                             B_3D[q][1][J] * scale[1];  // εxy
                                g_contrib += 0.5 * B_3D[q][2][I] * scale[2] *
                                             B_3D[q][2][J] * scale[2];  // εxz
                            } else if (a == 1 && b == 1) {
                                // u_y - u_y: shear εxy (dN/dx) and εyz (dN/dz)
                                g_contrib += 0.5 * B_3D[q][0][I] * scale[0] *
                                             B_3D[q][0][J] * scale[0];  // εxy
                                g_contrib += 0.5 * B_3D[q][2][I] * scale[2] *
                                             B_3D[q][2][J] * scale[2];  // εyz
                            } else if (a == 2 && b == 2) {
                                // u_z - u_z: shear εxz (dN/dx) and εyz (dN/dy)
                                g_contrib += 0.5 * B_3D[q][0][I] * scale[0] *
                                             B_3D[q][0][J] * scale[0];  // εxz
                                g_contrib += 0.5 * B_3D[q][1][I] * scale[1] *
                                             B_3D[q][1][J] * scale[1];  // εyz
                            } else if (a == 0 && b == 1) {
                                // u_x,I - u_y,J cross coupling via εxy
                                g_contrib += 0.5 * B_3D[q][1][I] * scale[1] *
                                             B_3D[q][0][J] * scale[0];
                            } else if (a == 1 && b == 0) {
                                // u_y,I - u_x,J cross coupling via εxy
                                g_contrib += 0.5 * B_3D[q][0][I] * scale[0] *
                                             B_3D[q][1][J] * scale[1];
                            } else if (a == 0 && b == 2) {
                                // u_x,I - u_z,J cross coupling via εxz
                                g_contrib += 0.5 * B_3D[q][2][I] * scale[2] *
                                             B_3D[q][0][J] * scale[0];
                            } else if (a == 2 && b == 0) {
                                // u_z,I - u_x,J cross coupling via εxz
                                g_contrib += 0.5 * B_3D[q][0][I] * scale[0] *
                                             B_3D[q][2][J] * scale[2];
                            } else if (a == 1 && b == 2) {
                                // u_y,I - u_z,J cross coupling via εyz
                                g_contrib += 0.5 * B_3D[q][2][I] * scale[2] *
                                             B_3D[q][1][J] * scale[1];
                            } else if (a == 2 && b == 1) {
                                // u_z,I - u_y,J cross coupling via εyz
                                g_contrib += 0.5 * B_3D[q][1][I] * scale[1] *
                                             B_3D[q][2][J] * scale[2];
                            }

                            // V contribution: (B^T m)(m^T B)
                            // m = [1,1,1,0,0,0]^T selects trace(ε) = εxx + εyy
                            // + εzz
                            Real BtmI =
                                (a == 0 ? B_3D[q][0][I] * scale[0] : 0.0) +
                                (a == 1 ? B_3D[q][1][I] * scale[1] : 0.0) +
                                (a == 2 ? B_3D[q][2][I] * scale[2] : 0.0);
                            Real BtmJ =
                                (b == 0 ? B_3D[q][0][J] * scale[0] : 0.0) +
                                (b == 1 ? B_3D[q][1][J] * scale[1] : 0.0) +
                                (b == 2 ? B_3D[q][2][J] * scale[2] : 0.0);
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

    void IsotropicStiffnessOperator3D::apply(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        TypedFieldBase<Real> & force) const {
        apply_impl(displacement, lambda, mu, 1.0, force, false);
    }

    void IsotropicStiffnessOperator3D::apply_increment(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        Real alpha, TypedFieldBase<Real> & force) const {
        apply_impl(displacement, lambda, mu, alpha, force, true);
    }

    void IsotropicStiffnessOperator3D::apply_impl(
        const TypedFieldBase<Real> & displacement,
        const TypedFieldBase<Real> & lambda, const TypedFieldBase<Real> & mu,
        Real alpha, TypedFieldBase<Real> & force, bool increment) const {

        // Get field collections and validate
        auto & disp_coll = displacement.get_collection();
        auto * disp_global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&disp_coll);
        if (!disp_global_fc) {
            throw RuntimeError(
                "IsotropicStiffnessOperator3D requires GlobalFieldCollection");
        }

        // Get dimensions from material field collection (defines number of
        // elements)
        auto & mat_coll = lambda.get_collection();
        auto * mat_global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&mat_coll);
        if (!mat_global_fc) {
            throw RuntimeError("IsotropicStiffnessOperator3D material fields "
                               "require GlobalFieldCollection");
        }

        // Validate ghost configuration for displacement/force fields
        auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
        auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
        if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1 ||
            nb_ghosts_right[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on the "
                               "right side of displacement/force fields "
                               "(nb_ghosts_right >= (1, 1, 1))");
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
        Index_t nelz = nb_elements[2];

        // Get number of interior nodes
        auto nb_interior =
            disp_global_fc->get_nb_subdomain_grid_pts_without_ghosts();
        Index_t nnx = nb_interior[0];
        Index_t nny = nb_interior[1];
        Index_t nnz = nb_interior[2];

        // Determine if periodic BC based on material field size
        // Periodic: nel == nn (N elements for N nodes, wraps around)
        // Non-periodic: nel == nn - 1 (N-1 elements for N nodes)
        bool periodic_x = (nelx == nnx);
        bool periodic_y = (nely == nny);
        bool periodic_z = (nelz == nnz);
        bool periodic = periodic_x && periodic_y && periodic_z;
        bool non_periodic =
            (nelx == nnx - 1) && (nely == nny - 1) && (nelz == nnz - 1);

        // Validate material field dimensions
        if (!periodic && !non_periodic) {
            throw RuntimeError(
                "Material field dimensions (" + std::to_string(nelx) + ", " +
                std::to_string(nely) + ", " + std::to_string(nelz) +
                ") must equal interior nodes (" + std::to_string(nnx) + ", " +
                std::to_string(nny) + ", " + std::to_string(nnz) +
                ") for periodic BC, or interior nodes - 1 (" +
                std::to_string(nnx - 1) + ", " + std::to_string(nny - 1) +
                ", " + std::to_string(nnz - 1) + ") for non-periodic BC");
        }

        // For periodic BC, need left ghosts on displacement/force fields
        if (periodic && (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 ||
                         nb_ghosts_left[2] < 1)) {
            throw RuntimeError(
                "IsotropicStiffnessOperator3D with periodic BC requires at "
                "least 1 "
                "ghost cell on the left side of displacement/force fields");
        }

        // For periodic BC, material field should also have left ghosts
        if (periodic &&
            (mat_nb_ghosts_left[0] < 1 || mat_nb_ghosts_left[1] < 1 ||
             mat_nb_ghosts_left[2] < 1)) {
            throw RuntimeError(
                "IsotropicStiffnessOperator3D with periodic BC requires at "
                "least 1 "
                "ghost cell on the left side of material fields (lambda, mu). "
                "Call communicate_ghosts on material fields once before "
                "apply.");
        }

        // Node dimensions (for displacement/force fields with ghosts)
        auto nb_nodes = disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nx = nb_nodes[0];
        Index_t ny = nb_nodes[1];
        Index_t nz = nb_nodes[2];

        // Ghost offsets (interior starts at this offset in the ghosted array)
        Index_t disp_ghost_offset_x = nb_ghosts_left[0];
        Index_t disp_ghost_offset_y = nb_ghosts_left[1];
        Index_t disp_ghost_offset_z = nb_ghosts_left[2];
        Index_t mat_ghost_offset_x = mat_nb_ghosts_left[0];
        Index_t mat_ghost_offset_y = mat_nb_ghosts_left[1];
        Index_t mat_ghost_offset_z = mat_nb_ghosts_left[2];

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

        // Offset data pointers to account for left ghosts
        Index_t disp_offset = disp_ghost_offset_x * disp_stride_x +
                              disp_ghost_offset_y * disp_stride_y +
                              disp_ghost_offset_z * disp_stride_z;
        Index_t force_offset = disp_ghost_offset_x * force_stride_x +
                               disp_ghost_offset_y * force_stride_y +
                               disp_ghost_offset_z * force_stride_z;
        Index_t mat_offset = mat_ghost_offset_x * mat_stride_x +
                             mat_ghost_offset_y * mat_stride_y +
                             mat_ghost_offset_z * mat_stride_z;

        const Real * disp_data = displacement.data() + disp_offset;
        const Real * lambda_data = lambda.data() + mat_offset;
        const Real * mu_data = mu.data() + mat_offset;
        Real * force_data = force.data() + force_offset;

        isotropic_stiffness_kernels::isotropic_stiffness_3d_host(
            disp_data, lambda_data, mu_data, force_data, nnx, nny,
            nnz,               // Number of interior nodes
            nelx, nely, nelz,  // Number of elements
            disp_stride_x, disp_stride_y, disp_stride_z, disp_stride_d,
            mat_stride_x, mat_stride_y, mat_stride_z, force_stride_x,
            force_stride_y, force_stride_z, force_stride_d, G_matrix.data(),
            V_matrix.data(), alpha, increment, periodic);
    }

    // ============================================================================
    // Host Kernel Implementations
    // ============================================================================

    namespace isotropic_stiffness_kernels {

        // Node offsets for 3D [node][dim]
        static const Index_t NODE_OFFSET_3D[8][3] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        void isotropic_stiffness_3d_host(
            const Real * MUGRID_RESTRICT displacement,
            const Real * MUGRID_RESTRICT lambda,
            const Real * MUGRID_RESTRICT mu, Real * MUGRID_RESTRICT force,
            Index_t nnx, Index_t nny, Index_t nnz, Index_t nelx, Index_t nely,
            Index_t nelz, Index_t disp_stride_x, Index_t disp_stride_y,
            Index_t disp_stride_z, Index_t disp_stride_d, Index_t mat_stride_x,
            Index_t mat_stride_y, Index_t mat_stride_z, Index_t force_stride_x,
            Index_t force_stride_y, Index_t force_stride_z,
            Index_t force_stride_d, const Real * G, const Real * V, Real alpha,
            bool increment, bool periodic) {

            constexpr Index_t NB_NODES = 8;
            constexpr Index_t NB_DOFS = 3;
            constexpr Index_t NB_ELEM_DOFS = NB_NODES * NB_DOFS;

            // Neighboring element offsets and corresponding local node index
            // Element at (ix + eox, iy + eoy, iz + eoz) has this node as local
            // node
            static const Index_t ELEM_OFFSETS[8][4] = {
                {-1, -1, -1,
                 7},  // Element (ix-1, iy-1, iz-1): local node 7 (corner 1,1,1)
                {0, -1, -1,
                 6},  // Element (ix,   iy-1, iz-1): local node 6 (corner 0,1,1)
                {-1, 0, -1,
                 5},  // Element (ix-1, iy,   iz-1): local node 5 (corner 1,0,1)
                {0, 0, -1,
                 4},  // Element (ix,   iy,   iz-1): local node 4 (corner 0,0,1)
                {-1, -1, 0,
                 3},  // Element (ix-1, iy-1, iz  ): local node 3 (corner 1,1,0)
                {0, -1, 0,
                 2},  // Element (ix,   iy-1, iz  ): local node 2 (corner 0,1,0)
                {-1, 0, 0,
                 1},  // Element (ix-1, iy,   iz  ): local node 1 (corner 1,0,0)
                {0, 0, 0, 0}
                // Element (ix,   iy,   iz  ): local node 0 (corner 0,0,0)
            };

            // Gather pattern: loop over interior NODES
            for (Index_t iz = 0; iz < nnz; ++iz) {
                for (Index_t iy = 0; iy < nny; ++iy) {
                    for (Index_t ix = 0; ix < nnx; ++ix) {
                        // Accumulate force for this node
                        Real f[NB_DOFS] = {0.0, 0.0, 0.0};

                        // Loop over neighboring elements
                        for (Index_t elem = 0; elem < 8; ++elem) {
                            Index_t ex = ix + ELEM_OFFSETS[elem][0];
                            Index_t ey = iy + ELEM_OFFSETS[elem][1];
                            Index_t ez = iz + ELEM_OFFSETS[elem][2];
                            Index_t local_node = ELEM_OFFSETS[elem][3];

                            // Handle boundary: skip out-of-bounds elements for
                            // non-periodic BC For periodic BC, ghost cells
                            // contain wrap-around data, so no skip needed
                            if (!periodic &&
                                (ex < 0 || ex >= nelx || ey < 0 || ey >= nely ||
                                 ez < 0 || ez >= nelz)) {
                                continue;
                            }

                            // Get material parameters
                            // For periodic BC with ghost cells: ex=-1 accesses
                            // left ghost (contains last element)
                            Index_t mat_idx = ex * mat_stride_x +
                                              ey * mat_stride_y +
                                              ez * mat_stride_z;
                            Real lam = lambda[mat_idx];
                            Real mu_val = mu[mat_idx];

                            // Gather displacements from all 8 nodes of this
                            // element
                            Real u[NB_ELEM_DOFS];
                            for (Index_t node = 0; node < NB_NODES; ++node) {
                                Index_t nx_pos = ex + NODE_OFFSET_3D[node][0];
                                Index_t ny_pos = ey + NODE_OFFSET_3D[node][1];
                                Index_t nz_pos = ez + NODE_OFFSET_3D[node][2];
                                Index_t disp_idx = nx_pos * disp_stride_x +
                                                   ny_pos * disp_stride_y +
                                                   nz_pos * disp_stride_z;
                                for (Index_t d = 0; d < NB_DOFS; ++d) {
                                    u[node * NB_DOFS + d] =
                                        displacement[disp_idx +
                                                     d * disp_stride_d];
                                }
                            }

                            // Compute only the rows that correspond to this
                            // node
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                Index_t row = local_node * NB_DOFS + d;
                                Real contrib = 0.0;
                                for (Index_t j = 0; j < NB_ELEM_DOFS; ++j) {
                                    contrib +=
                                        (2.0 * mu_val *
                                             G[row * NB_ELEM_DOFS + j] +
                                         lam * V[row * NB_ELEM_DOFS + j]) *
                                        u[j];
                                }
                                f[d] += contrib;
                            }
                        }

                        // Write force for this node
                        Index_t base = ix * force_stride_x +
                                       iy * force_stride_y +
                                       iz * force_stride_z;
                        if (increment) {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * force_stride_d] +=
                                    alpha * f[d];
                            }
                        } else {
                            for (Index_t d = 0; d < NB_DOFS; ++d) {
                                force[base + d * force_stride_d] = alpha * f[d];
                            }
                        }
                    }
                }
            }
        }

    }  // namespace isotropic_stiffness_kernels

}  // namespace muGrid
