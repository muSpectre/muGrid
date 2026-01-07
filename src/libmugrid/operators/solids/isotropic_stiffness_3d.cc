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
    // B_3D[quad][dim][node] where dim: 0=d/dx, 1=d/dy, 2=d/dz
    // Nodes: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(1,1,0),
    //        4=(0,0,1), 5=(1,0,1), 6=(0,1,1), 7=(1,1,1)
    // These values match FEMGradientOperator's 5-tetrahedra decomposition
    static const Real B_3D[5][3][8] = {
        // Tet 0: Central tetrahedron (nodes 1,2,4,7)
        {{0.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5},   // d/dx
         {0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5},   // d/dy
         {0.0, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5}},  // d/dz
        // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
        {{-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},    // d/dx
         {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},    // d/dy
         {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}},   // d/dz
        // Tet 2: Corner at (0,1,1) - nodes 2,4,6,7
        {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0},    // d/dx
         {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0},    // d/dy
         {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0}},   // d/dz
        // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
        {{0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0},    // d/dx
         {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0},    // d/dy
         {0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},   // d/dz
        // Tet 4: Corner at (1,1,0) - nodes 1,2,3,7
        {{0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0},    // d/dx
         {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},    // d/dy
         {0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0}}};

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
                            // Shear strain contributions
                            // Using single-line expressions to avoid parsing issues
                            Real dNI_dx = B_3D[q][0][I] * scale[0];
                            Real dNI_dy = B_3D[q][1][I] * scale[1];
                            Real dNI_dz = B_3D[q][2][I] * scale[2];
                            Real dNJ_dx = B_3D[q][0][J] * scale[0];
                            Real dNJ_dy = B_3D[q][1][J] * scale[1];
                            Real dNJ_dz = B_3D[q][2][J] * scale[2];

                            // Shear strain contributions with explicit intermediate
                            // variables to avoid potential compiler optimization issues
                            if (a == 0 && b == 0) {
                                // u_x - u_x: shear εxy (dN/dy) and εxz (dN/dz)
                                Real shear_xy = 0.5 * dNI_dy * dNJ_dy;
                                Real shear_xz = 0.5 * dNI_dz * dNJ_dz;
                                g_contrib += shear_xy + shear_xz;
                            } else if (a == 1 && b == 1) {
                                // u_y - u_y: shear εxy (dN/dx) and εyz (dN/dz)
                                Real shear_xy = 0.5 * dNI_dx * dNJ_dx;
                                Real shear_yz = 0.5 * dNI_dz * dNJ_dz;
                                g_contrib += shear_xy + shear_yz;
                            } else if (a == 2 && b == 2) {
                                // u_z - u_z: shear εxz (dN/dx) and εyz (dN/dy)
                                Real shear_xz = 0.5 * dNI_dx * dNJ_dx;
                                Real shear_yz = 0.5 * dNI_dy * dNJ_dy;
                                g_contrib += shear_xz + shear_yz;
                            } else if (a == 0 && b == 1) {
                                // u_x,I - u_y,J cross coupling via εxy
                                g_contrib += 0.5 * dNI_dy * dNJ_dx;
                            } else if (a == 1 && b == 0) {
                                // u_y,I - u_x,J cross coupling via εxy
                                g_contrib += 0.5 * dNI_dx * dNJ_dy;
                            } else if (a == 0 && b == 2) {
                                // u_x,I - u_z,J cross coupling via εxz
                                g_contrib += 0.5 * dNI_dz * dNJ_dx;
                            } else if (a == 2 && b == 0) {
                                // u_z,I - u_x,J cross coupling via εxz
                                g_contrib += 0.5 * dNI_dx * dNJ_dz;
                            } else if (a == 1 && b == 2) {
                                // u_y,I - u_z,J cross coupling via εyz
                                g_contrib += 0.5 * dNI_dz * dNJ_dy;
                            } else if (a == 2 && b == 1) {
                                // u_z,I - u_y,J cross coupling via εyz
                                g_contrib += 0.5 * dNI_dy * dNJ_dz;
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
        // Node-based indexing: requires ghosts on both left and right
        auto nb_ghosts_left = disp_global_fc->get_nb_ghosts_left();
        auto nb_ghosts_right = disp_global_fc->get_nb_ghosts_right();
        if (nb_ghosts_left[0] < 1 || nb_ghosts_left[1] < 1 ||
            nb_ghosts_left[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on the left side of "
                               "displacement/force fields");
        }
        if (nb_ghosts_right[0] < 1 || nb_ghosts_right[1] < 1 ||
            nb_ghosts_right[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on the right side of "
                               "displacement/force fields");
        }

        // Stencil requirements: the kernel gathers from 8 neighboring elements
        // at offsets [-1, 0] in each dimension, and each element accesses its
        // 8 corner nodes at offsets [0, 1]. This requires:
        // - 1 left ghost (for elements at -1 and their nodes)
        // - 1 right ghost (for nodes at +1 in elements at offset 0)
        constexpr Index_t STENCIL_LEFT = 1;
        constexpr Index_t STENCIL_RIGHT = 1;

        // Compute computable region: total with ghosts minus stencil requirements
        auto nb_with_ghosts =
            disp_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nnx = nb_with_ghosts[0] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nny = nb_with_ghosts[1] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nnz = nb_with_ghosts[2] - STENCIL_LEFT - STENCIL_RIGHT;

        // Material field dimensions (computable region)
        // Node-based indexing: material field must have same size as node field
        auto mat_nb_with_ghosts =
            mat_global_fc->get_nb_subdomain_grid_pts_with_ghosts();
        Index_t nelx = mat_nb_with_ghosts[0] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nely = mat_nb_with_ghosts[1] - STENCIL_LEFT - STENCIL_RIGHT;
        Index_t nelz = mat_nb_with_ghosts[2] - STENCIL_LEFT - STENCIL_RIGHT;

        // Validate material field computable region matches node field
        if (nelx != nnx || nely != nny || nelz != nnz) {
            throw RuntimeError(
                "IsotropicStiffnessOperator3D: material field computable region (" +
                std::to_string(nelx) + ", " + std::to_string(nely) + ", " +
                std::to_string(nelz) + ") must match node field computable region (" +
                std::to_string(nnx) + ", " + std::to_string(nny) + ", " +
                std::to_string(nnz) + ")");
        }

        // Validate material field ghost configuration matches node field
        auto mat_nb_ghosts_left = mat_global_fc->get_nb_ghosts_left();
        auto mat_nb_ghosts_right = mat_global_fc->get_nb_ghosts_right();
        if (mat_nb_ghosts_left[0] < 1 || mat_nb_ghosts_left[1] < 1 ||
            mat_nb_ghosts_left[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on the left side of material "
                               "fields (lambda, mu)");
        }
        if (mat_nb_ghosts_right[0] < 1 || mat_nb_ghosts_right[1] < 1 ||
            mat_nb_ghosts_right[2] < 1) {
            throw RuntimeError("IsotropicStiffnessOperator3D requires at least "
                               "1 ghost cell on the right side of material "
                               "fields (lambda, mu)");
        }

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
            V_matrix.data(), alpha, increment);
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
            bool increment) {

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
            SIndex_t s_mat_stride_x = static_cast<SIndex_t>(mat_stride_x);
            SIndex_t s_mat_stride_y = static_cast<SIndex_t>(mat_stride_y);
            SIndex_t s_mat_stride_z = static_cast<SIndex_t>(mat_stride_z);
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
                        Real f[NB_DOFS] = {0.0, 0.0, 0.0};

                        // Loop over neighboring elements (all 8 elements guaranteed
                        // to exist for nodes in this iteration range)
                        for (Index_t elem = 0; elem < 8; ++elem) {
                            // Element indices (can be -1 for periodic BC accessing ghost cells)
                            SIndex_t ex = ix + ELEM_OFFSETS[elem][0];
                            SIndex_t ey = iy + ELEM_OFFSETS[elem][1];
                            SIndex_t ez = iz + ELEM_OFFSETS[elem][2];
                            Index_t local_node = ELEM_OFFSETS[elem][3];

                            // Get material parameters
                            SIndex_t mat_idx = ex * s_mat_stride_x +
                                               ey * s_mat_stride_y +
                                               ez * s_mat_stride_z;
                            Real lam = lambda[mat_idx];
                            Real mu_val = mu[mat_idx];

                            // Gather displacements from all 8 nodes of this element
                            Real u[NB_ELEM_DOFS];
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

    }  // namespace isotropic_stiffness_kernels

}  // namespace muGrid
