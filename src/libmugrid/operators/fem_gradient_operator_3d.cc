/**
 * @file   fem_gradient_operator_3d.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Host implementation of hard-coded 3D linear FEM gradient operator
 *
 * Copyright © 2024 Lars Pastewka
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

#include "fem_gradient_operator_3d.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

    // =========================================================================
    // 3D Shape Function Gradients Definition
    // =========================================================================
    // These are computed for the 5-tet Kuhn triangulation of a unit cube.
    // =========================================================================

    namespace fem_gradient_kernels {

        // 3D shape function gradients [dim][quad][node]
        const Real B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D] = {
            // d/dx gradients (scaled by 1/hx at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0}},
            // d/dy gradients (scaled by 1/hy at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0}},
            // d/dz gradients (scaled by 1/hz at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0}}};

    }  // namespace fem_gradient_kernels

    // =========================================================================
    // FEMGradientOperator3D Implementation
    // =========================================================================

    FEMGradientOperator3D::FEMGradientOperator3D(std::vector<Real> grid_spacing)
        : Parent{}, grid_spacing{std::move(grid_spacing)} {
        // Default grid spacing is 1.0 in each direction
        if (this->grid_spacing.empty()) {
            this->grid_spacing.resize(DIM, 1.0);
        }
        if (static_cast<Index_t>(this->grid_spacing.size()) != DIM) {
            throw RuntimeError("Grid spacing must have 3 components for 3D operator");
        }
    }

    const GlobalFieldCollection &
    FEMGradientOperator3D::validate_fields(const Field & nodal_field,
                                           const Field & gradient_field,
                                           Index_t & nb_components) const {

        auto & nodal_collection = nodal_field.get_collection();
        auto & gradient_collection = gradient_field.get_collection();

        if (&nodal_collection != &gradient_collection) {
            throw RuntimeError("Nodal and gradient fields must belong to the "
                               "same field collection");
        }

        auto * global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&nodal_collection);
        if (!global_fc) {
            throw RuntimeError(
                "FEMGradientOperator3D requires GlobalFieldCollection");
        }

        if (global_fc->get_spatial_dim() != DIM) {
            throw RuntimeError("Field collection dimension (" +
                               std::to_string(global_fc->get_spatial_dim()) +
                               ") does not match operator dimension (3)");
        }

        Index_t nb_nodal_components = nodal_field.get_nb_components();
        Index_t nb_grad_components = gradient_field.get_nb_components();
        Index_t expected_grad_components =
            this->get_nb_output_components() * nb_nodal_components;

        if (nb_grad_components != expected_grad_components) {
            std::stringstream err_msg;
            err_msg << "Component mismatch: Expected gradient field with "
                    << expected_grad_components << " components ("
                    << this->get_nb_output_components() << " output components × "
                    << nb_nodal_components << " nodal components) but got "
                    << nb_grad_components << " components.";
            throw RuntimeError(err_msg.str());
        }

        nb_components = nb_nodal_components;
        return *global_fc;
    }

    std::vector<Real> FEMGradientOperator3D::get_quadrature_weights() const {
        // 5-tet decomposition:
        // - Central tetrahedron (tet 0): volume = 1/3 of voxel
        // - Corner tetrahedra (tet 1-4): volume = 1/6 of voxel each
        Real voxel_volume = this->grid_spacing[0] * this->grid_spacing[1] *
                            this->grid_spacing[2];
        return {voxel_volume / 3.0,   // Tet 0: Central
                voxel_volume / 6.0,   // Tet 1: Corner (0,0,0)
                voxel_volume / 6.0,   // Tet 2: Corner (1,1,0)
                voxel_volume / 6.0,   // Tet 3: Corner (1,0,1)
                voxel_volume / 6.0};  // Tet 4: Corner (0,1,1)
    }

    std::vector<Real> FEMGradientOperator3D::get_coefficients() const {
        using namespace fem_gradient_kernels;

        // Shape: (nb_operators=3, nb_quad_pts=5, nb_nodal_pts=1, 2, 2, 2)
        // Total size: 3 * 5 * 1 * 2 * 2 * 2 = 120
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        Real hz = this->grid_spacing[2];
        std::vector<Real> result;
        result.reserve(120);

        // Fortran-order: first index (operators) varies fastest
        for (Index_t k = 0; k < 2; ++k) {                  // stencil_z (outermost)
            for (Index_t j = 0; j < 2; ++j) {              // stencil_y
                for (Index_t i = 0; i < 2; ++i) {          // stencil_x
                    for (Index_t q = 0; q < NB_QUAD_3D; ++q) {     // quad pt
                        for (Index_t d = 0; d < DIM_3D; ++d) {     // operator (innermost)
                            // Map (i,j,k) to node index: node = k*4 + j*2 + i
                            Index_t node = k * 4 + j * 2 + i;
                            Real grad_val = B_3D_REF[d][q][node];
                            // Scale by inverse grid spacing
                            if (d == 0)
                                grad_val /= hx;
                            else if (d == 1)
                                grad_val /= hy;
                            else
                                grad_val /= hz;
                            result.push_back(grad_val);
                        }
                    }
                }
            }
        }
        return result;
    }

    void
    FEMGradientOperator3D::apply_impl(const TypedFieldBase<Real> & nodal_field,
                                      TypedFieldBase<Real> & gradient_field,
                                      Real alpha, bool increment) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        const Real * nodal = nodal_field.data();
        Real * gradient = gradient_field.data();

        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        Real hz = this->grid_spacing[2];

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];
        Index_t nz = nb_grid_pts[2];

        // Nodal field strides [components, sub_pts, x, y, z]
        Index_t nb_sub = this->get_nb_input_components();
        Index_t nodal_stride_c = 1;
        Index_t nodal_stride_n = nb_components;
        Index_t nodal_stride_x = nb_components * nb_sub;
        Index_t nodal_stride_y = nb_components * nb_sub * nx;
        Index_t nodal_stride_z = nb_components * nb_sub * nx * ny;

        // Gradient field strides [components, operators, nb_quad, x, y, z]
        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_c = 1;
        Index_t grad_stride_d = nb_components;
        Index_t grad_stride_q = nb_components * dim;
        Index_t grad_stride_x = nb_components * dim * nb_quad;
        Index_t grad_stride_y = nb_components * dim * nb_quad * nx;
        Index_t grad_stride_z = nb_components * dim * nb_quad * nx * ny;

        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * nodal_comp = nodal + comp * nodal_stride_c;
            Real * gradient_comp = gradient + comp * grad_stride_c;

            fem_gradient_kernels::fem_gradient_3d_host(
                nodal_comp, gradient_comp, nx, ny, nz, nodal_stride_x,
                nodal_stride_y, nodal_stride_z, nodal_stride_n,
                grad_stride_x, grad_stride_y, grad_stride_z, grad_stride_q,
                grad_stride_d, hx, hy, hz, alpha, increment);
        }
    }

    void FEMGradientOperator3D::transpose_impl(
        const TypedFieldBase<Real> & gradient_field,
        TypedFieldBase<Real> & nodal_field, Real alpha, bool increment,
        const std::vector<Real> & weights) const {

        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        const Real * gradient = gradient_field.data();
        Real * nodal = nodal_field.data();

        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        Real hz = this->grid_spacing[2];

        std::vector<Real> quad_weights =
            weights.empty() ? this->get_quadrature_weights() : weights;

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];
        Index_t nz = nb_grid_pts[2];

        Index_t nb_nodes = this->get_nb_input_components();
        Index_t nodal_stride_c = 1;
        Index_t nodal_stride_n = nb_components;
        Index_t nodal_stride_x = nb_components * nb_nodes;
        Index_t nodal_stride_y = nb_components * nb_nodes * nx;
        Index_t nodal_stride_z = nb_components * nb_nodes * nx * ny;

        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_c = 1;
        Index_t grad_stride_d = nb_components;
        Index_t grad_stride_q = nb_components * dim;
        Index_t grad_stride_x = nb_components * dim * nb_quad;
        Index_t grad_stride_y = nb_components * dim * nb_quad * nx;
        Index_t grad_stride_z = nb_components * dim * nb_quad * nx * ny;

        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * gradient_comp = gradient + comp * grad_stride_c;
            Real * nodal_comp = nodal + comp * nodal_stride_c;

            fem_gradient_kernels::fem_divergence_3d_host(
                gradient_comp, nodal_comp, nx, ny, nz, grad_stride_x,
                grad_stride_y, grad_stride_z, grad_stride_q, grad_stride_d,
                nodal_stride_x, nodal_stride_y, nodal_stride_z,
                nodal_stride_n, hx, hy, hz, quad_weights.data(), alpha,
                increment);
        }
    }

    void
    FEMGradientOperator3D::apply(const TypedFieldBase<Real> & nodal_field,
                                 TypedFieldBase<Real> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator3D::apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void
    FEMGradientOperator3D::transpose(const TypedFieldBase<Real> & gradient_field,
                                     TypedFieldBase<Real> & nodal_field,
                                     const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator3D::transpose_increment(
        const TypedFieldBase<Real> & gradient_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    void FEMGradientOperator3D::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field, Real alpha,
        bool increment) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        const Real * nodal = nodal_field.view().data();
        Real * gradient = gradient_field.view().data();

        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        Real hz = this->grid_spacing[2];

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];
        Index_t nz = nb_grid_pts[2];

        // For SoA layout
        Index_t nodal_stride_x = 1;
        Index_t nodal_stride_y = nx;
        Index_t nodal_stride_z = nx * ny;
        Index_t nodal_stride_n = nx * ny * nz;
        Index_t nodal_stride_c = nx * ny * nz * this->get_nb_input_components();

        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_x = 1;
        Index_t grad_stride_y = nx;
        Index_t grad_stride_z = nx * ny;
        Index_t grad_stride_q = nx * ny * nz;
        Index_t grad_stride_d = nx * ny * nz * nb_quad;
        Index_t grad_stride_c = nx * ny * nz * nb_quad * dim;

        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * nodal_comp = nodal + comp * nodal_stride_c;
            Real * gradient_comp = gradient + comp * grad_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
            fem_gradient_kernels::fem_gradient_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
            fem_gradient_kernels::fem_gradient_3d_hip(
#endif
                nodal_comp, gradient_comp, nx, ny, nz, nodal_stride_x,
                nodal_stride_y, nodal_stride_z, nodal_stride_n,
                grad_stride_x, grad_stride_y, grad_stride_z, grad_stride_q,
                grad_stride_d, hx, hy, hz, alpha, increment);
        }
    }

    void FEMGradientOperator3D::transpose_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field, Real alpha,
        bool increment, const std::vector<Real> & weights) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        const Real * gradient = gradient_field.view().data();
        Real * nodal = nodal_field.view().data();

        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        Real hz = this->grid_spacing[2];

        std::vector<Real> quad_weights =
            weights.empty() ? this->get_quadrature_weights() : weights;

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];
        Index_t nz = nb_grid_pts[2];

        Index_t nodal_stride_x = 1;
        Index_t nodal_stride_y = nx;
        Index_t nodal_stride_z = nx * ny;
        Index_t nodal_stride_n = nx * ny * nz;
        Index_t nodal_stride_c = nx * ny * nz * this->get_nb_input_components();

        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_x = 1;
        Index_t grad_stride_y = nx;
        Index_t grad_stride_z = nx * ny;
        Index_t grad_stride_q = nx * ny * nz;
        Index_t grad_stride_d = nx * ny * nz * nb_quad;
        Index_t grad_stride_c = nx * ny * nz * nb_quad * dim;

        // Allocate device memory for weights
        Real * d_quad_weights = nullptr;
#if defined(MUGRID_ENABLE_CUDA)
        cudaMalloc(&d_quad_weights, quad_weights.size() * sizeof(Real));
        cudaMemcpy(d_quad_weights, quad_weights.data(),
                   quad_weights.size() * sizeof(Real),
                   cudaMemcpyHostToDevice);
#elif defined(MUGRID_ENABLE_HIP)
        (void)hipMalloc(&d_quad_weights, quad_weights.size() * sizeof(Real));
        (void)hipMemcpy(d_quad_weights, quad_weights.data(),
                        quad_weights.size() * sizeof(Real),
                        hipMemcpyHostToDevice);
#endif

        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * gradient_comp = gradient + comp * grad_stride_c;
            Real * nodal_comp = nodal + comp * nodal_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
            fem_gradient_kernels::fem_divergence_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
            fem_gradient_kernels::fem_divergence_3d_hip(
#endif
                gradient_comp, nodal_comp, nx, ny, nz, grad_stride_x,
                grad_stride_y, grad_stride_z, grad_stride_q, grad_stride_d,
                nodal_stride_x, nodal_stride_y, nodal_stride_z,
                nodal_stride_n, hx, hy, hz, d_quad_weights, alpha,
                increment);
        }

#if defined(MUGRID_ENABLE_CUDA)
        cudaFree(d_quad_weights);
#elif defined(MUGRID_ENABLE_HIP)
        (void)hipFree(d_quad_weights);
#endif
    }

    void FEMGradientOperator3D::apply(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator3D::apply_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void FEMGradientOperator3D::transpose(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator3D::transpose_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }
#endif

    // =========================================================================
    // 3D Kernel Implementations
    // =========================================================================

    namespace fem_gradient_kernels {

        void fem_gradient_3d_host(
            const Real * MUGRID_RESTRICT nodal_input,
            Real * MUGRID_RESTRICT gradient_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
            Index_t grad_stride_q, Index_t grad_stride_d, Real hx, Real hy,
            Real hz, Real alpha, bool increment) {

            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;

                        // Get all 8 nodal values
                        Real n[8];
                        for (Index_t node = 0; node < 8; ++node) {
                            Index_t ox = NODE_OFFSET_3D[node][0];
                            Index_t oy = NODE_OFFSET_3D[node][1];
                            Index_t oz = NODE_OFFSET_3D[node][2];
                            n[node] =
                                nodal_input[nodal_base + ox * nodal_stride_x +
                                            oy * nodal_stride_y +
                                            oz * nodal_stride_z];
                        }

                        // Compute gradients for each tetrahedron
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
                            for (Index_t node = 0; node < 8; ++node) {
                                grad_x += B_3D_REF[0][q][node] * n[node];
                                grad_y += B_3D_REF[1][q][node] * n[node];
                                grad_z += B_3D_REF[2][q][node] * n[node];
                            }
                            grad_x *= inv_hx;
                            grad_y *= inv_hy;
                            grad_z *= inv_hz;

                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            if (increment) {
                                gradient_output[grad_idx + 0 * grad_stride_d] +=
                                    grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] +=
                                    grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] +=
                                    grad_z;
                            } else {
                                gradient_output[grad_idx + 0 * grad_stride_d] =
                                    grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] =
                                    grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] =
                                    grad_z;
                            }
                        }
                    }
                }
            }
        }

        void fem_divergence_3d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t grad_stride_x, Index_t grad_stride_y,
            Index_t grad_stride_z, Index_t grad_stride_q, Index_t grad_stride_d,
            Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n, Real hx, Real hy,
            Real hz, const Real * quad_weights, Real alpha, bool increment) {

            // Initialize output if not incrementing
            if (!increment) {
                for (Index_t iz = 0; iz < nz; ++iz) {
                    for (Index_t iy = 0; iy < ny; ++iy) {
                        for (Index_t ix = 0; ix < nx; ++ix) {
                            nodal_output[ix * nodal_stride_x +
                                         iy * nodal_stride_y +
                                         iz * nodal_stride_z] = 0.0;
                        }
                    }
                }
            }

            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;

                        // For each quadrature point
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real w = quad_weights[q];
                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            Real gx =
                                gradient_input[grad_idx + 0 * grad_stride_d];
                            Real gy =
                                gradient_input[grad_idx + 1 * grad_stride_d];
                            Real gz =
                                gradient_input[grad_idx + 2 * grad_stride_d];

                            // Accumulate B^T * g to each node
                            for (Index_t node = 0; node < 8; ++node) {
                                Real contrib =
                                    w * (B_3D_REF[0][q][node] * inv_hx * gx +
                                         B_3D_REF[1][q][node] * inv_hy * gy +
                                         B_3D_REF[2][q][node] * inv_hz * gz);
                                Index_t ox = NODE_OFFSET_3D[node][0];
                                Index_t oy = NODE_OFFSET_3D[node][1];
                                Index_t oz = NODE_OFFSET_3D[node][2];
                                nodal_output[nodal_base + ox * nodal_stride_x +
                                             oy * nodal_stride_y +
                                             oz * nodal_stride_z] += contrib;
                            }
                        }
                    }
                }
            }
        }

    }  // namespace fem_gradient_kernels

}  // namespace muGrid
