/**
 * @file   fem_gradient_operator_2d.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Host implementation of hard-coded 2D linear FEM gradient operator
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

#include "fem_gradient_2d.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

    // =========================================================================
    // FEMGradientOperator2D Implementation
    // =========================================================================

    FEMGradientOperator2D::FEMGradientOperator2D(std::vector<Real> grid_spacing)
        : Parent{}, grid_spacing{std::move(grid_spacing)} {
        // Default grid spacing is 1.0 in each direction
        if (this->grid_spacing.empty()) {
            this->grid_spacing.resize(DIM, 1.0);
        }
        if (static_cast<Index_t>(this->grid_spacing.size()) != DIM) {
            throw RuntimeError("Grid spacing must have 2 components for 2D operator");
        }
    }

    const GlobalFieldCollection &
    FEMGradientOperator2D::validate_fields(const Field & nodal_field,
                                           const Field & gradient_field,
                                           Index_t & nb_components) const {

        // Get field collections
        auto & nodal_collection = nodal_field.get_collection();
        auto & gradient_collection = gradient_field.get_collection();

        // Must be the same collection
        if (&nodal_collection != &gradient_collection) {
            throw RuntimeError("Nodal and gradient fields must belong to the "
                               "same field collection");
        }

        // Must be global field collection
        auto * global_fc =
            dynamic_cast<const GlobalFieldCollection *>(&nodal_collection);
        if (!global_fc) {
            throw RuntimeError(
                "FEMGradientOperator2D requires GlobalFieldCollection");
        }

        // Check dimension matches
        if (global_fc->get_spatial_dim() != DIM) {
            throw RuntimeError("Field collection dimension (" +
                               std::to_string(global_fc->get_spatial_dim()) +
                               ") does not match operator dimension (2)");
        }

        // Get and validate component counts
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

    std::vector<Real> FEMGradientOperator2D::get_quadrature_weights() const {
        // Each triangle has area = 0.5 * hx * hy
        // Total pixel area = hx * hy
        // Weight = triangle_area / pixel_area = 0.5
        Real pixel_area = this->grid_spacing[0] * this->grid_spacing[1];
        return {0.5 * pixel_area, 0.5 * pixel_area};
    }

    std::vector<Real> FEMGradientOperator2D::get_coefficients() const {
        using namespace fem_gradient_kernels;

        // Shape: (nb_operators=2, nb_quad_pts=2, nb_nodal_pts=1, 2, 2)
        // Total size: 2 * 2 * 1 * 2 * 2 = 16
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];
        std::vector<Real> result;
        result.reserve(16);

        // Fortran-order: first index (operators) varies fastest
        // Iterate: stencil_y, stencil_x, nodal_pts (=1), quad_pts, operators
        for (Index_t j = 0; j < 2; ++j) {              // stencil_y (outermost)
            for (Index_t i = 0; i < 2; ++i) {          // stencil_x
                // nodal_pts = 1, skip loop
                for (Index_t q = 0; q < NB_QUAD_2D; ++q) {  // quad pt
                    for (Index_t d = 0; d < DIM_2D; ++d) {  // operator (innermost)
                        // Map (i,j) to node index: node = j*2 + i
                        Index_t node = j * 2 + i;
                        Real grad_val = B_2D_REF[d][q][node];
                        // Scale by inverse grid spacing
                        grad_val *= (d == 0) ? (1.0 / hx) : (1.0 / hy);
                        result.push_back(grad_val);
                    }
                }
            }
        }
        return result;
    }

    void
    FEMGradientOperator2D::apply_impl(const TypedFieldBase<Real> & nodal_field,
                                      TypedFieldBase<Real> & gradient_field,
                                      Real alpha, bool increment) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        // Get grid dimensions (with ghosts)
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw data pointers
        const Real * nodal = nodal_field.data();
        Real * gradient = gradient_field.data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];

        // For continuous FEM, the nodal field has shape [components, sub_pts, nx, ny]
        // Field layout (column-major/AoS): components vary FASTEST, then sub_pts,
        // then spatial dimensions.
        Index_t nb_sub = this->get_nb_input_components();  // 1
        Index_t nodal_stride_c = 1;    // components are innermost
        Index_t nodal_stride_n = nb_components;
        Index_t nodal_stride_x = nb_components * nb_sub;
        Index_t nodal_stride_y = nb_components * nb_sub * nx;

        // For gradient field [components, operators, nb_quad, x, y]:
        // Column-major (AoS): components vary fastest
        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_c = 1;
        Index_t grad_stride_d = nb_components;
        Index_t grad_stride_q = nb_components * dim;
        Index_t grad_stride_x = nb_components * dim * nb_quad;
        Index_t grad_stride_y = nb_components * dim * nb_quad * nx;

        // Process each component independently
        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * nodal_comp = nodal + comp * nodal_stride_c;
            Real * gradient_comp = gradient + comp * grad_stride_c;

            fem_gradient_kernels::fem_gradient_2d_host(
                nodal_comp, gradient_comp, nx, ny, nodal_stride_x,
                nodal_stride_y, nodal_stride_n, grad_stride_x,
                grad_stride_y, grad_stride_q, grad_stride_d, hx, hy, alpha,
                increment);
        }
    }

    void FEMGradientOperator2D::transpose_impl(
        const TypedFieldBase<Real> & gradient_field,
        TypedFieldBase<Real> & nodal_field, Real alpha, bool increment,
        const std::vector<Real> & weights) const {

        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        // Get grid dimensions
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw data pointers
        const Real * gradient = gradient_field.data();
        Real * nodal = nodal_field.data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        // Get quadrature weights
        std::vector<Real> quad_weights =
            weights.empty() ? this->get_quadrature_weights() : weights;

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];

        // Nodal field strides [components, sub_pts, x, y]
        Index_t nb_nodes = this->get_nb_input_components();
        Index_t nodal_stride_c = 1;
        Index_t nodal_stride_n = nb_components;
        Index_t nodal_stride_x = nb_components * nb_nodes;
        Index_t nodal_stride_y = nb_components * nb_nodes * nx;

        // Gradient field strides [components, operators, nb_quad, x, y]
        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_c = 1;
        Index_t grad_stride_d = nb_components;
        Index_t grad_stride_q = nb_components * dim;
        Index_t grad_stride_x = nb_components * dim * nb_quad;
        Index_t grad_stride_y = nb_components * dim * nb_quad * nx;

        // Process each component independently
        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * gradient_comp = gradient + comp * grad_stride_c;
            Real * nodal_comp = nodal + comp * nodal_stride_c;

            fem_gradient_kernels::fem_divergence_2d_host(
                gradient_comp, nodal_comp, nx, ny, grad_stride_x,
                grad_stride_y, grad_stride_q, grad_stride_d, nodal_stride_x,
                nodal_stride_y, nodal_stride_n, hx, hy, quad_weights.data(),
                alpha, increment);
        }
    }

    void
    FEMGradientOperator2D::apply(const TypedFieldBase<Real> & nodal_field,
                                 TypedFieldBase<Real> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator2D::apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void
    FEMGradientOperator2D::transpose(const TypedFieldBase<Real> & gradient_field,
                                     TypedFieldBase<Real> & nodal_field,
                                     const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator2D::transpose_increment(
        const TypedFieldBase<Real> & gradient_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    void FEMGradientOperator2D::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field, Real alpha,
        bool increment) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        // Get grid dimensions (with ghosts)
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw device data pointers via view()
        const Real * nodal = nodal_field.view().data();
        Real * gradient = gradient_field.view().data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];

        // For SoA: nodal field [components, sub_pts, x, y]
        // Memory: x fastest, then y, then sub_pts, then components
        Index_t nodal_stride_x = 1;
        Index_t nodal_stride_y = nx;
        Index_t nodal_stride_n = nx * ny;
        Index_t nodal_stride_c = nx * ny * this->get_nb_input_components();

        // For SoA: gradient field [components, operators, nb_quad, x, y]
        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_x = 1;
        Index_t grad_stride_y = nx;
        Index_t grad_stride_q = nx * ny;
        Index_t grad_stride_d = nx * ny * nb_quad;
        Index_t grad_stride_c = nx * ny * nb_quad * dim;

        // Process each component independently
        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * nodal_comp = nodal + comp * nodal_stride_c;
            Real * gradient_comp = gradient + comp * grad_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
            fem_gradient_kernels::fem_gradient_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
            fem_gradient_kernels::fem_gradient_2d_hip(
#endif
                nodal_comp, gradient_comp, nx, ny, nodal_stride_x,
                nodal_stride_y, nodal_stride_n, grad_stride_x,
                grad_stride_y, grad_stride_q, grad_stride_d, hx, hy, alpha,
                increment);
        }
    }

    void FEMGradientOperator2D::transpose_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field, Real alpha,
        bool increment, const std::vector<Real> & weights) const {
        Index_t nb_components;
        const auto & collection =
            this->validate_fields(nodal_field, gradient_field, nb_components);

        // Get grid dimensions
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw device data pointers via view()
        const Real * gradient = gradient_field.view().data();
        Real * nodal = nodal_field.view().data();

        // Grid spacing
        Real hx = this->grid_spacing[0];
        Real hy = this->grid_spacing[1];

        // Get quadrature weights (host memory)
        std::vector<Real> quad_weights =
            weights.empty() ? this->get_quadrature_weights() : weights;

        Index_t nx = nb_grid_pts[0];
        Index_t ny = nb_grid_pts[1];

        // For SoA: nodal field [components, sub_pts, x, y]
        Index_t nodal_stride_x = 1;
        Index_t nodal_stride_y = nx;
        Index_t nodal_stride_n = nx * ny;
        Index_t nodal_stride_c = nx * ny * this->get_nb_input_components();

        // For SoA: gradient field [components, operators, nb_quad, x, y]
        Index_t dim = DIM;
        Index_t nb_quad = NB_QUAD;
        Index_t grad_stride_x = 1;
        Index_t grad_stride_y = nx;
        Index_t grad_stride_q = nx * ny;
        Index_t grad_stride_d = nx * ny * nb_quad;
        Index_t grad_stride_c = nx * ny * nb_quad * dim;

        // Process each component independently
        for (Index_t comp = 0; comp < nb_components; ++comp) {
            const Real * gradient_comp = gradient + comp * grad_stride_c;
            Real * nodal_comp = nodal + comp * nodal_stride_c;

#if defined(MUGRID_ENABLE_CUDA)
            fem_gradient_kernels::fem_divergence_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
            fem_gradient_kernels::fem_divergence_2d_hip(
#endif
                gradient_comp, nodal_comp, nx, ny, grad_stride_x,
                grad_stride_y, grad_stride_q, grad_stride_d, nodal_stride_x,
                nodal_stride_y, nodal_stride_n, hx, hy, quad_weights.data(),
                alpha, increment);
        }
    }

    void FEMGradientOperator2D::apply(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, 1.0, false);
    }

    void FEMGradientOperator2D::apply_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
        this->apply_impl(nodal_field, gradient_field, alpha, true);
    }

    void FEMGradientOperator2D::transpose(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, 1.0, false, weights);
    }

    void FEMGradientOperator2D::transpose_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        this->transpose_impl(gradient_field, nodal_field, alpha, true, weights);
    }
#endif

    // =========================================================================
    // 2D Kernel Implementations
    // =========================================================================

    namespace fem_gradient_kernels {

        void fem_gradient_2d_host(const Real * MUGRID_RESTRICT nodal_input,
                                  Real * MUGRID_RESTRICT gradient_output,
                                  Index_t nx, Index_t ny,
                                  Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment) {

            // Scale factors for shape function gradients
            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;

            // Process interior points (excluding last row/column which need
            // nodes from the next pixel)
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    // Base indices for this pixel
                    Index_t nodal_base =
                        ix * nodal_stride_x + iy * nodal_stride_y;
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;

                    // Get nodal values at pixel corners
                    // Node 0: (ix, iy), Node 1: (ix+1, iy)
                    // Node 2: (ix, iy+1), Node 3: (ix+1, iy+1)
                    Real n0 = nodal_input[nodal_base + 0 * nodal_stride_n];
                    Real n1 = nodal_input[nodal_base + nodal_stride_x +
                                          0 * nodal_stride_n];
                    Real n2 = nodal_input[nodal_base + nodal_stride_y +
                                          0 * nodal_stride_n];
                    Real n3 = nodal_input[nodal_base + nodal_stride_x +
                                          nodal_stride_y + 0 * nodal_stride_n];

                    // Triangle 0 (lower-left): nodes 0, 1, 2
                    // grad_x = (-n0 + n1) / hx
                    // grad_y = (-n0 + n2) / hy
                    Real grad_x_t0 = inv_hx * (-n0 + n1);
                    Real grad_y_t0 = inv_hy * (-n0 + n2);

                    // Triangle 1 (upper-right): nodes 1, 3, 2
                    // grad_x = (-n2 + n3) / hx
                    // grad_y = (-n1 + n3) / hy
                    Real grad_x_t1 = inv_hx * (-n2 + n3);
                    Real grad_y_t1 = inv_hy * (-n1 + n3);

                    // Store gradients
                    if (increment) {
                        // Quad 0 (Triangle 0)
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        0 * grad_stride_q] += grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        0 * grad_stride_q] += grad_y_t0;
                        // Quad 1 (Triangle 1)
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        1 * grad_stride_q] += grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        1 * grad_stride_q] += grad_y_t1;
                    } else {
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        0 * grad_stride_q] = grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        0 * grad_stride_q] = grad_y_t0;
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        1 * grad_stride_q] = grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        1 * grad_stride_q] = grad_y_t1;
                    }
                }
            }
        }

        void fem_divergence_2d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_n, Real hx, Real hy,
            const Real * quad_weights, Real alpha, bool increment) {

            // Scale factors including quadrature weights
            const Real w0_inv_hx = alpha * quad_weights[0] / hx;
            const Real w0_inv_hy = alpha * quad_weights[0] / hy;
            const Real w1_inv_hx = alpha * quad_weights[1] / hx;
            const Real w1_inv_hy = alpha * quad_weights[1] / hy;

            // Initialize output if not incrementing
            if (!increment) {
                for (Index_t iy = 0; iy < ny; ++iy) {
                    for (Index_t ix = 0; ix < nx; ++ix) {
                        nodal_output[ix * nodal_stride_x +
                                     iy * nodal_stride_y] = 0.0;
                    }
                }
            }

            // The transpose accumulates contributions from all quadrature
            // points to the nodal points.
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;
                    Index_t nodal_base =
                        ix * nodal_stride_x + iy * nodal_stride_y;

                    // Get gradient values at quadrature points
                    Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d +
                                                0 * grad_stride_q];
                    Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d +
                                                0 * grad_stride_q];
                    Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d +
                                                1 * grad_stride_q];
                    Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d +
                                                1 * grad_stride_q];

                    // Triangle 0 contributions: B^T * sigma
                    Real contrib_n0_t0 =
                        w0_inv_hx * (-gx_t0) + w0_inv_hy * (-gy_t0);
                    Real contrib_n1_t0 = w0_inv_hx * (gx_t0);
                    Real contrib_n2_t0 = w0_inv_hy * (gy_t0);

                    // Triangle 1 contributions
                    Real contrib_n1_t1 = w1_inv_hy * (-gy_t1);
                    Real contrib_n2_t1 = w1_inv_hx * (-gx_t1);
                    Real contrib_n3_t1 =
                        w1_inv_hx * (gx_t1) + w1_inv_hy * (gy_t1);

                    // Accumulate to nodal points
                    nodal_output[nodal_base] += contrib_n0_t0;
                    nodal_output[nodal_base + nodal_stride_x] +=
                        contrib_n1_t0 + contrib_n1_t1;
                    nodal_output[nodal_base + nodal_stride_y] +=
                        contrib_n2_t0 + contrib_n2_t1;
                    nodal_output[nodal_base + nodal_stride_x +
                                 nodal_stride_y] += contrib_n3_t1;
                }
            }
        }

    }  // namespace fem_gradient_kernels

}  // namespace muGrid
