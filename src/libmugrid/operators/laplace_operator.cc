/**
 * @file   laplace_operator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Dec 2024
 *
 * @brief  Host implementation of hard-coded Laplace operator
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

#include "laplace_operator.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

namespace muGrid {

    LaplaceOperator::LaplaceOperator(Index_t spatial_dim, Real scale)
        : Parent{}, spatial_dim{spatial_dim}, scale{scale} {
        if (spatial_dim != 2 && spatial_dim != 3) {
            throw RuntimeError("LaplaceOperator only supports 2D and 3D grids");
        }
    }

    const GlobalFieldCollection& LaplaceOperator::validate_fields(
        const Field &input_field,
        const Field &output_field) const {

        // Get field collections
        auto& input_collection = input_field.get_collection();
        auto& output_collection = output_field.get_collection();

        // Must be the same collection
        if (&input_collection != &output_collection) {
            throw RuntimeError("Input and output fields must belong to the "
                               "same field collection");
        }

        // Must be global field collection
        auto* global_fc = dynamic_cast<const GlobalFieldCollection*>(
            &input_collection);
        if (!global_fc) {
            throw RuntimeError("LaplaceOperator requires GlobalFieldCollection");
        }

        // Check dimension matches
        if (global_fc->get_spatial_dim() != this->spatial_dim) {
            throw RuntimeError("Field collection dimension (" +
                std::to_string(global_fc->get_spatial_dim()) +
                ") does not match operator dimension (" +
                std::to_string(this->spatial_dim) + ")");
        }

        // Check ghost layers (need at least 1 in each direction)
        auto left_ghosts = global_fc->get_nb_ghosts_left();
        auto right_ghosts = global_fc->get_nb_ghosts_right();

        for (Index_t d = 0; d < this->spatial_dim; ++d) {
            if (left_ghosts[d] < 1 || right_ghosts[d] < 1) {
                throw RuntimeError("LaplaceOperator requires at least 1 ghost "
                                   "layer in each direction");
            }
        }

        return *global_fc;
    }

    void LaplaceOperator::apply_impl(const TypedFieldBase<Real> &input_field,
                                      TypedFieldBase<Real> &output_field,
                                      Real alpha,
                                      bool increment) const {
        const auto& collection = this->validate_fields(input_field, output_field);

        // Get grid dimensions (with ghosts for input, without for output range)
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw data pointers
        const Real* input = input_field.data();
        Real* output = output_field.data();

        // Combine alpha with scale
        Real effective_scale = alpha * this->scale;

        if (this->spatial_dim == 2) {
            // For 2D with ArrayOfStructures layout, stride_x = 1, stride_y = nx
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            laplace_kernels::laplace_2d_host(
                input, output, nx, ny, 1, nx, effective_scale, increment);
        } else {
            // For 3D with ArrayOfStructures layout
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
            laplace_kernels::laplace_3d_host(
                input, output, nx, ny, nz, 1, nx, nx * ny, effective_scale, increment);
        }
    }

    void LaplaceOperator::apply(const TypedFieldBase<Real> &input_field,
                                 TypedFieldBase<Real> &output_field) const {
        this->apply_impl(input_field, output_field, 1.0, false);
    }

    void LaplaceOperator::apply_increment(const TypedFieldBase<Real> &input_field,
                                           const Real &alpha,
                                           TypedFieldBase<Real> &output_field) const {
        this->apply_impl(input_field, output_field, alpha, true);
    }

    void LaplaceOperator::transpose(const TypedFieldBase<Real> &input_field,
                                     TypedFieldBase<Real> &output_field,
                                     const std::vector<Real> &weights) const {
        // Laplacian is self-adjoint, so transpose = apply
        // weights are ignored (no quadrature weighting for Laplacian)
        (void)weights;
        this->apply(input_field, output_field);
    }

    void LaplaceOperator::transpose_increment(const TypedFieldBase<Real> &input_field,
                                               const Real &alpha,
                                               TypedFieldBase<Real> &output_field,
                                               const std::vector<Real> &weights) const {
        // Laplacian is self-adjoint, so transpose = apply
        (void)weights;
        this->apply_increment(input_field, alpha, output_field);
    }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    void LaplaceOperator::apply_impl(
        const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &output_field,
        Real alpha,
        bool increment) const {

        const auto& collection = this->validate_fields(input_field, output_field);

        // Get grid dimensions
        auto nb_grid_pts = collection.get_nb_subdomain_grid_pts_with_ghosts();

        // Get raw device data pointers via view()
        const Real* input = input_field.view().data();
        Real* output = output_field.view().data();

        // Combine alpha with scale
        Real effective_scale = alpha * this->scale;

        if (this->spatial_dim == 2) {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            // Device uses StructureOfArrays, but for scalar fields it's the same
#if defined(MUGRID_ENABLE_CUDA)
            laplace_kernels::laplace_2d_cuda(
                input, output, nx, ny, 1, nx, effective_scale, increment);
#elif defined(MUGRID_ENABLE_HIP)
            laplace_kernels::laplace_2d_hip(
                input, output, nx, ny, 1, nx, effective_scale, increment);
#endif
        } else {
            Index_t nx = nb_grid_pts[0];
            Index_t ny = nb_grid_pts[1];
            Index_t nz = nb_grid_pts[2];
#if defined(MUGRID_ENABLE_CUDA)
            laplace_kernels::laplace_3d_cuda(
                input, output, nx, ny, nz, 1, nx, nx * ny, effective_scale, increment);
#elif defined(MUGRID_ENABLE_HIP)
            laplace_kernels::laplace_3d_hip(
                input, output, nx, ny, nz, 1, nx, nx * ny, effective_scale, increment);
#endif
        }
    }

    void LaplaceOperator::apply(
        const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
        TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const {
        this->apply_impl(input_field, output_field, 1.0, false);
    }

    void LaplaceOperator::apply_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
        const Real &alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const {
        this->apply_impl(input_field, output_field, alpha, true);
    }
#endif

    namespace laplace_kernels {

        void laplace_2d_host(
            const Real* MUGRID_RESTRICT input,
            Real* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            Real scale,
            bool increment) {

            // Process interior points (excluding ghost layers)
            // Ghost layer is 1 pixel on each side
            for (Index_t iy = 1; iy < ny - 1; ++iy) {
                #if defined(_MSC_VER)
                #pragma loop(ivdep)
                #elif defined(__clang__)
                #pragma clang loop vectorize(enable) interleave(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (Index_t ix = 1; ix < nx - 1; ++ix) {
                    Index_t idx = ix * stride_x + iy * stride_y;

                    // 5-point stencil: [0,1,0; 1,-4,1; 0,1,0]
                    Real center = input[idx];
                    Real left   = input[idx - stride_x];
                    Real right  = input[idx + stride_x];
                    Real down   = input[idx - stride_y];
                    Real up     = input[idx + stride_y];

                    Real result = scale * (left + right + down + up - 4.0 * center);
                    if (increment) {
                        output[idx] += result;
                    } else {
                        output[idx] = result;
                    }
                }
            }
        }

        void laplace_3d_host(
            const Real* MUGRID_RESTRICT input,
            Real* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale,
            bool increment) {

            // Process interior points (excluding ghost layers)
            for (Index_t iz = 1; iz < nz - 1; ++iz) {
                for (Index_t iy = 1; iy < ny - 1; ++iy) {
                    #if defined(_MSC_VER)
                    #pragma loop(ivdep)
                    #elif defined(__clang__)
                    #pragma clang loop vectorize(enable) interleave(enable)
                    #elif defined(__GNUC__)
                    #pragma GCC ivdep
                    #endif
                    for (Index_t ix = 1; ix < nx - 1; ++ix) {
                        Index_t idx = ix * stride_x + iy * stride_y + iz * stride_z;

                        // 7-point stencil: center=-6, neighbors=+1
                        Real center = input[idx];
                        Real xm = input[idx - stride_x];
                        Real xp = input[idx + stride_x];
                        Real ym = input[idx - stride_y];
                        Real yp = input[idx + stride_y];
                        Real zm = input[idx - stride_z];
                        Real zp = input[idx + stride_z];

                        Real result = scale * (xm + xp + ym + yp + zm + zp - 6.0 * center);
                        if (increment) {
                            output[idx] += result;
                        } else {
                            output[idx] = result;
                        }
                    }
                }
            }
        }

    }  // namespace laplace_kernels

}  // namespace muGrid
