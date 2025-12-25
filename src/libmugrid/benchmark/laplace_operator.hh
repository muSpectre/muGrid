/**
 * @file   laplace_operator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Dec 2024
 *
 * @brief  Hard-coded Laplace operators for benchmarking purposes
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

#ifndef SRC_LIBMUGRID_BENCHMARK_LAPLACE_OPERATOR_HH_
#define SRC_LIBMUGRID_BENCHMARK_LAPLACE_OPERATOR_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    /**
     * @class LaplaceOperator
     * @brief Hard-coded Laplace operator for benchmarking purposes.
     *
     * This class provides optimized implementations of the discrete Laplace
     * operator using:
     * - 5-point stencil for 2D grids: [0,1,0; 1,-4,1; 0,1,0]
     * - 7-point stencil for 3D grids: center=-6, neighbors=+1
     *
     * The output is multiplied by a scale factor, which can be used to
     * incorporate grid spacing and sign conventions (e.g., for making
     * the operator positive-definite).
     *
     * The implementation is designed for benchmarking and performance
     * comparison with the generic sparse convolution operator.
     */
    class LaplaceOperator {
    public:
        /**
         * @brief Construct a Laplace operator for the given dimension.
         * @param spatial_dim Spatial dimension (2 or 3)
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        explicit LaplaceOperator(Index_t spatial_dim, Real scale = 1.0);

        //! Default constructor is deleted
        LaplaceOperator() = delete;

        //! Copy constructor is deleted
        LaplaceOperator(const LaplaceOperator &other) = delete;

        //! Move constructor
        LaplaceOperator(LaplaceOperator &&other) = default;

        //! Destructor
        ~LaplaceOperator() = default;

        //! Copy assignment operator is deleted
        LaplaceOperator &operator=(const LaplaceOperator &other) = delete;

        //! Move assignment operator
        LaplaceOperator &operator=(LaplaceOperator &&other) = default;

        /**
         * @brief Apply the Laplace operator on host memory fields.
         *
         * Computes output = Laplace(input) using the appropriate stencil
         * (5-point for 2D, 7-point for 3D).
         *
         * @param input_field Input field (with ghost layers populated)
         * @param output_field Output field
         */
        void apply(const TypedFieldBase<Real> &input_field,
                   TypedFieldBase<Real> &output_field) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Apply the Laplace operator on device memory fields.
         *
         * @param input_field Input field in device memory
         * @param output_field Output field in device memory
         */
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const;
#endif

        /**
         * @brief Get the spatial dimension.
         * @return Spatial dimension (2 or 3)
         */
        Index_t get_spatial_dim() const { return spatial_dim; }

        /**
         * @brief Get the number of stencil points.
         * @return Number of stencil points (5 for 2D, 7 for 3D)
         */
        Index_t get_nb_stencil_pts() const {
            return spatial_dim == 2 ? 5 : 7;
        }

        /**
         * @brief Get the scale factor.
         * @return Scale factor applied to output
         */
        Real get_scale() const { return scale; }

    private:
        Index_t spatial_dim;
        Real scale;

        /**
         * @brief Validate that fields are compatible with this operator.
         * @param input_field The input field
         * @param output_field The output field
         * @return Reference to the GlobalFieldCollection
         * @throws RuntimeError if validation fails
         */
        const GlobalFieldCollection& validate_fields(
            const Field &input_field,
            const Field &output_field) const;
    };

    // Host kernel implementations (inline for header-only use)
    namespace benchmark_kernels {

        /**
         * @brief Apply 5-point 2D Laplace stencil on host.
         *
         * Stencil: scale * [0, 1, 0]
         *                  [1,-4, 1]
         *                  [0, 1, 0]
         */
        void laplace_2d_host(
            const Real* __restrict__ input,
            Real* __restrict__ output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            Real scale);

        /**
         * @brief Apply 7-point 3D Laplace stencil on host.
         *
         * Stencil: scale * (center = -6, each of 6 neighbors = +1)
         */
        void laplace_3d_host(
            const Real* __restrict__ input,
            Real* __restrict__ output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale);

#if defined(MUGRID_ENABLE_CUDA)
        /**
         * @brief Apply 2D Laplace stencil on CUDA device.
         */
        void laplace_2d_cuda(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            Real scale);

        /**
         * @brief Apply 3D Laplace stencil on CUDA device.
         */
        void laplace_3d_cuda(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale);
#endif

#if defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Apply 2D Laplace stencil on HIP device.
         */
        void laplace_2d_hip(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            Real scale);

        /**
         * @brief Apply 3D Laplace stencil on HIP device.
         */
        void laplace_3d_hip(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale);
#endif

    }  // namespace benchmark_kernels

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_BENCHMARK_LAPLACE_OPERATOR_HH_
