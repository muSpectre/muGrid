/**
 * @file   laplace_operator_3d.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Hard-coded 3D Laplace operator with optimized stencil implementation
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

#ifndef SRC_LIBMUGRID_OPERATORS_LAPLACE_3D_HH_
#define SRC_LIBMUGRID_OPERATORS_LAPLACE_3D_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    /**
     * @class LaplaceOperator3D
     * @brief Hard-coded 3D Laplace operator with optimized 7-point stencil.
     *
     * This class provides an optimized implementation of the discrete Laplace
     * operator using a 7-point stencil for 3D grids: center=-6, neighbors=+1
     *
     * The output is multiplied by a scale factor, which can be used to
     * incorporate grid spacing and sign conventions (e.g., for making
     * the operator positive-definite for use with CG solvers).
     *
     * This operator inherits from GradientOperator and can be used
     * interchangeably with other gradient operators. The hard-coded
     * implementation provides significantly better performance (~3-10x) due
     * to compile-time known memory access patterns that enable SIMD
     * vectorization.
     *
     * Since the Laplacian is self-adjoint (symmetric), the transpose operation
     * is identical to the forward apply operation.
     */
    class LaplaceOperator3D : public LinearOperator {
    public:
        using Parent = LinearOperator;

        //! Number of stencil points (compile-time constant for 3D)
        static constexpr Index_t NB_STENCIL_PTS = 7;

        /**
         * @brief Construct a 3D Laplace operator.
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        explicit LaplaceOperator3D(Real scale = 1.0);

        //! Default constructor is deleted
        LaplaceOperator3D() = delete;

        //! Copy constructor is deleted
        LaplaceOperator3D(const LaplaceOperator3D &other) = delete;

        //! Move constructor
        LaplaceOperator3D(LaplaceOperator3D &&other) = default;

        //! Destructor
        ~LaplaceOperator3D() override = default;

        //! Copy assignment operator is deleted
        LaplaceOperator3D &operator=(const LaplaceOperator3D &other) = delete;

        //! Move assignment operator
        LaplaceOperator3D &operator=(LaplaceOperator3D &&other) = default;

        /**
         * @brief Apply the Laplace operator on host memory fields.
         *
         * Computes output = scale * Laplace(input) using the 7-point stencil.
         *
         * @param input_field Input field (with ghost layers populated)
         * @param output_field Output field
         */
        void apply(const TypedFieldBase<Real> &input_field,
                   TypedFieldBase<Real> &output_field) const override;

        /**
         * @brief Apply the Laplace operator with increment.
         *
         * Computes output += alpha * scale * Laplace(input)
         *
         * @param input_field Input field (with ghost layers populated)
         * @param alpha Scaling factor for the increment
         * @param output_field Output field to increment
         */
        void apply_increment(const TypedFieldBase<Real> &input_field,
                             const Real &alpha,
                             TypedFieldBase<Real> &output_field) const override;

        /**
         * @brief Apply the transpose (same as apply for symmetric Laplacian).
         *
         * Since the Laplacian is self-adjoint, transpose equals apply.
         *
         * @param input_field Input field (quadrature point field in base class terms)
         * @param output_field Output field (nodal field in base class terms)
         * @param weights Ignored for Laplacian (no quadrature weighting)
         */
        void transpose(const TypedFieldBase<Real> &input_field,
                       TypedFieldBase<Real> &output_field,
                       const std::vector<Real> &weights = {}) const override;

        /**
         * @brief Apply the transpose with increment (same as apply_increment).
         *
         * @param input_field Input field
         * @param alpha Scaling factor for the increment
         * @param output_field Output field to increment
         * @param weights Ignored for Laplacian
         */
        void transpose_increment(const TypedFieldBase<Real> &input_field,
                                 const Real &alpha,
                                 TypedFieldBase<Real> &output_field,
                                 const std::vector<Real> &weights = {}) const override;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Apply the Laplace operator on device memory fields.
         *
         * @param input_field Input field in device memory
         * @param output_field Output field in device memory
         */
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const;

        /**
         * @brief Apply the Laplace operator with increment on device memory.
         *
         * @param input_field Input field in device memory
         * @param alpha Scaling factor for the increment
         * @param output_field Output field in device memory
         */
        void apply_increment(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                             const Real &alpha,
                             TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const;
#endif

        /**
         * @brief Get the number of output components (always 1 for Laplacian).
         * @return 1
         */
        Index_t get_nb_output_components() const override { return 1; }

        /**
         * @brief Get the number of quadrature points (always 1 for Laplacian).
         * @return 1
         */
        Index_t get_nb_quad_pts() const override { return 1; }

        /**
         * @brief Get the number of input components (always 1 for Laplacian).
         * @return 1
         */
        Index_t get_nb_input_components() const override { return 1; }

        /**
         * @brief Get the spatial dimension.
         * @return 3
         */
        Dim_t get_spatial_dim() const override { return 3; }

        /**
         * @brief Get the number of stencil points.
         * @return 7
         */
        Index_t get_nb_stencil_pts() const { return NB_STENCIL_PTS; }

        /**
         * @brief Get the scale factor.
         * @return Scale factor applied to output
         */
        Real get_scale() const { return scale; }

        /**
         * @brief Get the stencil offset.
         * @return Stencil offset in pixels (centered: [-1,-1,-1])
         */
        Shape_t get_offset() const { return Shape_t{-1, -1, -1}; }

        /**
         * @brief Get the stencil shape.
         * @return Shape of the stencil ([3,3,3])
         */
        Shape_t get_stencil_shape() const { return Shape_t{3, 3, 3}; }

        /**
         * @brief Get the stencil coefficients in reshaped form.
         * @return Vector of stencil coefficients
         *
         * Returns 7-point stencil with center=-6*scale, neighbors=1*scale
         */
        std::vector<Real> get_coefficients() const {
            std::vector<Real> stencil(27, 0.0);
            // Center point at [1,1,1] = index 13
            stencil[13] = -6.0 * scale;
            // 6 neighbors
            stencil[13-1] = scale;   // [1,1,0]
            stencil[13+1] = scale;   // [1,1,2]
            stencil[13-3] = scale;   // [1,0,1]
            stencil[13+3] = scale;   // [1,2,1]
            stencil[13-9] = scale;   // [0,1,1]
            stencil[13+9] = scale;   // [2,1,1]
            return stencil;
        }

    private:
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

        /**
         * @brief Internal implementation of apply with optional increment.
         * @param input_field Input field
         * @param output_field Output field
         * @param alpha Scaling factor (0 means overwrite, non-zero means increment)
         * @param increment If true, add to output; if false, overwrite output
         */
        void apply_impl(const TypedFieldBase<Real> &input_field,
                        TypedFieldBase<Real> &output_field,
                        Real alpha,
                        bool increment) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Internal device implementation of apply with optional increment.
         */
        void apply_impl(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                        TypedFieldBase<Real, DefaultDeviceSpace> &output_field,
                        Real alpha,
                        bool increment) const;
#endif
    };

    // Kernel implementations
    namespace laplace_kernels {

        /**
         * @brief Apply 7-point 3D Laplace stencil on host.
         *
         * Stencil: scale * (center = -6, each of 6 neighbors = +1)
         *
         * @param input Input array
         * @param output Output array
         * @param nx Grid size in x (including ghosts)
         * @param ny Grid size in y (including ghosts)
         * @param nz Grid size in z (including ghosts)
         * @param stride_x Stride in x direction
         * @param stride_y Stride in y direction
         * @param stride_z Stride in z direction
         * @param scale Scale factor
         * @param increment If true, add to output; if false, overwrite
         */
        void laplace_3d_host(
            const Real* MUGRID_RESTRICT input,
            Real* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale,
            bool increment = false);

#if defined(MUGRID_ENABLE_CUDA)
        /**
         * @brief Apply 3D Laplace stencil on CUDA device.
         */
        void laplace_3d_cuda(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale,
            bool increment = false);
#endif

#if defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Apply 3D Laplace stencil on HIP device.
         */
        void laplace_3d_hip(
            const Real* input,
            Real* output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale,
            bool increment = false);
#endif

    }  // namespace laplace_kernels

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_LAPLACE_3D_HH_
