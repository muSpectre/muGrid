/**
 * @file   fem_gradient_operator_2d.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Hard-coded 2D linear FEM gradient operator
 *
 * This class provides optimized implementations of the gradient operator
 * for linear finite elements on 2D structured grids:
 * - 4 nodal points per pixel (corners at [0,0], [1,0], [0,1], [1,1])
 * - 2 triangles per pixel (lower-left and upper-right)
 * - 2 quadrature points (one per triangle, at centroid)
 * - 2 gradient components (d/dx, d/dy)
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_2D_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_2D_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"

#include <array>
#include <vector>

namespace muGrid {

    // Forward declaration
    class GlobalFieldCollection;

    /**
     * @class FEMGradientOperator2D
     * @brief Hard-coded 2D linear FEM gradient operator with optimized
     * implementation.
     *
     * This class provides optimized implementations of the gradient operator
     * for linear finite elements on 2D structured grids:
     *
     * **2D (Linear Triangles):**
     * - 4 nodal points per pixel (corners at [0,0], [1,0], [0,1], [1,1])
     * - 2 triangles per pixel (lower-left and upper-right)
     * - 2 quadrature points (one per triangle, at centroid)
     * - 2 gradient components (d/dx, d/dy)
     *
     * The apply() method computes the gradient (nodal → quadrature points).
     * The transpose() method computes the divergence (quadrature → nodal
     * points).
     *
     * Shape function gradients are compile-time constants for linear elements,
     * enabling SIMD vectorization and optimal performance.
     */
    class FEMGradientOperator2D : public LinearOperator {
       public:
        using Parent = LinearOperator;

        //! Number of nodes per pixel (compile-time constant for 2D)
        static constexpr Index_t NB_NODES = 4;

        //! Number of quadrature points per pixel (compile-time constant for 2D)
        static constexpr Index_t NB_QUAD = 2;

        //! Spatial dimension
        static constexpr Dim_t DIM = 2;

        /**
         * @brief Construct a 2D FEM gradient operator.
         * @param grid_spacing Grid spacing in each direction (default: [1.0, 1.0])
         *
         * The grid spacing is used to scale the shape function gradients.
         */
        explicit FEMGradientOperator2D(std::vector<Real> grid_spacing = {});

        //! Default constructor is deleted
        FEMGradientOperator2D() = delete;

        //! Copy constructor is deleted
        FEMGradientOperator2D(const FEMGradientOperator2D & other) = delete;

        //! Move constructor
        FEMGradientOperator2D(FEMGradientOperator2D && other) = default;

        //! Destructor
        ~FEMGradientOperator2D() override = default;

        //! Copy assignment operator is deleted
        FEMGradientOperator2D &
        operator=(const FEMGradientOperator2D & other) = delete;

        //! Move assignment operator
        FEMGradientOperator2D & operator=(FEMGradientOperator2D && other) = default;

        /**
         * @brief Apply the gradient operator (nodal → quadrature).
         *
         * Input field: nodal values with shape [nb_nodal_pts, ...]
         * Output field: gradient at quadrature points with shape [dim,
         * nb_quad_pts, ...]
         *
         * @param nodal_field Input field at nodal points
         * @param gradient_field Output gradient field at quadrature points
         */
        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & gradient_field) const override;

        /**
         * @brief Apply the gradient operator with increment.
         *
         * Computes: gradient_field += alpha * grad(nodal_field)
         *
         * @param nodal_field Input field at nodal points
         * @param alpha Scaling factor for the increment
         * @param gradient_field Output gradient field to increment
         */
        void
        apply_increment(const TypedFieldBase<Real> & nodal_field,
                        const Real & alpha,
                        TypedFieldBase<Real> & gradient_field) const override;

        /**
         * @brief Apply the transpose (divergence) operator (quadrature →
         * nodal).
         *
         * Computes: nodal_field = -div(gradient_field)
         *           (negative divergence for consistency with weak form)
         *
         * @param gradient_field Input gradient field at quadrature points
         * @param nodal_field Output field at nodal points
         * @param weights Quadrature weights (optional, default: equal weights)
         */
        void transpose(const TypedFieldBase<Real> & gradient_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const override;

        /**
         * @brief Apply the transpose (divergence) with increment.
         *
         * Computes: nodal_field += alpha * (-div(gradient_field))
         *
         * @param gradient_field Input gradient field at quadrature points
         * @param alpha Scaling factor for the increment
         * @param nodal_field Output field at nodal points to increment
         * @param weights Quadrature weights (optional)
         */
        void transpose_increment(
            const TypedFieldBase<Real> & gradient_field, const Real & alpha,
            TypedFieldBase<Real> & nodal_field,
            const std::vector<Real> & weights = {}) const override;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Apply the gradient operator on device memory fields.
         */
        void
        apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
              TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const;

        /**
         * @brief Apply the gradient operator with increment on device memory.
         */
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const;

        /**
         * @brief Apply the transpose on device memory.
         */
        void transpose(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;

        /**
         * @brief Apply the transpose with increment on device memory.
         */
        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;
#endif

        /**
         * @brief Get the number of output components (2 for 2D gradient).
         * @return 2 (d/dx, d/dy)
         */
        Index_t get_nb_output_components() const override { return DIM; }

        /**
         * @brief Get the number of quadrature points per pixel.
         * @return 2 (one per triangle)
         */
        Index_t get_nb_quad_pts() const override { return NB_QUAD; }

        /**
         * @brief Get the number of input components per pixel.
         *
         * For continuous FEM, nodes are shared between pixels via ghost
         * communication. Each pixel uses values from 4 grid points, but the
         * field itself has only 1 value per grid point.
         *
         * @return 1 (one scalar value per grid point)
         */
        Index_t get_nb_input_components() const override { return 1; }

        /**
         * @brief Get the spatial dimension.
         * @return 2
         */
        Dim_t get_spatial_dim() const override { return DIM; }

        /**
         * @brief Get the grid spacing.
         * @return Grid spacing vector
         */
        const std::vector<Real> & get_grid_spacing() const {
            return grid_spacing;
        }

        /**
         * @brief Get the quadrature weights.
         * @return Vector of quadrature weights (one per quadrature point)
         *
         * For 2D: Each triangle has weight = 0.5 * hx * hy (half the pixel area)
         */
        std::vector<Real> get_quadrature_weights() const;

        /**
         * @brief Get the stencil offset.
         * @return Stencil offset in pixels ([0,0])
         */
        Shape_t get_offset() const { return Shape_t{0, 0}; }

        /**
         * @brief Get the stencil shape.
         * @return Shape of the stencil ([2,2])
         */
        Shape_t get_stencil_shape() const { return Shape_t{2, 2}; }

        /**
         * @brief Get the stencil coefficients.
         * @return Vector of shape function gradients as flat array
         *
         * Returns the shape function gradients scaled by grid spacing.
         * Shape: (nb_output_components, nb_quad_pts, nb_input_components, 2, 2)
         */
        std::vector<Real> get_coefficients() const;

       private:
        std::vector<Real> grid_spacing;

        /**
         * @brief Validate that fields are compatible with this operator.
         * @param nodal_field Input field at nodal points
         * @param gradient_field Output field at gradient/quadrature points
         * @param nb_components Output: number of components in nodal field
         * @throws RuntimeError if validation fails
         */
        const GlobalFieldCollection &
        validate_fields(const Field & nodal_field, const Field & gradient_field,
                        Index_t & nb_components) const;

        /**
         * @brief Internal implementation of apply with optional increment.
         */
        void apply_impl(const TypedFieldBase<Real> & nodal_field,
                        TypedFieldBase<Real> & gradient_field, Real alpha,
                        bool increment) const;

        /**
         * @brief Internal implementation of transpose with optional increment.
         */
        void transpose_impl(const TypedFieldBase<Real> & gradient_field,
                            TypedFieldBase<Real> & nodal_field, Real alpha,
                            bool increment,
                            const std::vector<Real> & weights) const;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void
        apply_impl(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
                   Real alpha, bool increment) const;

        void transpose_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field, Real alpha,
            bool increment, const std::vector<Real> & weights) const;
#endif
    };

    // 2D Kernel implementations
    namespace fem_gradient_kernels {

        // =====================================================================
        // 2D Linear Triangle Shape Function Gradients
        // =====================================================================
        //
        // Pixel corners (nodal points):
        //   Node 0: (0, 0)  - bottom-left
        //   Node 1: (1, 0)  - bottom-right
        //   Node 2: (0, 1)  - top-left
        //   Node 3: (1, 1)  - top-right
        //
        // Two triangles per pixel:
        //   Triangle 0 (lower-left):  Nodes 0, 1, 2
        //   Triangle 1 (upper-right): Nodes 1, 3, 2
        //
        // For Triangle 0 with vertices at (0,0), (hx,0), (0,hy):
        //   N0 = 1 - x/hx - y/hy → dN0/dx = -1/hx, dN0/dy = -1/hy
        //   N1 = x/hx            → dN1/dx =  1/hx, dN1/dy =  0
        //   N2 = y/hy            → dN2/dx =  0,    dN2/dy =  1/hy
        //   N3 = 0               → dN3/dx =  0,    dN3/dy =  0
        //
        // For Triangle 1 with vertices at (hx,0), (hx,hy), (0,hy):
        //   N0 = 0               → dN0/dx =  0,    dN0/dy =  0
        //   N1 = 1 - y/hy        → dN1/dx =  0,    dN1/dy = -1/hy
        //   N2 = 1 - x/hx        → dN2/dx = -1/hx, dN2/dy =  0
        //   N3 = x/hx + y/hy - 1 → dN3/dx =  1/hx, dN3/dy =  1/hy
        //
        // =====================================================================

        // Number of nodes and quadrature points for 2D
        constexpr Index_t NB_NODES_2D = 4;
        constexpr Index_t NB_QUAD_2D = 2;
        constexpr Index_t DIM_2D = 2;

        // Shape function gradients for 2D [dim][quad][node]
        // Scaled by grid spacing at runtime
        constexpr Real B_2D_REF[DIM_2D][NB_QUAD_2D][NB_NODES_2D] = {
            // d/dx gradients
            {
                {-1.0, 1.0, 0.0, 0.0},  // Triangle 0: nodes 0,1,2,3
                {0.0, 0.0, -1.0, 1.0}   // Triangle 1: nodes 0,1,2,3
            },
            // d/dy gradients
            {
                {-1.0, 0.0, 1.0, 0.0},  // Triangle 0
                {0.0, -1.0, 0.0, 1.0}   // Triangle 1
            }};

        // Node offsets within pixel for 2D [node][dim]
        // Node ordering: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
        constexpr Index_t NODE_OFFSET_2D[NB_NODES_2D][DIM_2D] = {
            {0, 0},  // Node 0: bottom-left
            {1, 0},  // Node 1: bottom-right
            {0, 1},  // Node 2: top-left
            {1, 1}   // Node 3: top-right
        };

        // Quadrature weights for 2D (area of each triangle / total pixel area)
        constexpr Real QUAD_WEIGHT_2D[NB_QUAD_2D] = {0.5, 0.5};

        // =====================================================================
        // 2D Host Kernel Declarations
        // =====================================================================

        /**
         * @brief Apply 2D FEM gradient operator on host.
         */
        void fem_gradient_2d_host(const Real * MUGRID_RESTRICT nodal_input,
                                  Real * MUGRID_RESTRICT gradient_output,
                                  Index_t nx, Index_t ny,
                                  Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment);

        /**
         * @brief Apply 2D FEM transpose (divergence) operator on host.
         */
        void fem_divergence_2d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_n, Real hx, Real hy,
            const Real * quad_weights, Real alpha, bool increment);

#if defined(MUGRID_ENABLE_CUDA)
        void fem_gradient_2d_cuda(const Real * nodal_input,
                                  Real * gradient_output, Index_t nx,
                                  Index_t ny, Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment);

        void
        fem_divergence_2d_cuda(const Real * gradient_input, Real * nodal_output,
                               Index_t nx, Index_t ny, Index_t grad_stride_x,
                               Index_t grad_stride_y, Index_t grad_stride_q,
                               Index_t grad_stride_d, Index_t nodal_stride_x,
                               Index_t nodal_stride_y, Index_t nodal_stride_n,
                               Real hx, Real hy, const Real * quad_weights,
                               Real alpha, bool increment);
#endif

#if defined(MUGRID_ENABLE_HIP)
        void fem_gradient_2d_hip(const Real * nodal_input,
                                 Real * gradient_output, Index_t nx, Index_t ny,
                                 Index_t nodal_stride_x, Index_t nodal_stride_y,
                                 Index_t nodal_stride_n, Index_t grad_stride_x,
                                 Index_t grad_stride_y, Index_t grad_stride_q,
                                 Index_t grad_stride_d, Real hx, Real hy,
                                 Real alpha, bool increment);

        void fem_divergence_2d_hip(const Real * gradient_input,
                                   Real * nodal_output, Index_t nx, Index_t ny,
                                   Index_t grad_stride_x, Index_t grad_stride_y,
                                   Index_t grad_stride_q, Index_t grad_stride_d,
                                   Index_t nodal_stride_x,
                                   Index_t nodal_stride_y,
                                   Index_t nodal_stride_n, Real hx, Real hy,
                                   const Real * quad_weights, Real alpha,
                                   bool increment);
#endif

    }  // namespace fem_gradient_kernels

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_2D_HH_
