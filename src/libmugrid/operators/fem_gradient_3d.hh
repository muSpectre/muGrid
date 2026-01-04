/**
 * @file   fem_gradient_operator_3d.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Hard-coded 3D linear FEM gradient operator
 *
 * This class provides optimized implementations of the gradient operator
 * for linear finite elements on 3D structured grids:
 * - 8 nodal points per voxel (corners of unit cube)
 * - 5 tetrahedra per voxel (Kuhn triangulation)
 * - 5 quadrature points (one per tetrahedron, at centroid)
 * - 3 gradient components (d/dx, d/dy, d/dz)
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_3D_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_3D_HH_

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
     * @class FEMGradientOperator3D
     * @brief Hard-coded 3D linear FEM gradient operator with optimized
     * implementation.
     *
     * This class provides optimized implementations of the gradient operator
     * for linear finite elements on 3D structured grids:
     *
     * **3D (Linear Tetrahedra):**
     * - 8 nodal points per voxel (corners of unit cube)
     * - 5 tetrahedra per voxel (Kuhn triangulation)
     * - 5 quadrature points (one per tetrahedron, at centroid)
     * - 3 gradient components (d/dx, d/dy, d/dz)
     *
     * The apply() method computes the gradient (nodal → quadrature points).
     * The transpose() method computes the divergence (quadrature → nodal
     * points).
     *
     * Shape function gradients are compile-time constants for linear elements,
     * enabling SIMD vectorization and optimal performance.
     */
    class FEMGradientOperator3D : public LinearOperator {
       public:
        using Parent = LinearOperator;

        //! Number of nodes per voxel (compile-time constant for 3D)
        static constexpr Index_t NB_NODES = 8;

        //! Number of quadrature points per voxel (compile-time constant for 3D)
        static constexpr Index_t NB_QUAD = 5;

        //! Spatial dimension
        static constexpr Dim_t DIM = 3;

        /**
         * @brief Construct a 3D FEM gradient operator.
         * @param grid_spacing Grid spacing in each direction (default: [1.0, 1.0, 1.0])
         *
         * The grid spacing is used to scale the shape function gradients.
         */
        explicit FEMGradientOperator3D(std::vector<Real> grid_spacing = {});

        //! Default constructor is deleted
        FEMGradientOperator3D() = delete;

        //! Copy constructor is deleted
        FEMGradientOperator3D(const FEMGradientOperator3D & other) = delete;

        //! Move constructor
        FEMGradientOperator3D(FEMGradientOperator3D && other) = default;

        //! Destructor
        ~FEMGradientOperator3D() override = default;

        //! Copy assignment operator is deleted
        FEMGradientOperator3D &
        operator=(const FEMGradientOperator3D & other) = delete;

        //! Move assignment operator
        FEMGradientOperator3D & operator=(FEMGradientOperator3D && other) = default;

        /**
         * @brief Apply the gradient operator (nodal → quadrature).
         */
        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & gradient_field) const override;

        /**
         * @brief Apply the gradient operator with increment.
         */
        void
        apply_increment(const TypedFieldBase<Real> & nodal_field,
                        const Real & alpha,
                        TypedFieldBase<Real> & gradient_field) const override;

        /**
         * @brief Apply the transpose (divergence) operator (quadrature → nodal).
         */
        void transpose(const TypedFieldBase<Real> & gradient_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const override;

        /**
         * @brief Apply the transpose (divergence) with increment.
         */
        void transpose_increment(
            const TypedFieldBase<Real> & gradient_field, const Real & alpha,
            TypedFieldBase<Real> & nodal_field,
            const std::vector<Real> & weights = {}) const override;

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void
        apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
              TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const;

        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const;

        void transpose(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;

        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const;
#endif

        /**
         * @brief Get the number of output components (3 for 3D gradient).
         * @return 3 (d/dx, d/dy, d/dz)
         */
        Index_t get_nb_output_components() const override { return DIM; }

        /**
         * @brief Get the number of quadrature points per voxel.
         * @return 5 (one per tetrahedron)
         */
        Index_t get_nb_quad_pts() const override { return NB_QUAD; }

        /**
         * @brief Get the number of input components per voxel.
         * @return 1 (one scalar value per grid point)
         */
        Index_t get_nb_input_components() const override { return 1; }

        /**
         * @brief Get the spatial dimension.
         * @return 3
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
         * For 3D: 5-tet decomposition with:
         * - Central tetrahedron (tet 0): volume = 1/3 of voxel
         * - Corner tetrahedra (tet 1-4): volume = 1/6 of voxel each
         */
        std::vector<Real> get_quadrature_weights() const;

        /**
         * @brief Get the stencil offset.
         * @return Stencil offset in pixels ([0,0,0])
         */
        Shape_t get_offset() const { return Shape_t{0, 0, 0}; }

        /**
         * @brief Get the stencil shape.
         * @return Shape of the stencil ([2,2,2])
         */
        Shape_t get_stencil_shape() const { return Shape_t{2, 2, 2}; }

        /**
         * @brief Get the stencil coefficients.
         * @return Vector of shape function gradients as flat array
         */
        std::vector<Real> get_coefficients() const;

       private:
        std::vector<Real> grid_spacing;

        const GlobalFieldCollection &
        validate_fields(const Field & nodal_field, const Field & gradient_field,
                        Index_t & nb_components) const;

        void apply_impl(const TypedFieldBase<Real> & nodal_field,
                        TypedFieldBase<Real> & gradient_field, Real alpha,
                        bool increment) const;

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

    // 3D Kernel implementations
    namespace fem_gradient_kernels {

        // =====================================================================
        // 3D Linear Tetrahedra Shape Function Gradients (Kuhn Triangulation)
        // =====================================================================
        //
        // Voxel corners (nodal points) - binary indexing:
        //   Node 0: (0,0,0), Node 1: (1,0,0), Node 2: (0,1,0), Node 3: (1,1,0)
        //   Node 4: (0,0,1), Node 5: (1,0,1), Node 6: (0,1,1), Node 7: (1,1,1)
        //
        // 5 tetrahedra per voxel (Kuhn triangulation):
        //   Tet 0: Central tetrahedron (volume 1/3)
        //   Tet 1-4: Corner tetrahedra (volume 1/6 each)
        //
        // =====================================================================

        // Number of nodes and quadrature points for 3D
        constexpr Index_t NB_NODES_3D = 8;
        constexpr Index_t NB_QUAD_3D = 5;
        constexpr Index_t DIM_3D = 3;

        // Node offsets within voxel for 3D [node][dim]
        constexpr Index_t NODE_OFFSET_3D[NB_NODES_3D][DIM_3D] = {
            {0, 0, 0},  // Node 0
            {1, 0, 0},  // Node 1
            {0, 1, 0},  // Node 2
            {1, 1, 0},  // Node 3
            {0, 0, 1},  // Node 4
            {1, 0, 1},  // Node 5
            {0, 1, 1},  // Node 6
            {1, 1, 1}   // Node 7
        };

        // Tetrahedra node indices [tet][4 nodes]
        constexpr Index_t TET_NODES[NB_QUAD_3D][4] = {
            {1, 2, 4, 7},  // Tet 0: Central tetrahedron (volume 1/3)
            {0, 1, 2, 4},  // Tet 1: Corner at (0,0,0) (volume 1/6)
            {1, 2, 3, 7},  // Tet 2: Corner at (1,1,0) (volume 1/6)
            {1, 4, 5, 7},  // Tet 3: Corner at (1,0,1) (volume 1/6)
            {2, 4, 6, 7}   // Tet 4: Corner at (0,1,1) (volume 1/6)
        };

        // Shape function gradients for 3D [dim][quad][node]
        extern const Real B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D];

        // Quadrature weights for 3D
        constexpr Real QUAD_WEIGHT_3D[NB_QUAD_3D] = {
            1.0 / 3.0,  // Tet 0: Central tetrahedron
            1.0 / 6.0,  // Tet 1: Corner at (0,0,0)
            1.0 / 6.0,  // Tet 2: Corner at (1,1,0)
            1.0 / 6.0,  // Tet 3: Corner at (1,0,1)
            1.0 / 6.0   // Tet 4: Corner at (0,1,1)
        };

        // =====================================================================
        // 3D Host Kernel Declarations
        // =====================================================================

        void fem_gradient_3d_host(
            const Real * MUGRID_RESTRICT nodal_input,
            Real * MUGRID_RESTRICT gradient_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
            Index_t grad_stride_q, Index_t grad_stride_d, Real hx, Real hy,
            Real hz, Real alpha, bool increment);

        void fem_divergence_3d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t grad_stride_x, Index_t grad_stride_y,
            Index_t grad_stride_z, Index_t grad_stride_q, Index_t grad_stride_d,
            Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n, Real hx, Real hy,
            Real hz, const Real * quad_weights, Real alpha, bool increment);

#if defined(MUGRID_ENABLE_CUDA)
        void
        fem_gradient_3d_cuda(const Real * nodal_input, Real * gradient_output,
                             Index_t nx, Index_t ny, Index_t nz,
                             Index_t nodal_stride_x, Index_t nodal_stride_y,
                             Index_t nodal_stride_z, Index_t nodal_stride_n,
                             Index_t grad_stride_x, Index_t grad_stride_y,
                             Index_t grad_stride_z, Index_t grad_stride_q,
                             Index_t grad_stride_d, Real hx, Real hy, Real hz,
                             Real alpha, bool increment);

        void fem_divergence_3d_cuda(
            const Real * gradient_input, Real * nodal_output, Index_t nx,
            Index_t ny, Index_t nz, Index_t grad_stride_x,
            Index_t grad_stride_y, Index_t grad_stride_z, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_z,
            Index_t nodal_stride_n, Real hx, Real hy, Real hz,
            const Real * quad_weights, Real alpha, bool increment);
#endif

#if defined(MUGRID_ENABLE_HIP)
        void fem_gradient_3d_hip(const Real * nodal_input,
                                 Real * gradient_output, Index_t nx, Index_t ny,
                                 Index_t nz, Index_t nodal_stride_x,
                                 Index_t nodal_stride_y, Index_t nodal_stride_z,
                                 Index_t nodal_stride_n, Index_t grad_stride_x,
                                 Index_t grad_stride_y, Index_t grad_stride_z,
                                 Index_t grad_stride_q, Index_t grad_stride_d,
                                 Real hx, Real hy, Real hz, Real alpha,
                                 bool increment);

        void fem_divergence_3d_hip(
            const Real * gradient_input, Real * nodal_output, Index_t nx,
            Index_t ny, Index_t nz, Index_t grad_stride_x,
            Index_t grad_stride_y, Index_t grad_stride_z, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_z,
            Index_t nodal_stride_n, Real hx, Real hy, Real hz,
            const Real * quad_weights, Real alpha, bool increment);
#endif

    }  // namespace fem_gradient_kernels

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_3D_HH_
