/**
 * @file   fem_gradient.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Dimension-templated hard-coded linear FEM gradient operator (2D/3D)
 *
 * The 2D (linear triangles, 4 nodes / 2 quad points) and 3D (linear
 * tetrahedra, 8 nodes / 5 quad points) FEM gradient operators shared the
 * entire apply/transpose interface, the device dispatch and the field
 * validation, differing only in the dimension, the element counts and the
 * shape-function tables. They are unified here into a single
 * `template <Dim_t Dim> FEMGradientOperator`, with the dimension-varying facts
 * in `FEMGradientTraits<Dim>` and the divergent stride/kernel selection behind
 * `if constexpr`. The dimension-specialised stencil kernels keep their distinct
 * signatures (defined in fem_gradient_2d.cc / fem_gradient_3d.cc for the host
 * and fem_gradient_gpu.cc for the device). The historical class names remain as
 * type aliases (FEMGradientOperator2D / FEMGradientOperator3D) so no binding or
 * downstream code changes.
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_

#include "core/types.hh"
#include "field/field_typed.hh"
#include "collection/field_collection_global.hh"
#include "memory/memory_space.hh"
#include "operators/linear.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "memory/gpu_runtime.hh"
#include "memory/device_alloc.hh"
#endif

#include <sstream>
#include <string>
#include <vector>

namespace muGrid {

    // Kernel implementations and reference tables. The host kernels are defined
    // in fem_gradient_2d.cc / fem_gradient_3d.cc, the device kernels in
    // fem_gradient_gpu.cc; both translation units include this header for the
    // declarations and the shape-function tables below.
    namespace fem_gradient_kernels {

        // ---- 2D linear triangles -------------------------------------------
        constexpr Index_t NB_NODES_2D = 4;
        constexpr Index_t NB_QUAD_2D = 2;
        constexpr Index_t DIM_2D = 2;

        // Shape function gradients [dim][quad][node], scaled by grid spacing at
        // runtime.
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

        // Node offsets within a pixel [node][dim]; 0=(0,0) 1=(1,0) 2=(0,1)
        // 3=(1,1).
        constexpr Index_t NODE_OFFSET_2D[NB_NODES_2D][DIM_2D] = {
            {0, 0}, {1, 0}, {0, 1}, {1, 1}};

        // Quadrature weights (area of each triangle / total pixel area).
        constexpr Real QUAD_WEIGHT_2D[NB_QUAD_2D] = {0.5, 0.5};

        // ---- 3D linear tetrahedra (Kuhn triangulation) ---------------------
        constexpr Index_t NB_NODES_3D = 8;
        constexpr Index_t NB_QUAD_3D = 5;
        constexpr Index_t DIM_3D = 3;

        // Node offsets within a voxel [node][dim] (binary indexing).
        constexpr Index_t NODE_OFFSET_3D[NB_NODES_3D][DIM_3D] = {
            {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

        // Tetrahedra node indices [tet][4 nodes] (Kuhn triangulation).
        constexpr Index_t TET_NODES[NB_QUAD_3D][4] = {
            {1, 2, 4, 7},  // Tet 0: central (volume 1/3)
            {0, 1, 2, 4},  // Tet 1: corner (0,0,0) (volume 1/6)
            {1, 2, 3, 7},  // Tet 2: corner (1,1,0)
            {1, 4, 5, 7},  // Tet 3: corner (1,0,1)
            {2, 4, 6, 7}   // Tet 4: corner (0,1,1)
        };

        // Quadrature weights for the 5-tet decomposition.
        constexpr Real QUAD_WEIGHT_3D[NB_QUAD_3D] = {
            1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

        // Shape function gradients [dim][quad][node]; defined in
        // fem_gradient_3d.cc.
        extern const Real B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D];

        // =====================================================================
        // 2D host kernel declarations
        // =====================================================================
        void fem_gradient_2d_host(const Real * MUGRID_RESTRICT nodal_input,
                                  Real * MUGRID_RESTRICT gradient_output,
                                  Index_t nx, Index_t ny,
                                  Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment);

        void fem_divergence_2d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_n, Real hx, Real hy,
            const Real * quad_weights, Real alpha, bool increment);

        // =====================================================================
        // 3D host kernel declarations
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

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void fem_gradient_2d_gpu(const Real * nodal_input,
                                  Real * gradient_output, Index_t nx,
                                  Index_t ny, Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment);
        void
        fem_divergence_2d_gpu(const Real * gradient_input, Real * nodal_output,
                               Index_t nx, Index_t ny, Index_t grad_stride_x,
                               Index_t grad_stride_y, Index_t grad_stride_q,
                               Index_t grad_stride_d, Index_t nodal_stride_x,
                               Index_t nodal_stride_y, Index_t nodal_stride_n,
                               Real hx, Real hy, const Real * quad_weights,
                               Real alpha, bool increment);
        void
        fem_gradient_3d_gpu(const Real * nodal_input, Real * gradient_output,
                             Index_t nx, Index_t ny, Index_t nz,
                             Index_t nodal_stride_x, Index_t nodal_stride_y,
                             Index_t nodal_stride_z, Index_t nodal_stride_n,
                             Index_t grad_stride_x, Index_t grad_stride_y,
                             Index_t grad_stride_z, Index_t grad_stride_q,
                             Index_t grad_stride_d, Real hx, Real hy, Real hz,
                             Real alpha, bool increment);
        void fem_divergence_3d_gpu(
            const Real * gradient_input, Real * nodal_output, Index_t nx,
            Index_t ny, Index_t nz, Index_t grad_stride_x,
            Index_t grad_stride_y, Index_t grad_stride_z, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_z,
            Index_t nodal_stride_n, Real hx, Real hy, Real hz,
            const Real * quad_weights, Real alpha, bool increment);
#endif

    }  // namespace fem_gradient_kernels

    /**
     * @struct FEMGradientTraits
     * @brief Dimension-varying facts of the linear FEM gradient element.
     */
    template <Dim_t Dim>
    struct FEMGradientTraits;
    template <>
    struct FEMGradientTraits<2> {
        static constexpr Index_t nb_nodes = 4;  //!< 4 corner nodes
        static constexpr Index_t nb_quad = 2;   //!< 2 triangles
    };
    template <>
    struct FEMGradientTraits<3> {
        static constexpr Index_t nb_nodes = 8;  //!< 8 corner nodes
        static constexpr Index_t nb_quad = 5;   //!< 5 tetrahedra
    };

    /**
     * @class FEMGradientOperator
     * @brief Hard-coded linear FEM gradient operator (2D triangles / 3D tets).
     *
     * apply() computes the gradient (nodal → quadrature points); transpose()
     * computes the (negative) discretised divergence (quadrature → nodal
     * points). Shape function gradients are compile-time constants for linear
     * elements, enabling SIMD vectorization and good performance.
     */
    template <Dim_t Dim>
    class FEMGradientOperator : public LinearOperator {
        static_assert(Dim == 2 || Dim == 3,
                      "FEMGradientOperator is only implemented for 2D and 3D");

       public:
        using Parent = LinearOperator;

        //! Number of nodes per pixel/voxel (4 in 2D, 8 in 3D)
        static constexpr Index_t NB_NODES = FEMGradientTraits<Dim>::nb_nodes;
        //! Number of quadrature points per pixel/voxel (2 in 2D, 5 in 3D)
        static constexpr Index_t NB_QUAD = FEMGradientTraits<Dim>::nb_quad;
        //! Spatial dimension
        static constexpr Dim_t DIM = Dim;

        /**
         * @brief Construct a FEM gradient operator.
         * @param grid_spacing Grid spacing in each direction (default: all 1.0)
         */
        explicit FEMGradientOperator(std::vector<Real> grid_spacing = {})
            : Parent{}, grid_spacing{std::move(grid_spacing)} {
            if (this->grid_spacing.empty()) {
                this->grid_spacing.resize(Dim, 1.0);
            }
            if (static_cast<Index_t>(this->grid_spacing.size()) != Dim) {
                std::stringstream err;
                err << "Grid spacing must have " << Dim
                    << " components for " << Dim << "D operator";
                throw RuntimeError(err.str());
            }
        }

        //! Default constructor is deleted
        FEMGradientOperator() = delete;
        //! Copy constructor is deleted
        FEMGradientOperator(const FEMGradientOperator & other) = delete;
        //! Move constructor
        FEMGradientOperator(FEMGradientOperator && other) = default;
        //! Destructor
        ~FEMGradientOperator() override = default;
        //! Copy assignment operator is deleted
        FEMGradientOperator & operator=(const FEMGradientOperator & other) = delete;
        //! Move assignment operator
        FEMGradientOperator & operator=(FEMGradientOperator && other) = default;

        // ---- Host interface ----
        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & gradient_field) const override {
            this->apply_impl(nodal_field, gradient_field, 1.0, false);
        }
        void apply_increment(const TypedFieldBase<Real> & nodal_field,
                             const Real & alpha,
                             TypedFieldBase<Real> & gradient_field)
            const override {
            this->apply_impl(nodal_field, gradient_field, alpha, true);
        }
        void transpose(const TypedFieldBase<Real> & gradient_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const override {
            this->transpose_impl(gradient_field, nodal_field, 1.0, false,
                                 weights);
        }
        void transpose_increment(const TypedFieldBase<Real> & gradient_field,
                                 const Real & alpha,
                                 TypedFieldBase<Real> & nodal_field,
                                 const std::vector<Real> & weights = {})
            const override {
            this->transpose_impl(gradient_field, nodal_field, alpha, true,
                                 weights);
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // ---- Device interface ----
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field)
            const {
            this->apply_impl(nodal_field, gradient_field, 1.0, false);
        }
        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
            this->apply_impl(nodal_field, gradient_field, alpha, true);
        }
        void transpose(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            this->transpose_impl(gradient_field, nodal_field, 1.0, false,
                                 weights);
        }
        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            this->transpose_impl(gradient_field, nodal_field, alpha, true,
                                 weights);
        }
#endif

        //! Number of output components (gradient components = spatial dim).
        Index_t get_nb_output_components() const override { return DIM; }
        //! Number of quadrature points per pixel/voxel.
        Index_t get_nb_quad_pts() const override { return NB_QUAD; }
        //! Number of input components (one scalar per grid point).
        Index_t get_nb_input_components() const override { return 1; }
        //! Spatial dimension.
        Dim_t get_spatial_dim() const override { return DIM; }

        //! Grid spacing.
        const std::vector<Real> & get_grid_spacing() const {
            return grid_spacing;
        }

        //! Quadrature weights (one per quadrature point).
        std::vector<Real> get_quadrature_weights() const {
            if constexpr (Dim == 2) {
                // Each triangle has area = 0.5 * hx * hy; pixel area = hx * hy.
                Real pixel_area = grid_spacing[0] * grid_spacing[1];
                return {0.5 * pixel_area, 0.5 * pixel_area};
            } else {
                // 5-tet decomposition: central tet = 1/3 voxel, four corner
                // tets = 1/6 voxel each.
                Real voxel_volume =
                    grid_spacing[0] * grid_spacing[1] * grid_spacing[2];
                return {voxel_volume / 3.0, voxel_volume / 6.0,
                        voxel_volume / 6.0, voxel_volume / 6.0,
                        voxel_volume / 6.0};
            }
        }

        //! Stencil offset in pixels (the element spans [0, +1] in each axis).
        Shape_t get_offset() const override {
            if constexpr (Dim == 2) {
                return Shape_t{0, 0};
            } else {
                return Shape_t{0, 0, 0};
            }
        }

        //! Stencil shape in pixels (2 in every direction).
        Shape_t get_stencil_shape() const override {
            if constexpr (Dim == 2) {
                return Shape_t{2, 2};
            } else {
                return Shape_t{2, 2, 2};
            }
        }

        /**
         * @brief Ghost layers required by transpose().
         *
         * The transpose scatters into the same ghost buffers that apply()
         * reads (followed by ghost reduction), so it has the same ghost
         * requirement as apply().
         */
        GhostRequirement get_transpose_ghost_requirement() const override {
            return this->get_apply_ghost_requirement();
        }

        //! Shape function gradients as a flat (Fortran-order) array, scaled by
        //! the inverse grid spacing. Shape (DIM, NB_QUAD, 1, [2]*Dim).
        std::vector<Real> get_coefficients() const {
            using namespace fem_gradient_kernels;
            std::vector<Real> result;
            if constexpr (Dim == 2) {
                Real hx = grid_spacing[0], hy = grid_spacing[1];
                result.reserve(NB_QUAD_2D * DIM_2D * NB_NODES_2D);
                for (Index_t j = 0; j < 2; ++j) {        // stencil_y
                    for (Index_t i = 0; i < 2; ++i) {    // stencil_x
                        for (Index_t q = 0; q < NB_QUAD_2D; ++q) {
                            for (Index_t d = 0; d < DIM_2D; ++d) {
                                Index_t node = j * 2 + i;
                                Real grad_val = B_2D_REF[d][q][node];
                                grad_val *= (d == 0) ? (1.0 / hx) : (1.0 / hy);
                                result.push_back(grad_val);
                            }
                        }
                    }
                }
            } else {
                Real hx = grid_spacing[0], hy = grid_spacing[1],
                     hz = grid_spacing[2];
                result.reserve(NB_QUAD_3D * DIM_3D * NB_NODES_3D);
                for (Index_t k = 0; k < 2; ++k) {            // stencil_z
                    for (Index_t j = 0; j < 2; ++j) {        // stencil_y
                        for (Index_t i = 0; i < 2; ++i) {    // stencil_x
                            for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                                for (Index_t d = 0; d < DIM_3D; ++d) {
                                    Index_t node = k * 4 + j * 2 + i;
                                    Real grad_val = B_3D_REF[d][q][node];
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
            }
            return result;
        }

       private:
        std::vector<Real> grid_spacing;

        //! Operator name used in validation error messages.
        static const char * operator_name() {
            return Dim == 2 ? "FEMGradientOperator2D" : "FEMGradientOperator3D";
        }

        //! Common validation plus the operator-specific component-count check.
        const GlobalFieldCollection &
        validate_fields(const Field & nodal_field, const Field & gradient_field,
                        Index_t & nb_components) const {
            // Same collection, global, matching dimension, ghost layers (the
            // scatter-style transpose has the same requirement as apply).
            const auto & global_fc =
                this->check_fields(nodal_field, gradient_field,
                                   operator_name());

            const Index_t nb_nodal_components = nodal_field.get_nb_components();
            const Index_t nb_grad_components =
                gradient_field.get_nb_components();
            const Index_t expected_grad_components =
                this->get_nb_output_components() * nb_nodal_components;
            if (nb_grad_components != expected_grad_components) {
                std::stringstream err_msg;
                err_msg << "Component mismatch: Expected gradient field with "
                        << expected_grad_components << " components ("
                        << this->get_nb_output_components()
                        << " output components × " << nb_nodal_components
                        << " nodal components) but got " << nb_grad_components
                        << " components.";
                throw RuntimeError(err_msg.str());
            }
            nb_components = nb_nodal_components;
            return global_fc;
        }

        //! Host apply/gradient with optional increment.
        void apply_impl(const TypedFieldBase<Real> & nodal_field,
                        TypedFieldBase<Real> & gradient_field, Real alpha,
                        bool increment) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * nodal = nodal_field.data();
            Real * gradient = gradient_field.data();
            const Index_t nb_sub = this->get_nb_input_components();

            // AoS layout: components vary fastest, then sub-pts/quad/operators,
            // then the spatial axes.
            const Index_t nx = nb_grid_pts[0];
            const Index_t ny = nb_grid_pts[1];
            const Index_t nodal_stride_c = 1;
            const Index_t nodal_stride_n = nb_components;
            const Index_t nodal_stride_x = nb_components * nb_sub;
            const Index_t nodal_stride_y = nb_components * nb_sub * nx;
            const Index_t grad_stride_c = 1;
            const Index_t grad_stride_d = nb_components;
            const Index_t grad_stride_q = nb_components * DIM;
            const Index_t grad_stride_x = nb_components * DIM * NB_QUAD;
            const Index_t grad_stride_y = nb_components * DIM * NB_QUAD * nx;

            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real * nodal_comp = nodal + comp * nodal_stride_c;
                Real * gradient_comp = gradient + comp * grad_stride_c;
                if constexpr (Dim == 2) {
                    fem_gradient_kernels::fem_gradient_2d_host(
                        nodal_comp, gradient_comp, nx, ny, nodal_stride_x,
                        nodal_stride_y, nodal_stride_n, grad_stride_x,
                        grad_stride_y, grad_stride_q, grad_stride_d,
                        grid_spacing[0], grid_spacing[1], alpha, increment);
                } else {
                    const Index_t nz = nb_grid_pts[2];
                    const Index_t nodal_stride_z = nb_components * nb_sub * nx * ny;
                    const Index_t grad_stride_z =
                        nb_components * DIM * NB_QUAD * nx * ny;
                    fem_gradient_kernels::fem_gradient_3d_host(
                        nodal_comp, gradient_comp, nx, ny, nz, nodal_stride_x,
                        nodal_stride_y, nodal_stride_z, nodal_stride_n,
                        grad_stride_x, grad_stride_y, grad_stride_z,
                        grad_stride_q, grad_stride_d, grid_spacing[0],
                        grid_spacing[1], grid_spacing[2], alpha, increment);
                }
            }
        }

        //! Host transpose/divergence with optional increment.
        void transpose_impl(const TypedFieldBase<Real> & gradient_field,
                            TypedFieldBase<Real> & nodal_field, Real alpha,
                            bool increment,
                            const std::vector<Real> & weights) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * gradient = gradient_field.data();
            Real * nodal = nodal_field.data();
            const std::vector<Real> quad_weights =
                weights.empty() ? this->get_quadrature_weights() : weights;
            const Index_t nb_nodes = this->get_nb_input_components();

            const Index_t nx = nb_grid_pts[0];
            const Index_t ny = nb_grid_pts[1];
            const Index_t nodal_stride_c = 1;
            const Index_t nodal_stride_n = nb_components;
            const Index_t nodal_stride_x = nb_components * nb_nodes;
            const Index_t nodal_stride_y = nb_components * nb_nodes * nx;
            const Index_t grad_stride_c = 1;
            const Index_t grad_stride_d = nb_components;
            const Index_t grad_stride_q = nb_components * DIM;
            const Index_t grad_stride_x = nb_components * DIM * NB_QUAD;
            const Index_t grad_stride_y = nb_components * DIM * NB_QUAD * nx;

            for (Index_t comp = 0; comp < nb_components; ++comp) {
                const Real * gradient_comp = gradient + comp * grad_stride_c;
                Real * nodal_comp = nodal + comp * nodal_stride_c;
                if constexpr (Dim == 2) {
                    fem_gradient_kernels::fem_divergence_2d_host(
                        gradient_comp, nodal_comp, nx, ny, grad_stride_x,
                        grad_stride_y, grad_stride_q, grad_stride_d,
                        nodal_stride_x, nodal_stride_y, nodal_stride_n,
                        grid_spacing[0], grid_spacing[1], quad_weights.data(),
                        alpha, increment);
                } else {
                    const Index_t nz = nb_grid_pts[2];
                    const Index_t nodal_stride_z = nb_components * nb_nodes * nx * ny;
                    const Index_t grad_stride_z =
                        nb_components * DIM * NB_QUAD * nx * ny;
                    fem_gradient_kernels::fem_divergence_3d_host(
                        gradient_comp, nodal_comp, nx, ny, nz, grad_stride_x,
                        grad_stride_y, grad_stride_z, grad_stride_q,
                        grad_stride_d, nodal_stride_x, nodal_stride_y,
                        nodal_stride_z, nodal_stride_n, grid_spacing[0],
                        grid_spacing[1], grid_spacing[2], quad_weights.data(),
                        alpha, increment);
                }
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! Device apply/gradient with optional increment.
        void apply_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            Real alpha, bool increment) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * nodal = nodal_field.view().data();
            Real * gradient = gradient_field.view().data();

            // SoA layout: spatial axes fastest, then sub-pts/quad/operators,
            // then components.
            const Index_t nx = nb_grid_pts[0];
            const Index_t ny = nb_grid_pts[1];
            if constexpr (Dim == 2) {
                const Index_t nodal_stride_x = 1;
                const Index_t nodal_stride_y = nx;
                const Index_t nodal_stride_n = nx * ny;
                const Index_t nodal_stride_c =
                    nx * ny * this->get_nb_input_components();
                const Index_t grad_stride_x = 1;
                const Index_t grad_stride_y = nx;
                const Index_t grad_stride_q = nx * ny;
                const Index_t grad_stride_d = nx * ny * NB_QUAD;
                const Index_t grad_stride_c = nx * ny * NB_QUAD * DIM;
                for (Index_t comp = 0; comp < nb_components; ++comp) {
                    const Real * nodal_comp = nodal + comp * nodal_stride_c;
                    Real * gradient_comp = gradient + comp * grad_stride_c;
                    fem_gradient_kernels::fem_gradient_2d_gpu(
                        nodal_comp, gradient_comp, nx, ny, nodal_stride_x,
                        nodal_stride_y, nodal_stride_n, grad_stride_x,
                        grad_stride_y, grad_stride_q, grad_stride_d,
                        grid_spacing[0], grid_spacing[1], alpha, increment);
                }
            } else {
                const Index_t nz = nb_grid_pts[2];
                const Index_t nodal_stride_x = 1;
                const Index_t nodal_stride_y = nx;
                const Index_t nodal_stride_z = nx * ny;
                const Index_t nodal_stride_n = nx * ny * nz;
                const Index_t nodal_stride_c =
                    nx * ny * nz * this->get_nb_input_components();
                const Index_t grad_stride_x = 1;
                const Index_t grad_stride_y = nx;
                const Index_t grad_stride_z = nx * ny;
                const Index_t grad_stride_q = nx * ny * nz;
                const Index_t grad_stride_d = nx * ny * nz * NB_QUAD;
                const Index_t grad_stride_c = nx * ny * nz * NB_QUAD * DIM;
                for (Index_t comp = 0; comp < nb_components; ++comp) {
                    const Real * nodal_comp = nodal + comp * nodal_stride_c;
                    Real * gradient_comp = gradient + comp * grad_stride_c;
                    fem_gradient_kernels::fem_gradient_3d_gpu(
                        nodal_comp, gradient_comp, nx, ny, nz, nodal_stride_x,
                        nodal_stride_y, nodal_stride_z, nodal_stride_n,
                        grad_stride_x, grad_stride_y, grad_stride_z,
                        grad_stride_q, grad_stride_d, grid_spacing[0],
                        grid_spacing[1], grid_spacing[2], alpha, increment);
                }
            }
        }

        //! Device transpose/divergence with optional increment.
        void transpose_impl(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field, Real alpha,
            bool increment, const std::vector<Real> & weights) const {
            Index_t nb_components;
            const auto & collection =
                this->validate_fields(nodal_field, gradient_field,
                                      nb_components);
            const auto nb_grid_pts =
                collection.get_nb_subdomain_grid_pts_with_ghosts();
            const Real * gradient = gradient_field.view().data();
            Real * nodal = nodal_field.view().data();
            const std::vector<Real> quad_weights =
                weights.empty() ? this->get_quadrature_weights() : weights;

            const Index_t nx = nb_grid_pts[0];
            const Index_t ny = nb_grid_pts[1];
            if constexpr (Dim == 2) {
                const Index_t nodal_stride_x = 1;
                const Index_t nodal_stride_y = nx;
                const Index_t nodal_stride_n = nx * ny;
                const Index_t nodal_stride_c =
                    nx * ny * this->get_nb_input_components();
                const Index_t grad_stride_x = 1;
                const Index_t grad_stride_y = nx;
                const Index_t grad_stride_q = nx * ny;
                const Index_t grad_stride_d = nx * ny * NB_QUAD;
                const Index_t grad_stride_c = nx * ny * NB_QUAD * DIM;
                for (Index_t comp = 0; comp < nb_components; ++comp) {
                    const Real * gradient_comp = gradient + comp * grad_stride_c;
                    Real * nodal_comp = nodal + comp * nodal_stride_c;
                    fem_gradient_kernels::fem_divergence_2d_gpu(
                        gradient_comp, nodal_comp, nx, ny, grad_stride_x,
                        grad_stride_y, grad_stride_q, grad_stride_d,
                        nodal_stride_x, nodal_stride_y, nodal_stride_n,
                        grid_spacing[0], grid_spacing[1], quad_weights.data(),
                        alpha, increment);
                }
            } else {
                const Index_t nz = nb_grid_pts[2];
                const Index_t nodal_stride_x = 1;
                const Index_t nodal_stride_y = nx;
                const Index_t nodal_stride_z = nx * ny;
                const Index_t nodal_stride_n = nx * ny * nz;
                const Index_t nodal_stride_c =
                    nx * ny * nz * this->get_nb_input_components();
                const Index_t grad_stride_x = 1;
                const Index_t grad_stride_y = nx;
                const Index_t grad_stride_z = nx * ny;
                const Index_t grad_stride_q = nx * ny * nz;
                const Index_t grad_stride_d = nx * ny * nz * NB_QUAD;
                const Index_t grad_stride_c = nx * ny * nz * NB_QUAD * DIM;
                // The 3D divergence kernel reads the quadrature weights on the
                // device (unlike the 2D launch wrapper, which consumes them on
                // the host), so stage them in device memory.
                // Route through the device allocator chokepoint (single
                // owner + visible to the allocation profiler).
                Real * d_quad_weights = static_cast<Real *>(device_allocate(
                    quad_weights.size() * sizeof(Real), "fem-divergence-weights"));
                GPU_MEMCPY_H2D(d_quad_weights, quad_weights.data(),
                               quad_weights.size() * sizeof(Real));
                for (Index_t comp = 0; comp < nb_components; ++comp) {
                    const Real * gradient_comp = gradient + comp * grad_stride_c;
                    Real * nodal_comp = nodal + comp * nodal_stride_c;
                    fem_gradient_kernels::fem_divergence_3d_gpu(
                        gradient_comp, nodal_comp, nx, ny, nz, grad_stride_x,
                        grad_stride_y, grad_stride_z, grad_stride_q,
                        grad_stride_d, nodal_stride_x, nodal_stride_y,
                        nodal_stride_z, nodal_stride_n, grid_spacing[0],
                        grid_spacing[1], grid_spacing[2], d_quad_weights,
                        alpha, increment);
                }
                device_deallocate(d_quad_weights);
            }
        }
#endif
    };

    //! 2D linear-triangle FEM gradient. Preserves the historical name.
    using FEMGradientOperator2D = FEMGradientOperator<2>;
    //! 3D linear-tetrahedra FEM gradient. Preserves the historical name.
    using FEMGradientOperator3D = FEMGradientOperator<3>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_HH_
