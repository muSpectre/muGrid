/**
 * @file   fem_gradient_operator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2024
 *
 * @brief  Factory interface for dimension-specific FEM gradient operators
 *
 * This header provides:
 * - FEMGradientOperator2D: 2D FEM gradient with triangular elements
 * - FEMGradientOperator3D: 3D FEM gradient with tetrahedral elements
 * - FEMGradientOperator: Factory class for backwards compatibility
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

#ifndef SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_OPERATOR_HH_
#define SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_OPERATOR_HH_

// Include dimension-specific implementations
#include "fem_gradient_operator_2d.hh"
#include "fem_gradient_operator_3d.hh"

#include "core/exception.hh"

#include <memory>
#include <vector>

namespace muGrid {

    /**
     * @class FEMGradientOperator
     * @brief Wrapper class providing a unified interface to dimension-specific
     *        FEM gradient operators for backwards compatibility.
     *
     * This class wraps either FEMGradientOperator2D or FEMGradientOperator3D
     * internally, dispatching calls to the appropriate implementation based on
     * the spatial dimension specified at construction time.
     *
     * For new code, consider using FEMGradientOperator2D or FEMGradientOperator3D
     * directly for slightly better performance (avoids virtual dispatch).
     */
    class FEMGradientOperator : public GradientOperator {
       public:
        using Parent = GradientOperator;

        /**
         * @brief Construct a FEM gradient operator for the given dimension.
         * @param spatial_dim Spatial dimension (2 or 3)
         * @param grid_spacing Grid spacing in each direction (default: [1.0, ...])
         */
        explicit FEMGradientOperator(Dim_t spatial_dim,
                                     std::vector<Real> grid_spacing = {})
            : Parent{}, spatial_dim{spatial_dim}, grid_spacing{grid_spacing} {
            if (spatial_dim == 2) {
                impl_2d = std::make_unique<FEMGradientOperator2D>(grid_spacing);
            } else if (spatial_dim == 3) {
                impl_3d = std::make_unique<FEMGradientOperator3D>(grid_spacing);
            } else {
                throw RuntimeError(
                    "FEMGradientOperator only supports 2D and 3D grids");
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
        FEMGradientOperator &
        operator=(const FEMGradientOperator & other) = delete;

        //! Move assignment operator
        FEMGradientOperator & operator=(FEMGradientOperator && other) = default;

        void apply(const TypedFieldBase<Real> & nodal_field,
                   TypedFieldBase<Real> & gradient_field) const override {
            if (impl_2d) {
                impl_2d->apply(nodal_field, gradient_field);
            } else {
                impl_3d->apply(nodal_field, gradient_field);
            }
        }

        void
        apply_increment(const TypedFieldBase<Real> & nodal_field,
                        const Real & alpha,
                        TypedFieldBase<Real> & gradient_field) const override {
            if (impl_2d) {
                impl_2d->apply_increment(nodal_field, alpha, gradient_field);
            } else {
                impl_3d->apply_increment(nodal_field, alpha, gradient_field);
            }
        }

        void transpose(const TypedFieldBase<Real> & gradient_field,
                       TypedFieldBase<Real> & nodal_field,
                       const std::vector<Real> & weights = {}) const override {
            if (impl_2d) {
                impl_2d->transpose(gradient_field, nodal_field, weights);
            } else {
                impl_3d->transpose(gradient_field, nodal_field, weights);
            }
        }

        void transpose_increment(
            const TypedFieldBase<Real> & gradient_field, const Real & alpha,
            TypedFieldBase<Real> & nodal_field,
            const std::vector<Real> & weights = {}) const override {
            if (impl_2d) {
                impl_2d->transpose_increment(gradient_field, alpha, nodal_field,
                                             weights);
            } else {
                impl_3d->transpose_increment(gradient_field, alpha, nodal_field,
                                             weights);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void
        apply(const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
              TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
            if (impl_2d) {
                impl_2d->apply(nodal_field, gradient_field);
            } else {
                impl_3d->apply(nodal_field, gradient_field);
            }
        }

        void apply_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field) const {
            if (impl_2d) {
                impl_2d->apply_increment(nodal_field, alpha, gradient_field);
            } else {
                impl_3d->apply_increment(nodal_field, alpha, gradient_field);
            }
        }

        void transpose(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            if (impl_2d) {
                impl_2d->transpose(gradient_field, nodal_field, weights);
            } else {
                impl_3d->transpose(gradient_field, nodal_field, weights);
            }
        }

        void transpose_increment(
            const TypedFieldBase<Real, DefaultDeviceSpace> & gradient_field,
            const Real & alpha,
            TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
            const std::vector<Real> & weights = {}) const {
            if (impl_2d) {
                impl_2d->transpose_increment(gradient_field, alpha, nodal_field,
                                             weights);
            } else {
                impl_3d->transpose_increment(gradient_field, alpha, nodal_field,
                                             weights);
            }
        }
#endif

        Index_t get_nb_output_components() const override {
            return spatial_dim;
        }

        Index_t get_nb_quad_pts() const override {
            return spatial_dim == 2 ? 2 : 5;
        }

        Index_t get_nb_input_components() const override { return 1; }

        Dim_t get_spatial_dim() const override { return spatial_dim; }

        const std::vector<Real> & get_grid_spacing() const {
            return grid_spacing;
        }

        std::vector<Real> get_quadrature_weights() const {
            if (impl_2d) {
                return impl_2d->get_quadrature_weights();
            } else {
                return impl_3d->get_quadrature_weights();
            }
        }

        Shape_t get_offset() const { return Shape_t(spatial_dim, 0); }

        Shape_t get_stencil_shape() const { return Shape_t(spatial_dim, 2); }

        std::vector<Real> get_coefficients() const {
            if (impl_2d) {
                return impl_2d->get_coefficients();
            } else {
                return impl_3d->get_coefficients();
            }
        }

       private:
        Dim_t spatial_dim;
        std::vector<Real> grid_spacing;
        std::unique_ptr<FEMGradientOperator2D> impl_2d;
        std::unique_ptr<FEMGradientOperator3D> impl_3d;
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_FEM_GRADIENT_OPERATOR_HH_
