/**
 * @file   laplace_operator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Dec 2024
 *
 * @brief  Factory interface for dimension-specific Laplace operators
 *
 * This header provides:
 * - LaplaceOperator2D: Optimized 2D Laplace operator with 5-point stencil
 * - LaplaceOperator3D: Optimized 3D Laplace operator with 7-point stencil
 * - LaplaceOperator: Factory class for backwards compatibility
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

#ifndef SRC_LIBMUGRID_OPERATORS_LAPLACE_OPERATOR_HH_
#define SRC_LIBMUGRID_OPERATORS_LAPLACE_OPERATOR_HH_

// Include dimension-specific implementations
#include "laplace_operator_2d.hh"
#include "laplace_operator_3d.hh"

#include "core/exception.hh"

#include <memory>
#include <variant>

namespace muGrid {

    /**
     * @class LaplaceOperator
     * @brief Wrapper class providing a unified interface to dimension-specific
     *        Laplace operators for backwards compatibility.
     *
     * This class wraps either LaplaceOperator2D or LaplaceOperator3D internally,
     * dispatching calls to the appropriate implementation based on the spatial
     * dimension specified at construction time.
     *
     * For new code, consider using LaplaceOperator2D or LaplaceOperator3D
     * directly for slightly better performance (avoids virtual dispatch).
     */
    class LaplaceOperator : public GradientOperator {
    public:
        using Parent = GradientOperator;

        /**
         * @brief Construct a Laplace operator for the given dimension.
         * @param spatial_dim Spatial dimension (2 or 3)
         * @param scale Scale factor applied to the output (default: 1.0)
         */
        explicit LaplaceOperator(Index_t spatial_dim, Real scale = 1.0)
            : Parent{}, spatial_dim{spatial_dim}, scale{scale} {
            if (spatial_dim == 2) {
                impl_2d = std::make_unique<LaplaceOperator2D>(scale);
            } else if (spatial_dim == 3) {
                impl_3d = std::make_unique<LaplaceOperator3D>(scale);
            } else {
                throw RuntimeError("LaplaceOperator only supports 2D and 3D grids");
            }
        }

        //! Default constructor is deleted
        LaplaceOperator() = delete;

        //! Copy constructor is deleted
        LaplaceOperator(const LaplaceOperator &other) = delete;

        //! Move constructor
        LaplaceOperator(LaplaceOperator &&other) = default;

        //! Destructor
        ~LaplaceOperator() override = default;

        //! Copy assignment operator is deleted
        LaplaceOperator &operator=(const LaplaceOperator &other) = delete;

        //! Move assignment operator
        LaplaceOperator &operator=(LaplaceOperator &&other) = default;

        void apply(const TypedFieldBase<Real> &input_field,
                   TypedFieldBase<Real> &output_field) const override {
            if (impl_2d) {
                impl_2d->apply(input_field, output_field);
            } else {
                impl_3d->apply(input_field, output_field);
            }
        }

        void apply_increment(const TypedFieldBase<Real> &input_field,
                             const Real &alpha,
                             TypedFieldBase<Real> &output_field) const override {
            if (impl_2d) {
                impl_2d->apply_increment(input_field, alpha, output_field);
            } else {
                impl_3d->apply_increment(input_field, alpha, output_field);
            }
        }

        void transpose(const TypedFieldBase<Real> &input_field,
                       TypedFieldBase<Real> &output_field,
                       const std::vector<Real> &weights = {}) const override {
            if (impl_2d) {
                impl_2d->transpose(input_field, output_field, weights);
            } else {
                impl_3d->transpose(input_field, output_field, weights);
            }
        }

        void transpose_increment(const TypedFieldBase<Real> &input_field,
                                 const Real &alpha,
                                 TypedFieldBase<Real> &output_field,
                                 const std::vector<Real> &weights = {}) const override {
            if (impl_2d) {
                impl_2d->transpose_increment(input_field, alpha, output_field, weights);
            } else {
                impl_3d->transpose_increment(input_field, alpha, output_field, weights);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        void apply(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                   TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const {
            if (impl_2d) {
                impl_2d->apply(input_field, output_field);
            } else {
                impl_3d->apply(input_field, output_field);
            }
        }

        void apply_increment(const TypedFieldBase<Real, DefaultDeviceSpace> &input_field,
                             const Real &alpha,
                             TypedFieldBase<Real, DefaultDeviceSpace> &output_field) const {
            if (impl_2d) {
                impl_2d->apply_increment(input_field, alpha, output_field);
            } else {
                impl_3d->apply_increment(input_field, alpha, output_field);
            }
        }
#endif

        Index_t get_nb_output_components() const override { return 1; }
        Index_t get_nb_quad_pts() const override { return 1; }
        Index_t get_nb_input_components() const override { return 1; }
        Dim_t get_spatial_dim() const override { return spatial_dim; }

        Index_t get_nb_stencil_pts() const {
            return spatial_dim == 2 ? 5 : 7;
        }

        Real get_scale() const { return scale; }

        Shape_t get_offset() const {
            return Shape_t(spatial_dim, -1);
        }

        Shape_t get_stencil_shape() const {
            return Shape_t(spatial_dim, 3);
        }

        std::vector<Real> get_coefficients() const {
            if (impl_2d) {
                return impl_2d->get_coefficients();
            } else {
                return impl_3d->get_coefficients();
            }
        }

    private:
        Index_t spatial_dim;
        Real scale;
        std::unique_ptr<LaplaceOperator2D> impl_2d;
        std::unique_ptr<LaplaceOperator3D> impl_3d;
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_LAPLACE_OPERATOR_HH_
