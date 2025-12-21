/**
 * @file   convolution_operator_device.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  GPU implementations of ConvolutionOperator methods for device memory
 *
 * This file is compiled with the CUDA or HIP compiler depending on which
 * GPU backend is enabled. The implementation uses DefaultDeviceSpace which
 * resolves to CudaSpace or HIPSpace at compile time.
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

// Include cassert first as some headers use assert
#include <cassert>

#include "operators/convolution_operator.hh"
#include "collection/field_collection_global.hh"

namespace muGrid {

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & quadrature_point_field) const {
        quadrature_point_field.set_zero();
        this->apply_increment(nodal_field, 1., quadrature_point_field);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & quadrature_point_field) const {
        // Validate fields using generic version that works with base Field
        const auto & collection = this->validate_fields_generic(nodal_field,
                                                                quadrature_point_field);

        // Get component counts
        const Index_t nb_nodal_components = nodal_field.get_nb_components();
        const Index_t nb_quad_components = quadrature_point_field.get_nb_components();

        // Compute traversal parameters (use device storage order)
        const auto params = this->compute_traversal_params<DefaultDeviceSpace::storage_order>(
            collection, nb_nodal_components, nb_quad_components);

        // Ensure device sparse operator is cached (copies from host if needed)
        this->get_device_apply_operator<DefaultDeviceSpace>(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components);

        // Get raw data pointers from device fields
        const Real* nodal_data = nodal_field.view().data();
        Real* quad_data = quadrature_point_field.view().data();

        // Use device kernel
        this->apply_on_device<DefaultDeviceSpace>(
            nodal_data, quad_data, alpha, params);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose(
        const TypedFieldBase<Real, DefaultDeviceSpace> & quadrature_point_field,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        // set nodal field to zero
        nodal_field.set_zero();
        this->transpose_increment(quadrature_point_field, 1., nodal_field, weights);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose_increment(
        const TypedFieldBase<Real, DefaultDeviceSpace> & quadrature_point_field,
        const Real & alpha,
        TypedFieldBase<Real, DefaultDeviceSpace> & nodal_field,
        const std::vector<Real> & weights) const {
        // Note: weights are currently ignored for device implementation
        (void)weights;

        // Validate fields using generic version
        const auto & collection = this->validate_fields_generic(nodal_field,
                                                                quadrature_point_field,
                                                                true);

        // Get component counts
        const Index_t nb_nodal_components = nodal_field.get_nb_components();
        const Index_t nb_quad_components = quadrature_point_field.get_nb_components();

        // Compute traversal parameters (use device storage order)
        const auto params = this->compute_traversal_params<DefaultDeviceSpace::storage_order>(
            collection, nb_nodal_components, nb_quad_components);

        // Ensure device sparse operator is cached (copies from host if needed)
        this->get_device_transpose_operator<DefaultDeviceSpace>(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components);

        // Get raw data pointers from device fields
        Real* nodal_data = nodal_field.view().data();
        const Real* quad_data = quadrature_point_field.view().data();

        // Use device kernel
        this->transpose_on_device<DefaultDeviceSpace>(
            quad_data, nodal_data, alpha, params);
    }

}  // namespace muGrid
