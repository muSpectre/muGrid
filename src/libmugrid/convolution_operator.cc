/**
 * @file   gradient_operator_default.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   23 Jun 2020
 *
 * @brief  Implementation of member functions for the default gradient operator
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "convolution_operator.hh"
#include "grid_common.hh"
#include "field_collection_global.hh"
#include "ccoord_operations.hh"
#include "iterators.hh"
#include "exception.hh"

#include <sstream>

namespace muGrid {

    /* ---------------------------------------------------------------------- */
    ConvolutionOperator::ConvolutionOperator(
        const Shape_t & pixel_offset, const Eigen::MatrixXd & pixel_operator,
        const Shape_t & conv_pts_shape, const Index_t & nb_pixelnodal_pts,
        const Index_t & nb_quad_pts, const Index_t & nb_operators)
        : Parent{}, pixel_offset{pixel_offset}, pixel_operator{pixel_operator},
          conv_pts_shape{conv_pts_shape}, nb_pixelnodal_pts{nb_pixelnodal_pts},
          nb_quad_pts{nb_quad_pts}, nb_operators{nb_operators},
          spatial_dim{static_cast<Index_t>(conv_pts_shape.size())},
          nb_conv_pts{get_nb_from_shape(conv_pts_shape)} {
        // Check the dimension of the pixel operator
        if (pixel_operator.cols() !=
            this->nb_pixelnodal_pts * this->nb_conv_pts) {
            std::stringstream err_msg{};
            err_msg << "Size mismatch: Expected the operator has "
                    << this->nb_pixelnodal_pts * this->nb_conv_pts
                    << " columns. but received an operator with "
                    << pixel_operator.cols() << " columns";
            throw RuntimeError{err_msg.str()};
        }
        if (pixel_operator.rows() != this->nb_operators * this->nb_quad_pts) {
            std::stringstream err_msg{};
            err_msg << "Size mismatch: Expected the operator has "
                    << this->nb_operators * this->nb_quad_pts
                    << " rows. but received an operator with "
                    << pixel_operator.rows() << " rows";
            throw RuntimeError{err_msg.str()};
        }
    }

    /* ---------------------------------------------------------------------- */
    ConvolutionOperator::SparseOperator
    ConvolutionOperator::create_sparse_operator(const IntCoord_t & nb_grid_pts, const Index_t nb_nodal_components) const {
        // Helpers for conversion between index and coordinates
        const CcoordOps::Pixels kernel_pixels{IntCoord_t(this->conv_pts_shape),
                                              IntCoord_t(this->pixel_offset)};
        const CcoordOps::Pixels grid_pixels{nb_grid_pts};

        // An empty sequence to save output
        SparseOperator sparse_op{};
        // Loop through each value of pixel operator
        for (Index_t i_row = 0; i_row < this->pixel_operator.rows(); ++i_row) {
            for (Index_t i_col = 0; i_col < this->pixel_operator.cols();
                 ++i_col) {
                // Only the non-zero values are of the interest
                if (this->pixel_operator(i_row, i_col) != 0.) {
                    // repeat for each component
                    for (Index_t i_component = 0;
                         i_component < nb_nodal_components; ++i_component) {
                        // Get the index in quad field
                        auto index_diff_quad{i_row * nb_nodal_components +
                                             i_component};

                        // Decompose column index into node, stencil indices.
                        // (Given we know it is column-major flattened)
                        auto i_node{i_col % this->nb_pixelnodal_pts};
                        auto i_stencil{i_col / this->nb_pixelnodal_pts};

                        // Stencil index in `pixel_operator` is not aware of
                        // grid shape, so it must be decomposed to offset in
                        // coordinates, and reconstructed to index difference
                        // for the use of indexing pixels in the grid.
                        auto offset{kernel_pixels.get_coord(i_stencil)};
                        auto index_diff{grid_pixels.get_index(offset)};
                        auto index_diff_nodal{index_diff * nb_nodal_components *
                                                  this->nb_pixelnodal_pts +
                                              i_node * nb_nodal_components +
                                              i_component};

                        // Create an entry in sparse representation, with index
                        // differences and operator value
                        sparse_op.push_back(std::make_tuple(
                            index_diff_quad, index_diff_nodal,
                            this->pixel_operator(i_row, i_col)));
                    }
                }
            }
        }
        return sparse_op;
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply(
        const TypedFieldBase<Real> & nodal_field,
        TypedFieldBase<Real> & quadrature_point_field) const {
        quadrature_point_field.set_zero();
        this->apply_increment(nodal_field, 1., quadrature_point_field);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::apply_increment(
        const TypedFieldBase<Real> & nodal_field, const Real & alpha,
        TypedFieldBase<Real> & quadrature_point_field) const {
        // Both fields must be from the same field collection to ensure
        // compatible internal structure for pixel mapping
        if (&nodal_field.get_collection() !=
            &quadrature_point_field.get_collection()) {
            std::stringstream err_msg{};
            err_msg << "Field collection mismatch: nodal_field and "
                       "quadrature_point_field must be from the same "
                       "FieldCollection";
            throw RuntimeError{err_msg.str()};
            }

        // Get the collection object
        const auto & collection{dynamic_cast<GlobalFieldCollection &>(
            quadrature_point_field.get_collection())};

        // Check that fields are global
        if (collection.get_domain() !=
            FieldCollection::ValidityDomain::Global) {
            std::stringstream err_msg{};
            err_msg << "Field type error: nodal_field and "
                       "quadrature_point_field must be a global "
                       "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }

        // Check that fields have the same spatial dimensions as operator
        if (collection.get_spatial_dim() != this->spatial_dim) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: nodal_field and "
                       "quadrature_point_field are defined in "
                    << collection.get_spatial_dim()
                    << "D space, but this convolution operator is defined in "
                    << this->spatial_dim << "D space";
            throw RuntimeError{err_msg.str()};
        }

        // Check that fields have enough ghost cells on the left
        const auto & nb_ghosts_left{collection.get_nb_ghosts_left()};
        const auto min_ghosts_left{IntCoord_t(this->spatial_dim, 0) -
                                   IntCoord_t(this->pixel_offset)};
        for (auto direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_left[direction] < min_ghosts_left[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the convolution expects a minimum of "
                    << min_ghosts_left[direction]
                    << " cells on the left, but the provided fields have only "
                    << nb_ghosts_left[direction] << " ghosts on the left.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // Check that fields have enough ghost cells on the right
        const auto & nb_ghosts_right{collection.get_nb_ghosts_right()};
        const auto min_ghosts_right{IntCoord_t(this->conv_pts_shape) -
                                    IntCoord_t(this->spatial_dim, 1) +
                                    IntCoord_t(this->pixel_offset)};
        for (auto direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_right[direction] < min_ghosts_right[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the convolution expects a minimum of "
                    << min_ghosts_right[direction]
                    << " cells on the right, but the provided fields have only "
                    << nb_ghosts_right[direction] << " ghosts on the right.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // number of components in the field we'd like to apply the convolution
        Index_t nb_nodal_components{nodal_field.get_nb_components()};

        // number of components in the field where we'd like to write the result
        Index_t nb_quad_components{quadrature_point_field.get_nb_components()};

        // check if they match
        if (nb_quad_components != this->nb_operators * nb_nodal_components) {
            std::stringstream err_msg{};
            err_msg
                << "Size mismatch: Expected a quadrature field with "
                << this->nb_operators * nb_nodal_components << " components ("
                << this->nb_operators << " operators × " << nb_nodal_components
                << " components in the nodal field) but received a field with "
                << nb_quad_components << " components.";
            throw RuntimeError{err_msg.str()};
        }

        // Get a sparse representation of the operator; Note it needs to know
        // the whole domain (with ghosts) to get the correct pixel offset.
        const auto sparse_operator{this->create_sparse_operator(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components)};

        // Get the data pointer of both fields
        auto nodal_pixel{nodal_field.data()};
        auto quad_pixel{quadrature_point_field.data()};

        // Get number of elements in each pixel
        auto nodal_pixel_nb_elements{nb_nodal_components *
                                     this->nb_pixelnodal_pts};
        auto quad_pixel_nb_elements{nb_quad_components * this->nb_quad_pts};

        // Advance pointers to the first pixel that is not ghost.
        auto start_pixel_index{collection.get_pixels_index_diff()};
        nodal_pixel += start_pixel_index * nodal_pixel_nb_elements;
        quad_pixel += start_pixel_index * quad_pixel_nb_elements;

        // Get shape of the pixels without ghosts
        auto nb_pixels_without_ghosts{
            collection.get_pixels_shape_without_ghosts()};
        // Fill it up to 3D
        while (nb_pixels_without_ghosts.size() < 3) {
            nb_pixels_without_ghosts.push_back(1);
        }

        // Get number of ghosts
        Shape_t ghosts_count{collection.get_nb_ghosts_left() +
                             collection.get_nb_ghosts_right()};
        // Fill it up to 2D (we don't need to know the ghost in z)
        while (ghosts_count.size() < 2) {
            ghosts_count.push_back(0);
        }
        // Compute number of ghost elements to advance  for each related axis
        auto nodal_ghosts_count_x{ghosts_count[0] * nodal_pixel_nb_elements};
        auto nodal_ghosts_count_y{
            ghosts_count[1] * (nb_pixels_without_ghosts[0] + ghosts_count[0]) *
            nodal_pixel_nb_elements};
        auto quad_ghosts_count_x{ghosts_count[0] * quad_pixel_nb_elements};
        auto quad_ghosts_count_y{
            ghosts_count[1] * (nb_pixels_without_ghosts[0] + ghosts_count[0]) *
            quad_pixel_nb_elements};

        // For each pixel (without ghost)...
        for (Index_t z_index = 0; z_index < nb_pixels_without_ghosts[2];
             ++z_index) {
            for (Index_t y_index = 0; y_index < nb_pixels_without_ghosts[1];
                 ++y_index) {
                for (Index_t x_index = 0; x_index < nb_pixels_without_ghosts[0];
                     ++x_index) {
                    // For each non-zero entry in the operator
                    for (const auto & [quad_index, nodal_index, value] :
                         sparse_operator) {
                        // Add the contribution to the output
                        quad_pixel[quad_index] +=
                            alpha * nodal_pixel[nodal_index] * value;
                    }
                    // Advance the pointer to the next pixel
                    nodal_pixel += nodal_pixel_nb_elements;
                    quad_pixel += quad_pixel_nb_elements;
                }
                // Advance the pointer to skip the ghosts
                nodal_pixel += nodal_ghosts_count_x;
                quad_pixel += quad_ghosts_count_x;
            }
            // Advance the pointer to skip more ghosts
            nodal_pixel += nodal_ghosts_count_y;
            quad_pixel += quad_ghosts_count_y;
        }
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose(
        const TypedFieldBase<Real> & quadrature_point_field,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        // set nodal field to zero
        nodal_field.set_zero();
        this->transpose_increment(quadrature_point_field, 1., nodal_field,
                                  weights);
    }

    /* ---------------------------------------------------------------------- */
    void ConvolutionOperator::transpose_increment(
        const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
        TypedFieldBase<Real> & nodal_field,
        const std::vector<Real> & weights) const {
        // Both fields must be from the same field collection to ensure
        // compatible internal structure for pixel mapping
        if (&nodal_field.get_collection() !=
            &quadrature_point_field.get_collection()) {
            std::stringstream err_msg{};
            err_msg << "Field collection mismatch: nodal_field and "
                       "quadrature_point_field must be from the same "
                       "FieldCollection";
            throw RuntimeError{err_msg.str()};
            }

        // Get the collection object
        const auto & collection{dynamic_cast<GlobalFieldCollection &>(
            quadrature_point_field.get_collection())};

        // Check that fields are global
        if (collection.get_domain() !=
            FieldCollection::ValidityDomain::Global) {
            std::stringstream err_msg{};
            err_msg << "Field type error: nodal_field and "
                       "quadrature_point_field must be a global "
                       "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }

        // Check that fields have the same spatial dimensions as operator
        if (collection.get_spatial_dim() != this->spatial_dim) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: nodal_field and "
                       "quadrature_point_field are defined in "
                    << collection.get_spatial_dim()
                    << "D space, but this convolution operator is defined in "
                    << this->spatial_dim << "D space";
            throw RuntimeError{err_msg.str()};
        }

        // Check that fields have enough ghost cells on the left
        const auto & nb_ghosts_left{collection.get_nb_ghosts_left()};
        const auto min_ghosts_left{IntCoord_t(this->spatial_dim, 0) -
                                   IntCoord_t(this->pixel_offset)};
        for (auto direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_left[direction] < min_ghosts_left[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the convolution expects a minimum of "
                    << min_ghosts_left[direction]
                    << " cells on the left, but the provided fields have only "
                    << nb_ghosts_left[direction] << " ghosts on the left.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // Check that fields have enough ghost cells on the right
        const auto & nb_ghosts_right{collection.get_nb_ghosts_right()};
        const auto min_ghosts_right{IntCoord_t(this->conv_pts_shape) -
                                    IntCoord_t(this->spatial_dim, 1) +
                                    IntCoord_t(this->pixel_offset)};
        for (auto direction = 0; direction < collection.get_spatial_dim();
             ++direction) {
            if (nb_ghosts_right[direction] < min_ghosts_right[direction]) {
                std::stringstream err_msg{};
                err_msg
                    << "Ambiguous field shape: on axis " << direction
                    << ", the convolution expects a minimum of "
                    << min_ghosts_right[direction]
                    << " cells on the right, but the provided fields have only "
                    << nb_ghosts_right[direction] << " ghosts on the right.";
                throw RuntimeError{err_msg.str()};
            }
        }

        // number of components in the field we'd like to apply the convolution
        Index_t nb_nodal_components{nodal_field.get_nb_components()};

        // number of components in the field where we'd like to write the result
        Index_t nb_quad_components{quadrature_point_field.get_nb_components()};

        // check if they match
        if (nb_quad_components != this->nb_operators * nb_nodal_components) {
            std::stringstream err_msg{};
            err_msg
                << "Size mismatch: Expected a quadrature field with "
                << this->nb_operators * nb_nodal_components << " components ("
                << this->nb_operators << " operators × " << nb_nodal_components
                << " components in the nodal field) but received a field with "
                << nb_quad_components << " components.";
            throw RuntimeError{err_msg.str()};
        }

        // Get a sparse representation of the operator; Note it needs to know
        // the whole domain (with ghosts) to get the correct pixel offset.
        const auto sparse_operator{this->create_sparse_operator(
            collection.get_nb_subdomain_grid_pts_with_ghosts(),
            nb_nodal_components)};

        // Get the data pointer of both fields
        auto nodal_pixel{nodal_field.data()};
        auto quad_pixel{quadrature_point_field.data()};

        // Get number of elements in each pixel
        auto nodal_pixel_nb_elements{nb_nodal_components *
                                     this->nb_pixelnodal_pts};
        auto quad_pixel_nb_elements{nb_quad_components * this->nb_quad_pts};

        // Advance pointers to the first pixel that is not ghost.
        auto start_pixel_index{collection.get_pixels_index_diff()};
        nodal_pixel += start_pixel_index * nodal_pixel_nb_elements;
        quad_pixel += start_pixel_index * quad_pixel_nb_elements;

        // Get shape of the pixels without ghosts
        auto nb_pixels_without_ghosts{
            collection.get_pixels_shape_without_ghosts()};
        // Fill it up to 3D
        while (nb_pixels_without_ghosts.size() < 3) {
            nb_pixels_without_ghosts.push_back(1);
        }

        // Get number of ghosts
        Shape_t ghosts_count{collection.get_nb_ghosts_left() +
                             collection.get_nb_ghosts_right()};
        // Fill it up to 2D (we don't need to know the ghost in z)
        while (ghosts_count.size() < 2) {
            ghosts_count.push_back(0);
        }
        // Compute number of ghost elements to advance  for each related axis
        auto nodal_ghosts_count_x{ghosts_count[0] * nodal_pixel_nb_elements};
        auto nodal_ghosts_count_y{
            ghosts_count[1] * (nb_pixels_without_ghosts[0] + ghosts_count[0]) *
            nodal_pixel_nb_elements};
        auto quad_ghosts_count_x{ghosts_count[0] * quad_pixel_nb_elements};
        auto quad_ghosts_count_y{
            ghosts_count[1] * (nb_pixels_without_ghosts[0] + ghosts_count[0]) *
            quad_pixel_nb_elements};

        // For each pixel (without ghost)...
        for (Index_t z_index = 0; z_index < nb_pixels_without_ghosts[2];
             ++z_index) {
            for (Index_t y_index = 0; y_index < nb_pixels_without_ghosts[1];
                 ++y_index) {
                for (Index_t x_index = 0; x_index < nb_pixels_without_ghosts[0];
                     ++x_index) {
                    // For each non-zero entry in the operator
                    for (const auto & [quad_index, nodal_index, value] :
                         sparse_operator) {
                        // Add the contribution to the output. Note because the
                        // operator is transposed, thus quadrature point field
                        // acts as the input.
                        nodal_pixel[nodal_index] +=
                            alpha * quad_pixel[quad_index] * value;
                    }
                    // Advance the pointer to the next pixel
                    nodal_pixel += nodal_pixel_nb_elements;
                    quad_pixel += quad_pixel_nb_elements;
                }
                // Advance the pointer to skip the ghosts
                nodal_pixel += nodal_ghosts_count_x;
                quad_pixel += quad_ghosts_count_x;
            }
            // Advance the pointer to skip more ghosts
            nodal_pixel += nodal_ghosts_count_y;
            quad_pixel += quad_ghosts_count_y;
        }
    }

    /* ---------------------------------------------------------------------- */
    const Eigen::MatrixXd & ConvolutionOperator::get_pixel_operator() const {
        return this->pixel_operator;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_quad_pts() const {
        return this->nb_quad_pts;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_operators() const {
        return this->nb_operators;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_nb_nodal_pts() const {
        return this->nb_pixelnodal_pts;
    }

    /* ---------------------------------------------------------------------- */
    Index_t ConvolutionOperator::get_spatial_dim() const {
        return this->spatial_dim;
    }
}  // namespace muGrid
