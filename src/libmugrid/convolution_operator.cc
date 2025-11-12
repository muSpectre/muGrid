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
#include "field_map.hh"
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
        if (not quadrature_point_field.is_global()) {
            std::stringstream err_msg{};
            err_msg
                << "Field type error: quadrature_point_field must be a global "
                   "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }
        if (not nodal_field.is_global()) {
            std::stringstream err_msg{};
            err_msg << "Field type error: nodal_field must be a global "
                       "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }

        // Check that both fields have the same spatial dimensions
        if (nodal_field.get_collection().get_spatial_dim() !=
            quadrature_point_field.get_collection().get_spatial_dim()) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: nodal field is defined "
                       "in "
                    << nodal_field.get_collection().get_spatial_dim()
                    << "D space, but quadrature field is defined in "
                    << quadrature_point_field.get_collection()
                           .get_spatial_dim()
                    << "D space";
            throw RuntimeError{err_msg.str()};
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

        // get nodal field map, where the values at one location is interpreted
        // as a matrix with [nb_nodal_components] rows
        auto nodal_map{nodal_field.get_pixel_map(nb_nodal_components)};
        // get quadrature point field map, where the values at one location is
        // interpreted as a matrix with [nb_nodal_components] rows
        auto quad_map{
            quadrature_point_field.get_pixel_map(nb_nodal_components)};

        auto & collection{dynamic_cast<GlobalFieldCollection &>(
            quadrature_point_field.get_collection())};
        auto & pixels{collection.get_pixels()};

        // relative coordinates of the nodal points inside the convolution space
        CcoordOps::Pixels conv_space{IntCoord_t(this->conv_pts_shape),
                                     IntCoord_t(this->pixel_offset)};

        // For each pixel...
        for (auto && [id, base_ccoord] : pixels.enumerate()) {
            // which should be interpreted as a matrix with shape (c, o q)
            // (It is already set to zero in the caller function "apply")
            auto && quad_vals{quad_map[id]};

            // For each convolution points involved in the current pixel...
            for (auto && [index, offset] : akantu::enumerate(conv_space)) {
                auto && ccoord{pixels.get_neighbour(base_ccoord, offset)};
                // which should be interpreted as a matrix with shape (c, s)
                auto && nodal_vals{nodal_map[pixels.get_index(ccoord)]};

                // For each contributive nodal-pixel points, i.e. sub-pts of
                // nodal field
                for (Index_t i_pixelnodal = 0;
                     i_pixelnodal < this->nb_pixelnodal_pts; ++i_pixelnodal) {
                    // all the components are exactly in one column
                    auto && effective_nodal_vals{nodal_vals.col(i_pixelnodal)};
                    // the "operator" is interpreted as a matrix with shape (o
                    // q, s ijk), so the corresponding chunk is a colum;
                    // transpose so it becomes a row.
                    auto && effective_op_vals{
                        this->pixel_operator
                            .col(index * this->nb_pixelnodal_pts + i_pixelnodal)
                            .transpose()};
                    // compute (col * row, such that the operator is broadcasted
                    // to all components)
                    quad_vals +=
                        alpha * effective_nodal_vals * effective_op_vals;
                }
            }
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
        // check quadrature point field type == global
        if (not quadrature_point_field.is_global()) {
            std::stringstream err_msg{};
            err_msg
                << "Field type error: quadrature_point_field must be a global "
                   "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }
        // check nodal field type == global
        if (not nodal_field.is_global()) {
            std::stringstream err_msg{};
            err_msg << "Field type error: nodal_field must be a global "
                       "field (registered in a global FieldCollection)";
            throw RuntimeError{err_msg.str()};
        }

        // Check that both fields have the same spatial dimensions
        if (quadrature_point_field.get_collection().get_spatial_dim() !=
            nodal_field.get_collection().get_spatial_dim()) {
            std::stringstream err_msg{};
            err_msg << "Spatial dimension mismatch: quadrature field is "
                       "defined in "
                    << quadrature_point_field.get_collection()
                           .get_spatial_dim()
                    << "D space, but nodal field is defined in "
                    << nodal_field.get_collection().get_spatial_dim()
                    << "D space";
            throw RuntimeError{err_msg.str()};
        }

        // number of components in the gradient field
        Index_t nb_quad_components{quadrature_point_field.get_nb_components()};

        // number of components in the nodal field
        Index_t nb_nodal_components{nodal_field.get_nb_components()};

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

        // Both fields must be from the same field collection to ensure
        // compatible internal structure for pixel mapping
        if (&quadrature_point_field.get_collection() !=
            &nodal_field.get_collection()) {
            std::stringstream err_msg{};
            err_msg << "Field collection mismatch: quadrature_point_field and "
                       "nodal_field must be from the same FieldCollection";
            throw RuntimeError{err_msg.str()};
        }

        // get nodal field map, where the values at one location is interpreted
        // as a matrix with [nb_nodal_comps] rows
        auto nodal_map{nodal_field.get_pixel_map(nb_nodal_components)};
        // get quadrature point field map, where the values at one location is
        // interpreted as a matrix with [nb_nodal_comps] rows
        auto quad_map{
            quadrature_point_field.get_pixel_map(nb_nodal_components)};

        // preprocess weights
        bool use_default_weights{weights.size() == 0};
        std::vector<Real> default_weights{};
        if (use_default_weights) {
            default_weights.resize(this->nb_quad_pts, 1.);
        }
        const auto & quad_weights{use_default_weights ? default_weights
                                                      : weights};

        auto & collection{dynamic_cast<GlobalFieldCollection &>(
            quadrature_point_field.get_collection())};
        auto & pixels{collection.get_pixels()};

        // pixel offsets of the points inside the convolution space
        CcoordOps::Pixels conv_space{IntCoord_t(this->conv_pts_shape)};

        // For each pixel...
        for (auto && id_base_ccoord : pixels.enumerate()) {
            auto && id{std::get<0>(id_base_ccoord)};
            auto && base_ccoord{std::get<1>(id_base_ccoord)};

            // get the quadrature point value relative to this pixel
            // which should be interpreted as a matrix with shape (c, o q)
            auto && quad_vals{quad_map[id]};

            // For each convolution points involved in the current pixel...
            for (auto && tup : akantu::enumerate(conv_space)) {
                // get the nodal values relative to B-chunk
                auto && index{std::get<0>(tup)};
                auto && offset{std::get<1>(tup)};
                auto && ccoord{pixels.get_neighbour(base_ccoord, offset)};
                // which should be interpreted as a matrix with shape (c, s)
                auto && nodal_vals{nodal_map[pixels.get_index(ccoord)]};

                // Because of "quadrature weights", we need to loop quadrature
                // points For each quadrature points
                for (Index_t i_quad = 0; i_quad < this->nb_quad_pts; ++i_quad) {
                    // get the columns corresponding to this quadrature point,
                    // should have shape (c, o)
                    auto && effetive_quad_vals{quad_vals.block(
                        0, i_quad * this->nb_operators, nb_nodal_components,
                        this->nb_operators)};
                    // the operator is interpreted as a matrix with shape (o q,
                    // s ijk), get the corresponding block with shape (o, s)
                    auto && effective_op_vals{this->pixel_operator.block(
                        i_quad * this->nb_operators,
                        index * this->nb_pixelnodal_pts, this->nb_operators,
                        this->nb_pixelnodal_pts)};
                    // compute
                    nodal_vals += alpha * quad_weights[i_quad] *
                                  effetive_quad_vals * effective_op_vals;
                }
            }
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
