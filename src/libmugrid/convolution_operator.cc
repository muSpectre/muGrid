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
      const Eigen::MatrixXd & pixel_operator, const Shape_t & conv_pts_shape, 
      const Index_t & nb_field_comps, const Index_t & nb_pixelnodal_pts,
      const Index_t & nb_quad_pts, const Index_t & nb_operators)
      : Parent{}, pixel_operator{pixel_operator}, conv_pts_shape{conv_pts_shape},
          nb_field_comps{nb_field_comps}, nb_pixelnodal_pts{nb_pixelnodal_pts},
          nb_quad_pts{nb_quad_pts}, nb_operators{nb_operators},
          spatial_dim{static_cast<Index_t>(conv_pts_shape.size())},
          nb_conv_pts{get_nb_from_shape(conv_pts_shape)} {
    // Check the dimension of the pixel operator
    if (pixel_operator.cols() != this->nb_pixelnodal_pts * this->nb_conv_pts) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected the operator has "
              << this->nb_pixelnodal_pts * this->nb_conv_pts
              << " columns. but received an operator with "
              << pixel_operator.cols()
              << " columns";
      throw RuntimeError{err_msg.str()};
    }
    if (pixel_operator.rows() != this->nb_operators * this->nb_quad_pts) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected the operator has "
              << this->nb_operators * this->nb_quad_pts
              << " rows. but received an operator with "
              << pixel_operator.rows()
              << " rows";
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
      err_msg << "Field type error: quadrature_point_field must be a global "
                 "field (registered in a global FieldCollection)";
      throw RuntimeError{err_msg.str()};
    }
    if (not nodal_field.is_global()) {
      std::stringstream err_msg{};
      err_msg << "Field type error: nodal_field must be a global "
                 "field (registered in a global FieldCollection)";
      throw RuntimeError{err_msg.str()};
    }

    // number of components in the field we'd like to apply the convolution
    Index_t nb_nodal_component{nodal_field.get_nb_components()};

    // check number of components
    if (nb_nodal_component != this->nb_field_comps) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected a field with "
              << nb_field_comps
              << " components but received a field with "
              << nb_nodal_component
              << " components.";
      throw RuntimeError{err_msg.str()};
    }

    // number of components in the field where we'd like to write the result
    Index_t nb_quad_component{quadrature_point_field.get_nb_components()};

    // check number of components
    if (nb_quad_component != this->nb_operators * this->nb_field_comps) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected a field with "
              << this->nb_operators * this->nb_field_comps
              << " components but received a field with "
              << nb_quad_component
              << " components."
              << this->nb_operators << " " << this->nb_field_comps;
      throw RuntimeError{err_msg.str()};
    }

    // get nodal field map, where the values at one location is interpreted 
    // as a matrix with [nb_field_comps] rows
    auto nodal_map{nodal_field.get_pixel_map(this->nb_field_comps)};
    // get quadrature point field map, where the values at one location is 
    // interpreted as a matrix with [nb_operators] rows
    auto quad_map{quadrature_point_field.get_pixel_map(this->nb_operators)};

    auto & collection{dynamic_cast<GlobalFieldCollection &>(
        quadrature_point_field.get_collection())};
    auto & pixels{collection.get_pixels()};
    auto && nb_subdomain_grid_pts{pixels.get_nb_subdomain_grid_pts()};

    // relative coordinates of the nodal points inside the convolution space
    CcoordOps::DynamicPixels conv_space{DynCcoord_t(this->conv_pts_shape)};

    // For each pixel...
    for (auto && id_base_ccoord : pixels.enumerate()) {
      auto && id{std::get<0>(id_base_ccoord)};
      auto && base_ccoord{std::get<1>(id_base_ccoord)};

      // get the quadrature point value relative to this pixel
      auto && value{quad_map[id]};
      // Expected arrangement:
      // [op1|u_x[q1], op1|u_y[q1], op1|u_x[q2], op1|u_y[q2]]
      // [op2|u_x[q1], op2|u_y[q1], op2|u_x[q2], op2|u_y[q2]]

      // For each convolution points involved in the current pixel...
      for (auto && tup : akantu::enumerate(conv_space)) {
        // get the nodal values
        auto && index{std::get<0>(tup)};
        auto && offset{std::get<1>(tup)};
        // FIXME(yizhen): in parallel, it doesn't work
        auto && ccoord{(base_ccoord + offset) % nb_subdomain_grid_pts};
        // transpose to make the shape compatible for multiplication
        auto && nodal_vals{nodal_map[pixels.get_index(ccoord)].transpose()};

        // For each quadrature point in the current pixel...
        for (Index_t idx_quad=0; idx_quad < this->nb_quad_pts; ++idx_quad) {
          // get the chunk that corresponding to this quadrature point
          auto && quad_vals{value.block(
              0, idx_quad * this->nb_field_comps,
              this->nb_operators, this->nb_field_comps)};
          // get the chunk that represents the contribution of this node to
          // this quadrature point
          auto && B_block{this->pixel_operator.block(
              idx_quad * this->nb_operators, index * this->nb_pixelnodal_pts,
              this->nb_operators, this->nb_pixelnodal_pts)};
          // compute
          quad_vals += alpha * B_block * nodal_vals;
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
      err_msg << "Field type error: quadrature_point_field must be a global "
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

    // number of components in the gradient field
    Index_t nb_quad_component{quadrature_point_field.get_nb_components()};

    // number of components in the nodal field
    Index_t nb_nodal_component{nodal_field.get_nb_components()};

    if (nb_quad_component != this->nb_operators * nb_nodal_component) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected a vector with "
              << this->nb_operators * nb_nodal_component
              << " entries (number of gradient components in single quad "
                 "point), but received a "
                 "vector of size "
              << nb_quad_component;
      throw RuntimeError{err_msg.str()};
    }

    // get nodal field map, where the values at one location is interpreted 
    // as a matrix with [nb_field_comps] rows
    auto nodal_map{nodal_field.get_pixel_map(this->nb_field_comps)};
    // get quadrature point field map, where the values at one location is 
    // interpreted as a matrix with [nb_operators] rows
    auto quad_map{quadrature_point_field.get_pixel_map(this->nb_operators)};

    // preprocess weights
    bool use_default_weights{weights.size() == 0};
    std::vector<Real> default_weights{};
    if (use_default_weights) {
      default_weights.resize(this->nb_quad_pts, 1.);
    }
    const auto & quad_weights{use_default_weights ? default_weights : weights};

    auto & collection{dynamic_cast<GlobalFieldCollection &>(
        quadrature_point_field.get_collection())};
    auto & pixels{collection.get_pixels()};
    auto && nb_subdomain_grid_pts{pixels.get_nb_subdomain_grid_pts()};

    // pixel offsets of the points inside the convolution space
    CcoordOps::DynamicPixels conv_space{DynCcoord_t(this->conv_pts_shape)};

    // For each pixel...
    for (auto && id_base_ccoord : pixels.enumerate()) {
      auto && id{std::get<0>(id_base_ccoord)};
      auto && base_ccoord{std::get<1>(id_base_ccoord)};

      // get the quadrature point value relative to this pixel
      auto && value{quad_map[id]};

      // For each convolution points involved in the current pixel...
      for (auto && tup : akantu::enumerate(conv_space)) {
        // get the nodal values relative to B-chunk
        auto && index{std::get<0>(tup)};
        auto && offset{std::get<1>(tup)};
        auto && ccoord{(base_ccoord + offset) % nb_subdomain_grid_pts};
        auto && nodal_vals{nodal_map[pixels.get_index(ccoord)]};

        // For each quadrature point
        for (Index_t idx_quad=0; idx_quad < this->nb_quad_pts; ++idx_quad) {
          // get the chunk that corresponding to this quadrature point
          auto && quad_vals{value.block(
              0, idx_quad * this->nb_field_comps,
              this->nb_operators, this->nb_field_comps)};
          // get the chunk that represents the contribution of this node to
          // this quadrature point; and transpose it.
          auto && B_block_T{this->pixel_operator.block(
              idx_quad * this->nb_operators, index * this->nb_pixelnodal_pts,
              this->nb_operators, this->nb_pixelnodal_pts).transpose()};
          // compute
          nodal_vals += alpha * quad_weights[idx_quad] * B_block_T * quad_vals;
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  const Eigen::MatrixXd &
  ConvolutionOperator::get_pixel_operator() const {
    return this->pixel_operator;
  }

  /* ---------------------------------------------------------------------- */
  Index_t ConvolutionOperator::get_nb_quad_pts() const {
    return this->nb_quad_pts;
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
