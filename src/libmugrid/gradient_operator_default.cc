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

#include "gradient_operator_default.hh"
#include "field_collection_global.hh"
#include "field_map.hh"
#include "ccoord_operations.hh"
#include "iterators.hh"
#include "exception.hh"

#include <sstream>

namespace muGrid {

  Eigen::MatrixXd permutation(const Eigen::VectorXi & nodal_indices,
                              const Eigen::MatrixXi & pixel_offsets,
                              const Index_t & nb_pixelnodes) {
    auto && spatial_dim{pixel_offsets.cols()};
    // the number of all possible nodes is the nodes of this pixel plus the
    // right/upper/frontal pixel
    const auto && nb_nodes{nb_pixelnodes * ipow(2, spatial_dim)};
    auto && nb_elemnodes{pixel_offsets.rows()};
    Eigen::MatrixXd perm{Eigen::MatrixXd::Zero(nb_elemnodes, nb_nodes)};

    auto && linear_index{[&spatial_dim, &nb_pixelnodes](
                             auto && nodal_id, auto && offset) -> Index_t {
      Index_t ret_val{nodal_id};
      Index_t stride{nb_pixelnodes};
      for (Index_t i{0}; i < spatial_dim; ++i) {
        ret_val += stride * offset(i);
        // 2 is the number of pixels we consider in each
        // direction
        stride *= 2;
      }
      return ret_val;  // linear indexing of pixel nodes
    }};

    for (Index_t i{0}; i < nb_elemnodes; ++i) {
      const auto & nodal_id{nodal_indices(i)};
      auto && offset{pixel_offsets.row(i)};
      perm(i, linear_index(nodal_id, offset)) = 1.;
    }
    return perm;
  }

  /* ---------------------------------------------------------------------- */
  GradientOperatorDefault::GradientOperatorDefault(
      const Index_t & spatial_dim, const Index_t & nb_quad_pts,
      const Index_t & nb_elements, const Index_t & nb_elemnodal_pts,
      const Index_t & nb_pixelnodal_pts,
      const std::vector<std::vector<Eigen::MatrixXd>> & shape_fn_gradients,
      const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts)
      : Parent{}, spatial_dim{spatial_dim}, nb_quad_pts{nb_quad_pts},
        nb_elements{nb_elements}, nb_elemnodal_pts{nb_elemnodal_pts},
        nb_pixelnodal_pts{nb_pixelnodal_pts},
        nb_possible_nodal_contribution{ipow(2, this->spatial_dim) *
                                       this->nb_pixelnodal_pts},
        // TODO(junge): Check with Martin whether this can be true. Why does it
        // not depend on rank?
        nb_grad_component_per_pixel{this->spatial_dim * this->nb_quad_pts *
                                    this->nb_elements} {
    this->pixel_gradient.resize(nb_grad_component_per_pixel,
                                nb_possible_nodal_contribution);
    Index_t counter{0};
    for (Index_t e{0}; e < this->nb_elements; ++e) {
      auto & nodal_indices{std::get<0>(nodal_pts.at(e))};  // n of (n,(i,j,k))
      auto & nodal_pix_coords{
          std::get<1>(nodal_pts.at(e))};  // (i,j,k) of (n,(i,j,k))

      if (nodal_indices.rows() != this->nb_elemnodal_pts) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: Expected a vector with "
                << this->nb_elemnodal_pts
                << " entries (number of nodes per element). but received a "
                   "vector of size "
                << nodal_indices.rows();
        throw RuntimeError{err_msg.str()};
      }
      if (nodal_pix_coords.rows() != this->nb_elemnodal_pts) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: Expected a vector with "
                << this->nb_elemnodal_pts
                << " entries (number of nodes per element). but received a "
                   "vector of size "
                << nodal_pix_coords.rows();
        throw RuntimeError{err_msg.str()};
      }
      if (nodal_pix_coords.cols() != this->spatial_dim) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: Expected a vector with " << this->spatial_dim
                << " entries (spatial dimension). but received a "
                   "vector of size "
                << nodal_pix_coords.cols();
        throw RuntimeError{err_msg.str()};
      }

      auto && permutation_matrix{permutation(nodal_indices, nodal_pix_coords,
                                             this->nb_pixelnodal_pts)};
      for (Index_t q{0}; q < this->nb_quad_pts; ++q) {
        auto & grad{shape_fn_gradients.at(q).at(e)};  // B_(q,e)
        if (grad.rows() != this->spatial_dim) {
          std::stringstream err_msg{};
          err_msg << "Size mismatch: Expected a vector with "
                  << this->spatial_dim
                  << " entries (spatial dimension). but received a "
                     "vector of size "
                  << grad.rows();
          throw RuntimeError{err_msg.str()};
        }
        if (grad.cols() != this->nb_elemnodal_pts) {
          std::stringstream err_msg{};
          err_msg << "Size mismatch: Expected a vector with "
                  << this->nb_elemnodal_pts
                  << " entries (spatial dimension). but received a "
                     "vector of size "
                  << grad.cols();
          throw RuntimeError{err_msg.str()};
        }

        auto && gradient_block{this->pixel_gradient.block(
            this->spatial_dim * counter++, 0, this->spatial_dim,
            nb_possible_nodal_contribution)};
        gradient_block.noalias() = grad * permutation_matrix;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void GradientOperatorDefault::apply_gradient(
      const TypedFieldBase<Real> & nodal_field,
      TypedFieldBase<Real> & quadrature_point_field) const {
    quadrature_point_field.set_zero();
    this->apply_gradient_increment(nodal_field, 1., quadrature_point_field);
  }

  /* ---------------------------------------------------------------------- */
  void GradientOperatorDefault::apply_gradient_increment(
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

    // number of components in the field we'd like to derive
    Index_t nb_nodal_component{nodal_field.get_nb_components()};

    // number of components in the field where we'd like to write the derivative
    Index_t nb_quad_component{quadrature_point_field.get_nb_components()};

    // we take a gradient in all directions of every component
    if (nb_quad_component != this->spatial_dim * nb_nodal_component) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected a vector with "
              << this->spatial_dim * nb_nodal_component
              << " entries (number of gradient components in single quad "
                 "point), but received a "
                 "vector of size "
              << nb_quad_component;
      throw RuntimeError{err_msg.str()};
    }

    // nodal field is always represented as a column vector
    auto nodal_map{nodal_field.get_pixel_map(nb_nodal_component)};
    /*
     * quad field has dim column vectors: each row represents the gradient of
     * one component of the nodal field in each direction
     */
    auto quad_map{quadrature_point_field.get_pixel_map(nb_nodal_component)};

    auto & collection{dynamic_cast<GlobalFieldCollection &>(
        quadrature_point_field.get_collection())};
    auto & pixels{collection.get_pixels()};
    auto && nb_subdomain_grid_pts{pixels.get_nb_subdomain_grid_pts()};

    CcoordOps::DynamicPixels offsets{CcoordOps::get_cube(this->spatial_dim, 2)};
    for (auto && id_base_ccoord : pixels.enumerate()) {
      auto && id{std::get<0>(id_base_ccoord)};
      auto && base_ccoord{std::get<1>(id_base_ccoord)};

      // get the quadrature point value relative to this pixel
      auto && grad_val{quad_map[id]};  // [ u_x[q1], u_y[q1], u_x[q2], u_y[q2]]

      for (auto && tup : akantu::enumerate(offsets)) {
        auto && index{std::get<0>(tup)};
        auto && offset{std::get<1>(tup)};
        auto && ccoord{(base_ccoord + offset) % nb_subdomain_grid_pts};

        // get the right chunk of B: This chunk represents the contribution of
        // the nodal point values in this current offset pixel to the the
        // gradients of the base pixel
        auto && B_block{this->pixel_gradient.block(
            0, index * this->nb_pixelnodal_pts,
            this->nb_grad_component_per_pixel, this->nb_pixelnodal_pts)};
        // get the nodal values relative to B-chunk
        auto && nodal_vals{nodal_map[pixels.get_index(ccoord)]};

        grad_val += alpha * nodal_vals * B_block.transpose();
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void GradientOperatorDefault::apply_transpose(
      const TypedFieldBase<Real> & quadrature_point_field,
      TypedFieldBase<Real> & nodal_field,
      const std::vector<Real> & weights) const {
    // set nodal field to zero
    nodal_field.set_zero();
    this->apply_transpose_increment(quadrature_point_field, 1., nodal_field,
                                    weights);
  }

  /* ---------------------------------------------------------------------- */
  void GradientOperatorDefault::apply_transpose_increment(
      const TypedFieldBase<Real> & quadrature_point_field, const Real & alpha,
      TypedFieldBase<Real> & nodal_field,
      const std::vector<Real> & weights) const {
    auto && nb_pixel_quad_pts{this->get_nb_pixel_quad_pts()};
    std::vector<Real> use_weights{};
    bool default_weights{weights.size() == 0};
    if (default_weights) {
      use_weights.resize(nb_pixel_quad_pts, 1.);
    }
    const auto & quad_weights{default_weights ? use_weights : weights};
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

    if (nb_quad_component != this->spatial_dim * nb_nodal_component) {
      std::stringstream err_msg{};
      err_msg << "Size mismatch: Expected a vector with "
              << this->spatial_dim * nb_nodal_component
              << " entries (number of gradient components in single quad "
                 "point), but received a "
                 "vector of size "
              << nb_quad_component;
      throw RuntimeError{err_msg.str()};
    }

    auto nodal_map{nodal_field.get_pixel_map(nb_nodal_component)};
    auto quad_map{quadrature_point_field.get_pixel_map(nb_nodal_component)};

    auto & collection{dynamic_cast<GlobalFieldCollection &>(
        quadrature_point_field.get_collection())};
    auto & pixels{collection.get_pixels()};
    auto && nb_subdomain_grid_pts{pixels.get_nb_subdomain_grid_pts()};

    // pixel index offsets for whole stencil of [ij,i+j,ij+,i+j+] in 2D  ...
    CcoordOps::DynamicPixels offsets{CcoordOps::get_cube(this->spatial_dim, 2)};

    // loop over pixels
    for (auto && id_base_ccoord : pixels.enumerate()) {
      auto && id{std::get<0>(id_base_ccoord)};  // linear index of pixel
      auto && base_ccoord{
          std::get<1>(id_base_ccoord)};  // ijk spatial coords of pixel

      // get the quadrature point value relative to this pixel
      auto && grad_val{quad_map[id] * quad_weights[id % nb_pixel_quad_pts]};

      // loop over offsets
      for (auto && tup : akantu::enumerate(offsets)) {
        auto && index{std::get<0>(tup)};
        auto && offset{std::get<1>(tup)};
        auto && ccoord{(base_ccoord + offset) % nb_subdomain_grid_pts};

        // get the right chunk of B: This chunk represents the contribution of
        // the gradients of the base pixel to the
        // in this current offset pixel nodal point values
        auto && B_block{this->pixel_gradient.block(
            0, index * this->nb_pixelnodal_pts,
            this->nb_grad_component_per_pixel, this->nb_pixelnodal_pts)};

        // get the nodal values relative to B-chunk
        auto && nodal_vals{nodal_map[pixels.get_index(ccoord)]};

        nodal_vals += alpha * grad_val * B_block;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  const Eigen::MatrixXd & GradientOperatorDefault::get_pixel_gradient() const {
    return this->pixel_gradient;
  }

  /* ---------------------------------------------------------------------- */
  Index_t GradientOperatorDefault::get_nb_pixel_quad_pts() const {
    return this->nb_quad_pts * this->nb_elements;
  }

  /* ---------------------------------------------------------------------- */
  Index_t GradientOperatorDefault::get_nb_pixel_nodal_pts() const {
    return this->nb_pixelnodal_pts;
  }

  /* ---------------------------------------------------------------------- */
  Index_t GradientOperatorDefault::get_spatial_dim() const {
    return this->spatial_dim;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & GradientOperatorDefault::get_nb_quad_pts_per_element() const {
    return this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & GradientOperatorDefault::get_nb_elements() const {
    return this->nb_elements;
  }

}  // namespace muGrid
