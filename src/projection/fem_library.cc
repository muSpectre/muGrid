/**
 * @file   fem_library.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   18 Jan 2021
 *
 * @brief  implementation for finite-element discretisation factory functions
 *
 * Copyright © 2021 Till Junge, Martin Ladecký
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#include "fem_library.hh"

#include <libmugrid/exception.hh>

#include <vector>

namespace muSpectre {
  namespace FEMLibrary {

    /* ---------------------------------------------------------------------- */
    std::shared_ptr<FEMStencilBase> linear_1d(std::shared_ptr<CellData> cell) {
      // create all the input
      if (cell->get_spatial_dim() != oneD) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: Linear_interval is" << oneD
                << " dimensional stencil,"
                << "but cell spatial_dim =" << cell->get_spatial_dim();
        throw muGrid::RuntimeError{err_msg.str()};
      }

      const Index_t nb_quad_pts_per_element{OneQuadPt};
      const Index_t nb_elements_per_pixel{1};
      const Index_t nb_element_nodal_pts{2};
      const Index_t nb_pixel_nodal_pts{OneNode};

      auto && domain_size{cell->get_domain_lengths()};
      auto && nb_grid_points{cell->get_nb_domain_grid_pts()};
      const Real del_x{domain_size[0] / nb_grid_points[0]};

      std::vector<Real> quadrature_weights{del_x};
      // ----------- nodal_pts ------------------
      Eigen::VectorXi nodal_indices{2};
      nodal_indices << 0, 0;
      Eigen::MatrixXi offsets{2, 1};
      offsets << 0, 1;
      const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts{
        std::make_tuple(nodal_indices, offsets)};

      // ----------- shape functions gradients ------------------
      // per-quad-pt-vector of per-element-vector of matrices.
      Eigen::MatrixXd B{nb_quad_pts_per_element, nb_element_nodal_pts};
      B << -1. / del_x, 1. / del_x;
      std::vector<std::vector<Eigen::MatrixXd>> shape_fn_gradients{{B}};

      // FemStencil construction
      return std::make_shared<FEMStencil<muGrid::GradientOperatorDefault>>(
          nb_quad_pts_per_element, nb_elements_per_pixel, nb_element_nodal_pts,
          nb_pixel_nodal_pts, shape_fn_gradients, nodal_pts, quadrature_weights,
          cell);
    }

    /* ---------------------------------------------------------------------- */
    std::shared_ptr<FEMStencilBase>
    linear_triangle_straight(std::shared_ptr<CellData> cell) {
      // create all the input
      if (cell->get_spatial_dim() != twoD) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: linear_triangle is" << twoD
                << " dimensional stencil,"
                << "but cell spatial_dim =" << cell->get_spatial_dim();
        throw muGrid::RuntimeError{err_msg.str()};
      }

      const Index_t nb_quad_pts_per_element{OneQuadPt};
      const Index_t nb_elements_per_pixel{2};
      const Index_t nb_element_nodal_pts{3};
      const Index_t nb_pixel_nodal_pts{OneNode};

      auto domain_size{cell->get_domain_lengths()};
      auto nb_grid_points{cell->get_nb_domain_grid_pts()};
      const Real del_x{domain_size[0] / nb_grid_points[0]};
      const Real del_y{domain_size[1] / nb_grid_points[1]};

      const std::vector<Real> quadrature_weights{0.5 * del_x * del_y,
                                                 0.5 * del_x * del_y};

      // ----------- nodal_pts ------------------
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(nb_element_nodal_pts)};
      Eigen::MatrixXi pixel_offsets{nb_element_nodal_pts,
                                    cell->get_spatial_dim()};
      //! first element
      // clang-format off
      pixel_offsets <<  0, 0,
          1, 0,
          0, 1;
      // clang-format on
      nodal_pts.emplace_back(node_ids, pixel_offsets);

      //! second element
      // clang-format off
      pixel_offsets <<  1, 1,
          0, 1,
          1, 0;
      // clang-format on
      nodal_pts.emplace_back(node_ids, pixel_offsets);

      // ----------- shape functions gradients ------------------
      // per-quad-pt-vector of per-element-vector of matrices.
      std::vector<std::vector<Eigen::MatrixXd>> shape_fn_gradients(
          nb_quad_pts_per_element);
      Eigen::MatrixXd B1(twoD, nb_element_nodal_pts);
      // clang-format off
      B1 << -1./del_x, 1./del_x,        0,
          -1./del_y,        0, 1./del_y;
      // clang-format on

      Eigen::MatrixXd B2(twoD, nb_element_nodal_pts);
      // clang-format off
      B2 << 1./del_x, -1./del_x,         0,
          1./del_y,         0, -1./del_y;
      // clang-format on

      constexpr Index_t q_id{0};
      shape_fn_gradients[q_id].push_back(B1);
      shape_fn_gradients[q_id].push_back(B2);

      return std::make_shared<FEMStencil<muGrid::GradientOperatorDefault>>(
          nb_quad_pts_per_element, nb_elements_per_pixel, nb_element_nodal_pts,
          nb_pixel_nodal_pts, shape_fn_gradients, nodal_pts, quadrature_weights,
          cell);
    }

    /* ---------------------------------------------------------------------- */
    std::shared_ptr<FEMStencilBase>
    bilinear_quadrangle(std::shared_ptr<CellData> cell) {
      // TODO(Ladecky) Not tested
      // create all the input
      if (cell->get_spatial_dim() != twoD) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: bilinear_quadrangle is" << twoD
                << " dimensional stencil,"
                << "but cell spatial_dim =" << cell->get_spatial_dim();
        throw muGrid::RuntimeError{err_msg.str()};
      }

      const Index_t nb_quad_pts_per_element{FourQuadPts};
      const Index_t nb_elements_per_pixel{1};
      const Index_t nb_element_nodal_pts{4};
      const Index_t nb_pixel_nodal_pts{OneNode};

      auto domain_lengths{cell->get_domain_lengths()};
      auto nb_grid_points{cell->get_nb_domain_grid_pts()};
      const Real del_x{domain_lengths[0] / nb_grid_points[0]};
      const Real del_y{domain_lengths[1] / nb_grid_points[1]};

      const Real det_jacobian{del_x / 2 * del_y / 2};

      Eigen::MatrixXd inv_jacobian(cell->get_spatial_dim(),
                                   cell->get_spatial_dim());

      inv_jacobian << (del_y / 2) / det_jacobian, 0, 0,
          (del_x / 2) / det_jacobian;

      const std::vector<Real> quadrature_weights{
          0.25 * det_jacobian, 0.25 * det_jacobian, 0.25 * det_jacobian,
          0.25 * det_jacobian};

      // ----------- nodal_pts ------------------
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(nb_element_nodal_pts)};
      Eigen::MatrixXi pixel_offsets{nb_element_nodal_pts,
                                    cell->get_spatial_dim()};
      //! first element
      pixel_offsets << 0, 0,  // clang-format off
        1, 0,
        0, 1,
        1, 1;  // clang-format on
      nodal_pts.emplace_back(node_ids, pixel_offsets);

      // ----------- shape functions gradients ------------------
      /* ----------------------------------------------------------------------
       */

      Eigen::MatrixXd quad_coords(cell->get_spatial_dim(),
                                  nb_element_nodal_pts);

      Real crd_0{-1. / (sqrt(3))};
      Real crd_1{+1. / (sqrt(3))};

      quad_coords.col(0) << crd_0, crd_0;
      quad_coords.col(1) << crd_1, crd_0;
      quad_coords.col(2) << crd_0, crd_1;
      quad_coords.col(3) << crd_1, crd_1;

      // per-quad-pt-vector of per-element-vector of matrices.
      std::vector<std::vector<Eigen::MatrixXd>> shape_fn_gradients(
          nb_quad_pts_per_element);
      Eigen::MatrixXd B_q{
          Eigen::MatrixXd::Zero(cell->get_spatial_dim(), nb_element_nodal_pts)};

      for (Index_t q{0}; q < nb_quad_pts_per_element; ++q) {
        auto x_q{quad_coords.col(q)};
        // clang-format off
        B_q <<
            (+x_q(1) - 1.)/4,
            (-x_q(1) + 1.)/4,
            (-x_q(1) - 1.)/4,
            (+x_q(1) + 1.)/4,
            (+x_q(0) - 1.)/4,
            (-x_q(0) - 1.)/4,
            (-x_q(0) + 1.)/4,
            (+x_q(0) + 1.)/4;
        // clang-format on

        shape_fn_gradients[q].push_back(inv_jacobian * B_q);
      }

      return std::make_shared<FEMStencil<muGrid::GradientOperatorDefault>>(
          nb_quad_pts_per_element, nb_elements_per_pixel, nb_element_nodal_pts,
          nb_pixel_nodal_pts, shape_fn_gradients, nodal_pts, quadrature_weights,
          cell);
    }
  }  // namespace FEMLibrary
}  // namespace muSpectre
