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
      /*
       *          η
       * (-1,1)   |   (1,1)
       *     x---------x
       *     |    |    |
       *     |    |    |
       *   --|---------|----->  ξ
       *     |    |    |
       *     |    |    |
       *     x---------x
       * (-1,-1)  |   (1,-1)
       *
       * N₁ = (1 - ξ) (1 - η) / 4
       * N₂ = (1 + ξ) (1 - η) / 4
       * N₃ = (1 + ξ) (1 + η) / 4
       * N₄ = (1 - ξ) (1 + η) / 4
       * ∂N₁/∂ξ  = - (1 - η) / 4, ∂N₁/∂η = - (1 - ξ) / 4
       * ∂N₂/∂ξ  = + (1 - η) / 4, ∂N₂/∂η = - (1 + ξ) / 4
       * ∂N₃/∂ξ  = + (1 + η) / 4, ∂N₃/∂η = + (1 + ξ) / 4
       * ∂N₄/∂ξ  = - (1 + η) / 4, ∂N₄/∂η = + (1 - ξ) / 4
       */
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

      Eigen::MatrixXd inv_jacobian(twoD, twoD);

      inv_jacobian << (del_y / 2) / det_jacobian, 0, 0,
          (del_x / 2) / det_jacobian;

      const std::vector<Real> quadrature_weights{
          0.25 * det_jacobian, 0.25 * det_jacobian, 0.25 * det_jacobian,
          0.25 * det_jacobian};

      // ----------- nodal_pts ------------------
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(nb_element_nodal_pts)};
      Eigen::MatrixXi pixel_offsets{nb_element_nodal_pts, twoD};
      //! first element
      pixel_offsets << 0, 0,  // clang-format off
        1, 0,
        0, 1,
        1, 1;  // clang-format on
      nodal_pts.emplace_back(node_ids, pixel_offsets);

      // ----------- shape functions gradients ------------------
      /* ----------------------------------------------------------------------
       */

      Eigen::MatrixXd quad_coords(twoD, nb_element_nodal_pts);

      Real crd_0{-1. / (sqrt(3))};
      Real crd_1{+1. / (sqrt(3))};

      quad_coords.col(0) << crd_0, crd_0;
      quad_coords.col(1) << crd_1, crd_0;
      quad_coords.col(2) << crd_0, crd_1;
      quad_coords.col(3) << crd_1, crd_1;

      // per-quad-pt-vector of per-element-vector of matrices.
      std::vector<std::vector<Eigen::MatrixXd>> shape_fn_gradients(
          nb_quad_pts_per_element);
      Eigen::MatrixXd B_q{Eigen::MatrixXd::Zero(twoD, nb_element_nodal_pts)};

      for (Index_t q{0}; q < nb_quad_pts_per_element; ++q) {
        auto && x_q{quad_coords.col(q)};
        auto && xi{x_q(0)};
        auto && eta{x_q(1)};
        // clang-format off
        B_q <<
          (+ eta - 1.)/4,  (-eta + 1.)/4, (-eta - 1.)/4,  (+eta + 1.)/4,
          (+  xi - 1.)/4,  (- xi - 1.)/4, (- xi + 1.)/4,  (+ xi + 1.)/4;
        // clang-format on

        shape_fn_gradients[q].push_back(inv_jacobian * B_q);
      }

      return std::make_shared<FEMStencil<muGrid::GradientOperatorDefault>>(
          nb_quad_pts_per_element, nb_elements_per_pixel, nb_element_nodal_pts,
          nb_pixel_nodal_pts, shape_fn_gradients, nodal_pts, quadrature_weights,
          cell);
    }
    /* ---------------------------------------------------------------------- */
    std::shared_ptr<FEMStencilBase>
    trilinear_hexahedron(std::shared_ptr<CellData> cell) {
      /*
       *                    ζ
       *                    ^
       *         (-1,1,1)   |     (1,1,1)
       *                7---|------6
       *               /|   |     /|
       *              / |   |    / |
       *   (-1,-1,1) 4----------5  | (1,-1,1)
       *             |  |   |   |  |
       *             |  |   |   |  |
       *             |  |   +---|-------> ξ
       *             |  |  /    |  |
       *   (-1,1,-1) |  3-/-----|--2 (1,1,-1)
       *             | / /      | /
       *             |/ /       |/
       *             0-/--------1
       *   (-1,-1,-1) /        (1,-1,-1)
       *             /
       *            η
       *
       * N₁ = (1 - ξ) (1 - η) (1 - ζ) / 8
       * N₂ = (1 + ξ) (1 - η) (1 - ζ) / 8
       * N₃ = (1 + ξ) (1 + η) (1 - ζ) / 8
       * N₄ = (1 - ξ) (1 + η) (1 - ζ) / 8
       * N₅ = (1 - ξ) (1 - η) (1 + ζ) / 8
       * N₆ = (1 + ξ) (1 - η) (1 + ζ) / 8
       * N₇ = (1 + ξ) (1 + η) (1 + ζ) / 8
       * N₈ = (1 - ξ) (1 + η) (1 + ζ) / 8
       *
       * ∂N₁/∂ξ = - (1 - η) (1 - ζ) / 8
       * ∂N₁/∂η = - (1 - ξ) (1 - ζ) / 8
       * ∂N₁/∂ζ = - (1 - ξ) (1 - η) / 8
       *
       * ∂N₂/∂ξ = + (1 - η) (1 - ζ) / 8
       * ∂N₂/∂η = - (1 + ξ) (1 - ζ) / 8
       * ∂N₂/∂ζ = - (1 + ξ) (1 - η) / 8
       *
       * ∂N₃/∂ξ = + (1 + η) (1 - ζ) / 8
       * ∂N₃/∂η = + (1 + ξ) (1 - ζ) / 8
       * ∂N₃/∂ζ = - (1 + ξ) (1 + η) / 8
       *
       * ∂N₄/∂ξ = - (1 + η) (1 - ζ) / 8
       * ∂N₄/∂η = + (1 - ξ) (1 - ζ) / 8
       * ∂N₄/∂ζ = - (1 - ξ) (1 + η) / 8
       *
       * ∂N₅/∂ξ = - (1 - η) (1 + ζ) / 8
       * ∂N₅/∂η = - (1 - ξ) (1 + ζ) / 8
       * ∂N₅/∂ζ = + (1 - ξ) (1 - η) / 8
       *
       * ∂N₆/∂ξ = + (1 - η) (1 + ζ) / 8
       * ∂N₆/∂η = - (1 + ξ) (1 + ζ) / 8
       * ∂N₆/∂ζ = + (1 + ξ) (1 - η) / 8
       *
       * ∂N₇/∂ξ = + (1 + η) (1 + ζ) / 8
       * ∂N₇/∂η = + (1 + ξ) (1 + ζ) / 8
       * ∂N₇/∂ζ = + (1 + ξ) (1 + η) / 8
       *
       * ∂N₈/∂ξ = - (1 + η) (1 + ζ) / 8
       * ∂N₈/∂η = + (1 - ξ) (1 + ζ) / 8
       * ∂N₈/∂ζ = + (1 - ξ) (1 + η) / 8
       *
       * quad points:
       * ξ0  = -1/√3, η0 = -1/√3, ζ0 = -1/√3
       * ξ1  =  1/√3, η1 = -1/√3, ζ1 = -1/√3
       * ξ2  =  1/√3, η2 =  1/√3, ζ2 = -1/√3
       * ξ3, = -1/√3, η3 =  1/√3, ζ3 = -1/√3
       * ξ4  = -1/√3, η4 = -1/√3, ζ4 =  1/√3
       * ξ5  =  1/√3, η5 = -1/√3, ζ5 =  1/√3
       * ξ6  =  1/√3, η6 =  1/√3, ζ6 =  1/√3
       * ξ7  = -1/√3, η7 =  1/√3, ζ7 =  1/√3
       */
      // TODO(Ladecky) Not tested
      // create all the input
      constexpr auto ThreeD{threeD};
      if (cell->get_spatial_dim() != ThreeD) {
        std::stringstream err_msg{};
        err_msg << "Size mismatch: trilinear_hexahedron is a" << ThreeD
                << " dimensional stencil,"
                << "but cell spatial_dim =" << cell->get_spatial_dim();
        throw muGrid::RuntimeError{err_msg.str()};
      }

      constexpr Index_t EightQuadPtsPerElement{EightQuadPts};
      constexpr Index_t OneElementPerPixel{1};
      constexpr Index_t EightNodalPtsPerElement{8};
      constexpr Index_t OneNodalPtPerPixel{OneNode};

      auto && domain_lengths{cell->get_domain_lengths()};
      auto && nb_grid_points{cell->get_nb_domain_grid_pts()};
      const Real del_x{domain_lengths[0] / nb_grid_points[0]};
      const Real del_y{domain_lengths[1] / nb_grid_points[1]};
      const Real del_z{domain_lengths[2] / nb_grid_points[2]};

      const Real det_jacobian{del_x / 2 * del_y / 2 * del_z / 2};

      const Eigen::DiagonalMatrix<Real, ThreeD> inv_jacobian{
          2 / del_x, 2 / del_y, 2 / del_z};

      const std::vector<Real> quadrature_weights(EightQuadPtsPerElement,
                                                 det_jacobian / 8.);

      // ----------- nodal_pts ------------------
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> nodal_pts{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(EightNodalPtsPerElement)};
      Eigen::MatrixXi pixel_offsets{EightNodalPtsPerElement, ThreeD};
      //! first element
      pixel_offsets <<  // clang-format off
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 1, 1;  // clang-format on
      nodal_pts.emplace_back(node_ids, pixel_offsets);

      // ----------- shape functions gradients ------------------
      /* ----------------------------------------------------------------------
       */

      Eigen::MatrixXd quad_coords(ThreeD, EightNodalPtsPerElement);

      Real crd_0{-1. / (sqrt(3))};
      Real crd_1{+1. / (sqrt(3))};

      quad_coords.col(0) << crd_0, crd_0, crd_0;
      quad_coords.col(1) << crd_1, crd_0, crd_0;
      quad_coords.col(2) << crd_1, crd_1, crd_0;
      quad_coords.col(3) << crd_0, crd_1, crd_0;
      quad_coords.col(4) << crd_0, crd_0, crd_1;
      quad_coords.col(5) << crd_1, crd_0, crd_1;
      quad_coords.col(6) << crd_1, crd_1, crd_1;
      quad_coords.col(7) << crd_0, crd_1, crd_1;

      // per-quad-pt-vector of per-element-vector of matrices.
      std::vector<std::vector<Eigen::MatrixXd>> shape_fn_gradients(
          EightQuadPtsPerElement);
      Eigen::MatrixXd B_q{
          Eigen::MatrixXd::Zero(ThreeD, EightNodalPtsPerElement)};

      for (Index_t q{0}; q < EightQuadPtsPerElement; ++q) {
        auto && x_q{quad_coords.col(q)};
        auto && xi{x_q(0)};
        auto && eta{x_q(1)};
        auto && zeta{x_q(2)};

        // clang-format off
        B_q <<
          - (1 -  eta) * (1 - zeta) / 8, + (1 -  eta) * (1 - zeta) / 8, + (1 +  eta) * (1 - zeta) / 8, - (1 +  eta) * (1 - zeta) / 8, - (1 -  eta) * (1 + zeta) / 8, + (1 -  eta) * (1 + zeta) / 8, + (1 +  eta) * (1 + zeta) / 8, - (1 +  eta) * (1 + zeta) / 8, //NOLINT
          - (1 -   xi) * (1 - zeta) / 8, - (1 +   xi) * (1 - zeta) / 8, + (1 +   xi) * (1 - zeta) / 8, + (1 -   xi) * (1 - zeta) / 8, - (1 -   xi) * (1 + zeta) / 8, - (1 +   xi) * (1 + zeta) / 8, + (1 +   xi) * (1 + zeta) / 8, + (1 -   xi) * (1 + zeta) / 8, //NOLINT
          - (1 -   xi) * (1 -  eta) / 8, - (1 +   xi) * (1 -  eta) / 8, - (1 +   xi) * (1 +  eta) / 8, - (1 -   xi) * (1 +  eta) / 8, + (1 -   xi) * (1 -  eta) / 8, + (1 +   xi) * (1 -  eta) / 8, + (1 +   xi) * (1 +  eta) / 8, + (1 -   xi) * (1 +  eta) / 8; //NOLINT
        // clang-format on

        shape_fn_gradients[q].push_back(inv_jacobian * B_q);
      }

      return std::make_shared<FEMStencil<muGrid::GradientOperatorDefault>>(
          EightQuadPtsPerElement, OneElementPerPixel, EightNodalPtsPerElement,
          OneNodalPtPerPixel, shape_fn_gradients, nodal_pts, quadrature_weights,
          cell);
    }
  }  // namespace FEMLibrary
}  // namespace muSpectre
