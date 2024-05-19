/**
 * @file   test_discrete_gradient_operator.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   23 Jul 2020
 *
 * @brief  common headers for shape function gradients
 *
 * Copyright © 2020 Till Junge
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

#include "libmugrid/grid_common.hh"
#include "libmugrid/gradient_operator_default.hh"

#ifndef TESTS_LIBMUGRID_TEST_DISCRETE_GRADIENT_OPERATOR_HH_
#define TESTS_LIBMUGRID_TEST_DISCRETE_GRADIENT_OPERATOR_HH_

namespace muGrid {

  template <Index_t Dim_>
  struct FixtureBase {
    constexpr static Real del_x{2.};
    constexpr static Real del_y{3.};
    constexpr static Real del_z{5.};
    constexpr static Index_t Dim{Dim_};
    template <Index_t Rank>
    using Grad_t = Eigen::Matrix<Real, ipow(Dim, Rank - 1), Dim>;
    constexpr static Real Del_(Index_t dir) {
      switch (dir) {
      case 0: {
        return del_x;
        break;
      }
      case 1: {
        return del_y;
        break;
      }
      case 2: {
        return del_z;
        break;
      }
      default:
        throw std::runtime_error("unknown dimension");
        break;
      }
    }
  };

  template <Index_t Dim_>
  constexpr Real FixtureBase<Dim_>::del_x;
  template <Index_t Dim_>
  constexpr Real FixtureBase<Dim_>::del_y;
  template <Index_t Dim_>
  constexpr Real FixtureBase<Dim_>::del_z;
  template <Index_t Dim_>
  constexpr Index_t FixtureBase<Dim_>::Dim;

  struct FixtureTriangularStraight : public FixtureBase<twoD> {
    constexpr static Index_t NbElements{2};
    constexpr static Index_t NbQuadPerELement{OneQuadPt};
    constexpr static Index_t NbElemNodalPts{3};
    constexpr static Index_t NbNode{1};

    FixtureTriangularStraight()
        : d_operator{Dim,
                     NbQuadPerELement,
                     NbElements,
                     NbElemNodalPts,
                     NbNode,
                     this->get_quad_pt_B(),
                     this->get_nodal_pts()} {}

    /* ---------------------------------------------------------------------- */
    static std::vector<std::vector<Eigen::MatrixXd>> get_quad_pt_B() {
      // per-quad-pt-vector of per-element-vector of matrices.
      std::vector<std::vector<Eigen::MatrixXd>> ret_val(NbQuadPerELement);

      Eigen::MatrixXd B1(Dim, NbElemNodalPts);
      // clang-format off
      B1 << -1./del_x, 1./del_x,        0,
            -1./del_y,        0, 1./del_y;
      // clang-format on

      Eigen::MatrixXd B2(Dim, NbElemNodalPts);
      // clang-format off
      B2 << 1./del_x, -1./del_x,         0,
            1./del_y,         0, -1./del_y;
      // clang-format on

      constexpr Index_t q_id{0};
      ret_val[q_id].push_back(B1);
      ret_val[q_id].push_back(B2);
      return ret_val;
    }

    /* ---------------------------------------------------------------------- */
    static Eigen::MatrixXd get_full_pixel_B() {
      const Index_t full_nb_nodal_pts{NbNode * ipow(2, Dim)};
      Eigen::MatrixXd B(Dim * NbQuadPerELement * NbElements, full_nb_nodal_pts);
      // clang-format off
      B << -1./del_x,  1./del_x,         0,        0,
           -1./del_y,         0,  1./del_y,        0,
                   0,         0, -1./del_x, 1./del_x,
                   0, -1./del_y,         0, 1./del_y;
      // clang-format on
      return B;
    }

    /* ---------------------------------------------------------------------- */
    static std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>>
    get_nodal_pts() {
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> ret_val{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(NbElemNodalPts)};
      Eigen::MatrixXi pixel_offsets{NbElemNodalPts, Dim};

      //! first element
      pixel_offsets << 0, 0,  // clang-format off
                       1, 0,
                       0, 1;  // clang-format on
      ret_val.push_back(std::make_tuple(node_ids, pixel_offsets));

      //! second element
      pixel_offsets << 1, 1,  // clang-format off
                       0, 1,
                       1, 0;  // clang-format on
      ret_val.push_back(std::make_tuple(node_ids, pixel_offsets));

      return ret_val;
    }
    GradientOperatorDefault d_operator;
  };

  struct FixtureBilinearQuadrilat : public FixtureBase<twoD> {
    constexpr static Index_t NbElements{1};
    constexpr static Index_t NbQuadPerELement{FourQuadPts};
    constexpr static Index_t NbElemNodalPts{4};
    constexpr static Index_t NbNode{1};  // Nb nodal subpoints

    FixtureBilinearQuadrilat()
        : d_operator{Dim,
                     NbQuadPerELement,
                     NbElements,
                     NbElemNodalPts,
                     NbNode,
                     this->get_quad_pt_B(),
                     this->get_nodal_pts()} {}

    /* ---------------------------------------------------------------------- */
    static std::vector<std::vector<Eigen::MatrixXd>> get_quad_pt_B() {
      // per-quad-pt-vector of per-element-vector of matrices.
      // ret_val[q,e] = B[q,e]
      // clang-format off
      //      ii        i+j       ij+       i+j+
      // [phi_{0,x} phi_{1,x} phi_{2,x} phi_{3,x};
      //  phi_{0,y} phi_{1,y} phi_{2,y} phi_{3,y}]
      // clang-format on
      std::vector<std::vector<Eigen::MatrixXd>> ret_val(NbQuadPerELement);

      auto quad_coords{get_quad_coords()};
      Real del_xy{del_x * del_y};

      Eigen::MatrixXd B_q{Eigen::MatrixXd::Zero(Dim, NbElemNodalPts)};

      for (Index_t q{0}; q < NbQuadPerELement; ++q) {
        auto x_q{quad_coords.col(q)};
        // clang-format off
        B_q <<
          (x_q(1) - 1.) * del_y / (2 * del_xy),
            (-x_q(1) + 1.) * del_y / (2 * del_xy),
            (-x_q(1) - 1.) * del_y / (2 * del_xy),
            (x_q(1) + 1.) * del_y / (2 * del_xy),
          (x_q(0) - 1.) * del_x / (2 * del_xy),
            (-x_q(0) - 1.) * del_x / (2 * del_xy),
            (-x_q(0) + 1.) * del_x / (2 * del_xy),
            (x_q(0) + 1.) * del_x / (2 * del_xy);
        // clang-format on

        ret_val[q].push_back(B_q);
      }

      return ret_val;
    }
    /* ---------------------------------------------------------------------- */
    static Eigen::MatrixXd get_quad_coords() {
      Eigen::MatrixXd quad_coords(Dim, NbQuadPerELement);

      Real crd_0{-1. / (sqrt(3))};
      Real crd_1{+1. / (sqrt(3))};

      quad_coords.col(0) << crd_0, crd_0;
      quad_coords.col(1) << crd_1, crd_0;
      quad_coords.col(2) << crd_0, crd_1;
      quad_coords.col(3) << crd_1, crd_1;

      return quad_coords;
    }

    /* ---------------------------------------------------------------------- */
    static Eigen::MatrixXd get_full_pixel_B() {
      const Index_t full_nb_nodal_pts{NbNode * ipow(2, Dim)};
      Eigen::MatrixXd B(Dim * NbQuadPerELement * NbElements, full_nb_nodal_pts);

      auto quad_coords{get_quad_coords()};

      Real del_xy{del_x * del_y};
      // this is recycled
      // clang-format off
      B <<
        (quad_coords(1, 0) - 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 0) + 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 0) - 1.) * del_y / (2 * del_xy),
          (quad_coords(1, 0) + 1.) * del_y / (2 * del_xy),
        (quad_coords(0, 0) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 0) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 0) + 1.) * del_x / (2 * del_xy),
          (quad_coords(0, 0) + 1.) * del_x / (2 * del_xy),

        (quad_coords(1, 1) - 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 1) + 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 1) - 1.) * del_y / (2 * del_xy),
          (quad_coords(1, 1) + 1.) * del_y / (2 * del_xy),
        (quad_coords(0, 1) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 1) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 1) + 1.) * del_x / (2 * del_xy),
          (quad_coords(0, 1) + 1.) * del_x / (2 * del_xy),

        (quad_coords(1, 2) - 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 2) + 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 2) - 1.) * del_y / (2 * del_xy),
          (quad_coords(1, 2) + 1.) * del_y / (2 * del_xy),
        (quad_coords(0, 2) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 2) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 2) + 1.) * del_x / (2 * del_xy),
          (quad_coords(0, 2) + 1.) * del_x / (2 * del_xy),

        (quad_coords(1, 3) - 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 3) + 1.) * del_y / (2 * del_xy),
          (-quad_coords(1, 3) - 1.) * del_y / (2 * del_xy),
          (quad_coords(1, 3) + 1.) * del_y / (2 * del_xy),
        (quad_coords(0, 3) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 3) - 1.) * del_x / (2 * del_xy),
          (-quad_coords(0, 3) + 1.) * del_x / (2 * del_xy),
          (quad_coords(0, 3) + 1.) * del_x / (2 * del_xy);
      // clang-format on
      return B;
    }

    /* ---------------------------------------------------------------------- */
    static std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>>
    get_nodal_pts() {
      std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> ret_val{};
      Eigen::VectorXi node_ids{Eigen::VectorXi::Zero(NbElemNodalPts)};
      Eigen::MatrixXi pixel_offsets{NbElemNodalPts, Dim};

      //! first element
      pixel_offsets << 0, 0,  // clang-format off
          1, 0,
          0, 1,
          1, 1;  // clang-format on
      ret_val.push_back(std::make_tuple(node_ids, pixel_offsets));

      return ret_val;
    }
    GradientOperatorDefault d_operator;
  };

  constexpr Index_t FixtureTriangularStraight::NbElements;
  constexpr Index_t FixtureTriangularStraight::NbElemNodalPts;
  constexpr Index_t FixtureTriangularStraight::NbQuadPerELement;
  constexpr Index_t FixtureTriangularStraight::NbNode;

  constexpr Index_t FixtureBilinearQuadrilat::NbElements;
  constexpr Index_t FixtureBilinearQuadrilat::NbElemNodalPts;
  constexpr Index_t FixtureBilinearQuadrilat::NbQuadPerELement;
  constexpr Index_t FixtureBilinearQuadrilat::NbNode;

}  // namespace muGrid

#endif  // TESTS_LIBMUGRID_TEST_DISCRETE_GRADIENT_OPERATOR_HH_
