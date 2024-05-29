/**
 * @file   test_discrete_gradient_operator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   25 Jun 2020
 *
 * @brief  Tests for the discrete gradient operator
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

#include "tests.hh"
#include "test_goodies.hh"
#include "test_discrete_gradient_operator.hh"

#include "libmugrid/gradient_operator_default.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/mapped_field.hh"

#include <boost/mpl/list.hpp>

namespace muGrid {

  using DOperatorFixtures =
      boost::mpl::list<FixtureTriangularStraight, FixtureBilinearQuadrilat>;

  BOOST_AUTO_TEST_SUITE(discrete_gradient_operator);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, DOperatorFixtures,
                                   Fix) {
    Real error{testGoodies::rel_error(Fix::get_full_pixel_B(),
                                      Fix::d_operator.get_pixel_gradient())};

    BOOST_CHECK_EQUAL(error, 0);
    if (error != 0) {
      std::cout << "Full B received:" << std::endl
                << Fix::d_operator.get_pixel_gradient() << std::endl;
      std::cout << "Full B expected: " << std::endl
                << Fix::get_full_pixel_B() << std::endl;
    }
  }

  template <Index_t Rank, class Fixture>
  GlobalFieldCollection
  create_uniform_gradient(const Eigen::Ref<Eigen::MatrixXd> & grad_input) {
    typename Fixture::template Grad_t<Rank> grad{grad_input};
    DynCcoord_t nb_grid_points(Fixture::Dim);
    DynRcoord_t grid_length(Fixture::Dim);

    for (Index_t i{0}; i < Fixture::Dim; ++i) {
      nb_grid_points[i] = 3 + i;
      grid_length[i] = nb_grid_points[i] * Fixture::Del_(i);
    }

    const std::string nodal_pt_tag{"nodal_pt"};
    const std::string quad_pt_tag{"quad_pt"};
    // rank is rank of the gradient
    using Disp_t = Eigen::Matrix<Real, ipow(Fixture::Dim, Rank - 1), 1>;
    using Pos_t = Eigen::Matrix<Real, Fixture::Dim, 1>;
    GlobalFieldCollection collection{nb_grid_points, nb_grid_points};

    MappedMatrixField<Real, Mapping::Mut, Disp_t::RowsAtCompileTime,
                      Disp_t::ColsAtCompileTime, IterUnit::Pixel>
        nodal_field{"nodal_field", collection, PixelTag};

    // fill nodal field
    // not using auto && because g++ 5
    for (auto id_pixel : collection.get_pixels().enumerate()) {
      auto && id{std::get<0>(id_pixel)};
      auto && ccoord{std::get<1>(id_pixel)};

      Pos_t position(eigen(ccoord).cast<Real>().array() *
                     eigen(grid_length / nb_grid_points).array());
      nodal_field[id] = grad * position;
    }

    return collection;
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(gradient_test, Fix, DOperatorFixtures, Fix) {
    if (Fix::NbNode != OneNode) {
      throw std::runtime_error(
          "Can't handle multiple nodal subpoints in a grid. Please implement a "
          "test for multiple subpoints.");
    }

    for (Index_t grad_rank{1}; grad_rank <= 2; ++grad_rank) {
      Eigen::MatrixXd grad{};
      if (grad_rank == 1) {
        grad = Fix::template Grad_t<1>::Random();
      } else {
        grad = Fix::template Grad_t<2>::Random();
      }

      GlobalFieldCollection collection{
          grad_rank == 1 ? create_uniform_gradient<1, Fix>(grad)
                         : create_uniform_gradient<2, Fix>(grad)};
      std::string quad_pt_tag{"quad_pt"};
      collection.set_nb_sub_pts(quad_pt_tag,
                                Fix::NbElements * Fix::NbQuadPerELement);
      auto & quad_pt_field{collection.register_real_field(
          "quad_pt_field", ipow(Fix::Dim, grad_rank), quad_pt_tag)};

      Fix::d_operator.apply_gradient(dynamic_cast<TypedFieldBase<Real> &>(
                                         collection.get_field("nodal_field")),
                                     quad_pt_field);
      auto && quad_pt_map{quad_pt_field.get_pixel_map()};
      auto && pixels{collection.get_pixels()};
      auto && nb_domain_grid_pts{pixels.get_nb_subdomain_grid_pts()};

      DynCcoord_t valid_grid_pts{nb_domain_grid_pts};
      for (auto && val : valid_grid_pts) {
        --val;
      }

      // not using auto && because g++ 5
      for (auto id_pixel : pixels.enumerate()) {
        auto && id{std::get<0>(id_pixel)};
        auto && ccoord{std::get<1>(id_pixel)};

        if (ccoord == ccoord % valid_grid_pts) {
          auto && all_quad_grads{quad_pt_map[id]};
          // loop over all quad_pts
          for (Index_t q{0}; q < Fix::NbQuadPerELement * Fix::NbElements; ++q) {
            Eigen::Map<Eigen::MatrixXd> quad_pt_grad{
                all_quad_grads.col(q).data(), grad.rows(), grad.cols()};
            auto && error{testGoodies::rel_error(quad_pt_grad, grad)};
            if (error >= tol) {
              std::cout << "at quad pt " << q << " of ccoord " << ccoord
                        << ", reference :" << std::endl
                        << grad << std::endl
                        << "computed  :" << std::endl
                        << quad_pt_grad << std::endl;
            }
            BOOST_CHECK_LE(error, tol);
          }
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(transpose_operator, Fix, DOperatorFixtures,
                                   Fix) {
    const DynCcoord_t nb_subdomain_grid_pts{
        Fix::Dim == twoD ? DynCcoord_t{2, 2} : DynCcoord_t{2, 2, 2}};
    GlobalFieldCollection collection{nb_subdomain_grid_pts,
                                     nb_subdomain_grid_pts};
    const std::string nodal_pt_tag{"nodal_pt"};
    const std::string quad_pt_tag{"quad_pt"};
    collection.set_nb_sub_pts(nodal_pt_tag, Fix::NbNode);
    collection.set_nb_sub_pts(quad_pt_tag,
                              Fix::NbQuadPerELement * Fix::NbElements);

    for (Index_t u_rank{0}; u_rank < 2; ++u_rank) {
      std::stringstream rank_{};
      rank_ << u_rank;
      auto rank_str{rank_.str()};
      auto & u{collection.register_real_field(
          "u" + rank_str, ipow(Fix::Dim, u_rank), nodal_pt_tag)};
      auto & v{collection.register_real_field(
          "v" + rank_str, ipow(Fix::Dim, u_rank), nodal_pt_tag)};
      auto & BTBu{collection.register_real_field(
          "BᵀB·u" + rank_str, ipow(Fix::Dim, u_rank), nodal_pt_tag)};
      auto & Bu{collection.register_real_field(
          "B·u" + rank_str, ipow(Fix::Dim, u_rank + 1), quad_pt_tag)};
      auto & Bv{collection.register_real_field(
          "B·v" + rank_str, ipow(Fix::Dim, u_rank + 1), quad_pt_tag)};

      // init with random values to avoid special cases
      u.eigen_vec().setRandom();
      v.eigen_vec().setRandom();

      // check than (B·u, B·v) == (BᵀB·u, v) for any u, v
      this->d_operator.apply_gradient(u, Bu);
      this->d_operator.apply_gradient(v, Bv);
      this->d_operator.apply_transpose(Bu, BTBu);

      auto bubv{Bu.eigen_vec().dot(Bv.eigen_vec())};
      auto btbuv{BTBu.eigen_vec().dot(v.eigen_vec())};
      auto error{testGoodies::rel_error(bubv, btbuv)};

      BOOST_CHECK_LE(error, tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
