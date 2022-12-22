/**
 * @file   test_discrete_greens_operator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladeckÿ <m.ladecky@gmail.com>
 *
 * @date   16 Jul 2020
 *
 * @brief  tests for the inverse circulant operator
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include "projection/discrete_greens_operator.hh"
#include <libmufft/pocketfft_engine.hh>

#include <libmugrid/iterators.hh>
#include <libmugrid/field_collection_global.hh>
#include <libmugrid/ccoord_operations.hh>

#include <libmugrid/mapped_field.hh>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(greens_operator);

  template <Dim_t Rank>
  struct RankHolder {
    constexpr static Dim_t value() { return Rank; }
  };
  using ranks = boost::mpl::list<RankHolder<0>, RankHolder<1>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(recover_impulse_entry, Fix, ranks, Fix) {
    constexpr Index_t Dim{twoD};
    constexpr Index_t displacement_rank{Fix::value()};
    constexpr Index_t nb_grid_pts{5};
    constexpr Index_t nb_impulse_component_per_pixel{
        muGrid::ipow(Dim, displacement_rank)};
    const DynCcoord_t nb_subdomain_grid_pts{
        muGrid::CcoordOps::get_cube(Dim, nb_grid_pts)};
    muGrid::GlobalFieldCollection collection{Dim, nb_subdomain_grid_pts,
                                             nb_subdomain_grid_pts};

    using Field_t = muGrid::MappedMatrixField<
        Real, Mapping::Mut, nb_impulse_component_per_pixel,
        nb_impulse_component_per_pixel, IterUnit::Pixel>;

    Field_t impulse{"impulse", collection, PixelTag};
    Field_t impulse_fluctuation{"impulse_fluctuation", collection, PixelTag};
    Field_t response{"response", collection, PixelTag};
    Field_t recovered{"recovered impulse", collection, PixelTag};
    using ColField_t = muGrid::MappedMatrixField<
        Real, Mapping::Mut, nb_impulse_component_per_pixel, 1, IterUnit::Pixel>;
    ColField_t random_input{"random", collection, PixelTag};
    random_input.get_field().eigen_vec().setRandom();
    Eigen::MatrixXd mean{random_input.get_map().mean()};
    for (auto && val : random_input.get_map()) {
      val -= mean;
    }

    ColField_t random_response{"random_response", collection, PixelTag};

    impulse[0].setIdentity();

    auto & pixels{collection.get_pixels()};

    // pixel index offsets for whole stencil of [ij,i+j,ij+,i+j+] in 2D  ...
    muGrid::CcoordOps::DynamicPixels offsets{
        muGrid::CcoordOps::get_cube(Dim, 2)};

    // random stiffness matrix
    Eigen::MatrixXd pixel_stiffness{};
    if (displacement_rank == 0) {
      Real ki{1}, kj{2}, kd{3};
      pixel_stiffness.resize(4, 4);
      // clang-format off
      pixel_stiffness <<
        -(ki + kj + kd),  ki,  kj, kd,
                     ki, -ki,   0,  0,
                     kj,   0, -kj,  0,
                     kd,   0,   0, -kd;
      // clang-format on
    } else {
      pixel_stiffness.resize(8, 8);
      Real ki{1}, kj{2}, kd{3};
      // clang-format off
      pixel_stiffness <<
        -(ki + kj + kd),               0,  ki,   0,  kj,   0,  kd,   0,
                      0, -(ki + kj + kd),   0,  ki,   0,  kj,   0,  kd,
                     ki,               0, -ki,   0,   0,   0,   0,   0,
                      0,              ki,   0, -ki,   0,   0,   0,   0,
                     kj,               0,   0,   0, -kj,   0,   0,   0,
                      0,              kj,   0,   0,   0, -kj,   0,   0,
                     kd,               0,   0,   0,   0,   0, -kd,   0,
                      0,              kd,   0,   0,   0,   0,   0, -kd;
      // clang-format on
    }

    // loop over pixels
    // not using auto && because g++ 5
    for (auto id_base_ccoord : pixels.enumerate()) {
      auto && base_ccoord{
          std::get<1>(id_base_ccoord)};  // ijk spatial coords of pixel

      // loop over offsets
      for (auto && input_tup : akantu::enumerate(offsets)) {
        auto && input_index{std::get<0>(input_tup)};
        auto && input_offset{std::get<1>(input_tup)};
        auto && input_ccoord{(base_ccoord + input_offset) %
                             nb_subdomain_grid_pts};
        auto && impulse_val{impulse[pixels.get_index(input_ccoord)]};
        auto && random_input_val{random_input[pixels.get_index(input_ccoord)]};
        // loop over offsets
        for (auto && output_tup : akantu::enumerate(offsets)) {
          auto && output_index{std::get<0>(output_tup)};
          auto && output_offset{std::get<1>(output_tup)};
          auto && output_ccoord{(base_ccoord + output_offset) %
                                nb_subdomain_grid_pts};

          // get the right chunk of Stiffness
          auto && K_block{pixel_stiffness.block(
              output_index * nb_impulse_component_per_pixel,
              input_index * nb_impulse_component_per_pixel,
              nb_impulse_component_per_pixel, nb_impulse_component_per_pixel)};

          // get the nodal values relative to B-chunk
          auto && response_vals{response[pixels.get_index(output_ccoord)]};
          auto && random_response_vals{
              random_response[pixels.get_index(output_ccoord)]};

          response_vals += impulse_val * K_block;
          random_response_vals += K_block * random_input_val;
        }
      }
    }
    // create an fft engine
    muFFT::FFTEngine_ptr engine{
        std::make_shared<muFFT::PocketFFTEngine>(nb_subdomain_grid_pts)};

    /**
     * TODO(junge) After rebase, this should be done by the operators
     * constructor
     */
    engine->create_plan(nb_impulse_component_per_pixel);
    engine->create_plan(nb_impulse_component_per_pixel *
                        nb_impulse_component_per_pixel);
    DiscreteGreensOperator inverse_op{engine, response.get_field(),
                                      displacement_rank};

    /* ---------------------------------------------------------------------- */
    using ColField_t = muGrid::MappedMatrixField<
        Real, Mapping::Mut, nb_impulse_component_per_pixel, 1, IterUnit::Pixel>;
    ColField_t col_response{"col_response", collection, PixelTag};
    ColField_t col_recovered{"col_recovered", collection, PixelTag};
    for (Index_t col{0}; col < nb_impulse_component_per_pixel; ++col) {
      for (auto && col_full : akantu::zip(col_response, response)) {
        auto col_val{std::get<0>(col_full)};
        auto full_val{std::get<1>(col_full)};
        col_val = full_val.col(col);
      }

      inverse_op.apply(col_response.get_field(), col_recovered.get_field());

      for (auto && col_full : akantu::zip(col_recovered, recovered)) {
        auto col_val{std::get<0>(col_full)};
        auto full_val{std::get<1>(col_full)};

        full_val.col(col) = col_val;
      }
    }

    impulse_fluctuation.get_field().set_zero();
    impulse_fluctuation.get_map() = -impulse.get_map().mean();
    impulse_fluctuation.get_field() += impulse.get_field();

    auto && error{muGrid::testGoodies::rel_error(
        impulse_fluctuation.get_field().eigen_vec(),
        recovered.get_field().eigen_vec())};
    BOOST_CHECK_LE(error, tol);
    if (not(error <= tol)) {
      std::cout << "recovered values:" << std::endl
                << recovered.get_field().eigen_pixel() << std::endl;
      std::cout << "reference values:" << std::endl
                << impulse.get_field().eigen_pixel() << std::endl;
      std::cout << "reference values:" << std::endl
                << impulse_fluctuation.get_field().eigen_pixel() << std::endl;
    }

    inverse_op.apply(random_response.get_field(), col_recovered.get_field());
    error =
        muGrid::testGoodies::rel_error(random_input.get_field().eigen_pixel(),
                                       col_recovered.get_field().eigen_pixel());
    BOOST_CHECK_LE(error, tol);
    if (not(error <= tol)) {
      std::cout << "recovered values:" << std::endl
                << col_recovered.get_field().eigen_pixel() << std::endl;
      std::cout << "reference values:" << std::endl
                << random_input.get_field().eigen_pixel() << std::endl;
    }

    inverse_op.apply(random_response.get_field());
    error = muGrid::testGoodies::rel_error(
        random_input.get_field().eigen_pixel(),
        random_response.get_field().eigen_pixel());
    BOOST_CHECK_LE(error, tol);
    if (not(error <= tol)) {
      std::cout << "recovered values:" << std::endl
                << random_response.get_field().eigen_pixel() << std::endl;
      std::cout << "reference values:" << std::endl
                << random_input.get_field().eigen_pixel() << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
