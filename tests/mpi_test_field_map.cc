/**
 * @file   mpi_test_field_map.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   19 Nov 2021
 *
 * @brief  parallel test fro FieldMap, especially with empty processors
 *
 * Copyright © 2021 Till Junge
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

#include "mpi_field_test_fixtures.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(mpi_field_maps);

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(fieldmap_size_sum_mean_tests_empty_proc,
                          MpiFieldMapFixtureEmptyProcs) {
    constexpr Dim_t nb_comps{MpiFieldMapFixtureEmptyProcs::NbComponent};
    // init random fields
    auto && eigen_pix{this->pixel_field.eigen_vec()};
    eigen_pix.setRandom();
    auto && eigen_quad_pt{this->quad_pt_field.eigen_vec()};
    eigen_quad_pt.setRandom();

    // --- test FieldMaps size() ---
    const auto nb_pix{this->fc.get_nb_pixels()};
    const auto & nb_quad_pts{nb_pix * this->fc.get_nb_sub_pts("quad")};

    BOOST_CHECK_EQUAL(this->pixel_map.size(), nb_pix);
    BOOST_CHECK_EQUAL(this->quad_pt_map.size(), nb_quad_pts);
    BOOST_CHECK_EQUAL(this->pixel_quad_pt_map.size(), nb_pix);

    // --- test FieldMaps sum() ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_sum;
    pix_sum.setZero();
    for (Index_t pix{0}; pix < nb_pix; pix++) {
      pix_sum += this->pixel_map[pix];
    }
    BOOST_CHECK_EQUAL(this->pixel_map.sum(), pix_sum);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_sum;
    quad_pt_sum.setZero();
    for (Index_t qpt{0}; qpt < nb_quad_pts; qpt++) {
      quad_pt_sum += this->quad_pt_map[qpt];
    }
    BOOST_CHECK_EQUAL(this->quad_pt_map.sum(), quad_pt_sum);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_sum;
    pix_quad_sum.setZero();
    for (Index_t pix{0}; pix < nb_pix; pix++) {
      pix_quad_sum += this->pixel_quad_pt_map[pix];
    }
    BOOST_CHECK_EQUAL(this->pixel_quad_pt_map.sum(), pix_quad_sum);

    // --- test FieldMap mean ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_mean{nb_pix != 0 ? pix_sum / nb_pix
                                                          : pix_sum};
    BOOST_CHECK_EQUAL(this->pixel_map.mean(), pix_mean);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean{
        nb_quad_pts != 0 ? quad_pt_sum / nb_quad_pts : quad_pt_sum};
    BOOST_CHECK_EQUAL(this->quad_pt_map.mean(), quad_pt_mean);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean{
        nb_pix != 0 ? pix_quad_sum / nb_pix : pix_quad_sum};
    BOOST_CHECK_EQUAL(this->pixel_quad_pt_map.mean(), pix_quad_mean);

    // --- get total sum and mean by comm.sum on all processors ---
    // --- comm.sum of FieldMap sum ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_sum_all{comm.sum(pix_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_map.sum()), pix_sum_all);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_sum_all{comm.sum(quad_pt_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->quad_pt_map.sum()), quad_pt_sum_all);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_sum_all{
        comm.sum(pix_quad_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_quad_pt_map.sum()),
                      pix_quad_sum_all);

    // --- comm.sum of FieldMap mean ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_mean_all{comm.sum(pix_sum) /
                                                  comm.sum(nb_pix)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_map.mean()), pix_mean_all);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean_all{comm.sum(quad_pt_sum) /
                                                      comm.sum(nb_quad_pts)};
    BOOST_CHECK_EQUAL(comm.sum(this->quad_pt_map.mean()), quad_pt_mean_all);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean_all{
        comm.sum(pix_quad_sum) / comm.sum(nb_pix)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_quad_pt_map.mean()),
                      pix_quad_mean_all);
  };

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(fieldmap_size_sum_mean_tests_full_proc,
                          MpiFieldMapFixtureFullProcs) {
    constexpr Dim_t nb_comps{MpiFieldMapFixtureFullProcs::NbComponent};
    // init random fields
    auto && eigen_pix{this->pixel_field.eigen_vec()};
    eigen_pix.setRandom();
    auto && eigen_quad_pt{this->quad_pt_field.eigen_vec()};
    eigen_quad_pt.setRandom();

    // --- test FieldMaps size() ---
    const auto nb_pix{this->fc.get_nb_pixels()};
    const auto & nb_quad_pts{nb_pix * this->fc.get_nb_sub_pts("quad")};

    BOOST_CHECK_EQUAL(this->pixel_map.size(), nb_pix);
    BOOST_CHECK_EQUAL(this->quad_pt_map.size(), nb_quad_pts);
    BOOST_CHECK_EQUAL(this->pixel_quad_pt_map.size(), nb_pix);

    // --- test FieldMaps sum() ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_sum;
    pix_sum.setZero();
    for (Index_t pix{0}; pix < nb_pix; pix++) {
      pix_sum += this->pixel_map[pix];
    }
    BOOST_CHECK_EQUAL(this->pixel_map.sum(), pix_sum);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_sum;
    quad_pt_sum.setZero();
    for (Index_t qpt{0}; qpt < nb_quad_pts; qpt++) {
      quad_pt_sum += this->quad_pt_map[qpt];
    }
    BOOST_CHECK_EQUAL(this->quad_pt_map.sum(), quad_pt_sum);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_sum;
    pix_quad_sum.setZero();
    for (Index_t pix{0}; pix < nb_pix; pix++) {
      pix_quad_sum += this->pixel_quad_pt_map[pix];
    }
    BOOST_CHECK_EQUAL(this->pixel_quad_pt_map.sum(), pix_quad_sum);

    // --- test FieldMap mean ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_mean{nb_pix != 0 ? pix_sum / nb_pix
                                                          : pix_sum};
    Eigen::Matrix<Real, nb_comps, 1> pix_mean_comp{this->pixel_map.mean()};
    for (Index_t i = 0; i < pix_mean.rows(); i++) {
      for (Index_t j = 0; j < pix_mean.cols(); j++) {
        BOOST_CHECK_CLOSE(pix_mean_comp(i, j), pix_mean(i, j), 1E-12);
      }
    }

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean{
        nb_quad_pts != 0 ? quad_pt_sum / nb_quad_pts : quad_pt_sum};
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean_comp{
        this->quad_pt_map.mean()};
    for (Index_t i = 0; i < quad_pt_mean.rows(); i++) {
      for (Index_t j = 0; j < quad_pt_mean.cols(); j++) {
        BOOST_CHECK_CLOSE(quad_pt_mean_comp(i, j), quad_pt_mean(i, j), 1E-12);
      }
    }

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean{
        nb_pix != 0 ? pix_quad_sum / nb_pix : pix_quad_sum};
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean_comp{
        this->pixel_quad_pt_map.mean()};
    for (Index_t i = 0; i < pix_quad_mean.rows(); i++) {
      for (Index_t j = 0; j < pix_quad_mean.cols(); j++) {
        BOOST_CHECK_CLOSE(pix_quad_mean_comp(i, j), pix_quad_mean(i, j), 1E-12);
      }
    }

    // --- get total sum and mean by comm.sum on all processors ---
    // --- comm.sum of FieldMap sum ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_sum_all{comm.sum(pix_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_map.sum()), pix_sum_all);

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_sum_all{comm.sum(quad_pt_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->quad_pt_map.sum()), quad_pt_sum_all);

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_sum_all{
        comm.sum(pix_quad_sum)};
    BOOST_CHECK_EQUAL(comm.sum(this->pixel_quad_pt_map.sum()),
                      pix_quad_sum_all);

    // --- comm.sum of FieldMap mean ---
    // iterate over pixels
    Eigen::Matrix<Real, nb_comps, 1> pix_mean_all{comm.sum(pix_sum) /
                                                  comm.sum(nb_pix)};
    Eigen::Matrix<Real, nb_comps, 1> pix_mean_all_comp{
        comm.sum(this->pixel_map.mean()) / comm.size()};
    for (Index_t i = 0; i < pix_mean_all.rows(); i++) {
      for (Index_t j = 0; j < pix_mean_all.cols(); j++) {
        BOOST_CHECK_CLOSE(pix_mean_all_comp(i, j), pix_mean_all(i, j), 1E-12);
      }
    }

    // iterate over quad points
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean_all{comm.sum(quad_pt_sum) /
                                                      comm.sum(nb_quad_pts)};
    Eigen::Matrix<Real, nb_comps, 1> quad_pt_mean_all_comp{
        comm.sum(this->quad_pt_map.mean()) / comm.size()};
    for (Index_t i = 0; i < quad_pt_mean_all.rows(); i++) {
      for (Index_t j = 0; j < quad_pt_mean_all.cols(); j++) {
        BOOST_CHECK_CLOSE(quad_pt_mean_all_comp(i, j), quad_pt_mean_all(i, j),
                          1E-12);
      }
    }

    // iterate over pixels of a field with quad points
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean_all{
        comm.sum(pix_quad_sum) / comm.sum(nb_pix)};
    Eigen::Matrix<Real, nb_comps, nb_comps> pix_quad_mean_all_comp{
        comm.sum(this->pixel_quad_pt_map.mean()) / comm.size()};
    for (Index_t i = 0; i < pix_quad_mean_all.rows(); i++) {
      for (Index_t j = 0; j < pix_quad_mean_all.cols(); j++) {
        BOOST_CHECK_CLOSE(pix_quad_mean_all_comp(i, j), pix_quad_mean_all(i, j),
                          1E-12);
      }
    }
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
