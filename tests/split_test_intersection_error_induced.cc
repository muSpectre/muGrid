/**
 * @file   split_test_intersection_error_induced.cc
 *
 * @author Ali Faslfi <ali.faslfi@epfl.ch>
 *
 * @date   21 Jun 2018
 *
 * @brief  Testing how much nb_grid_pts can result in the performance of corkpp
 * for geometrical analysis and boolean operation
 *
 * Copyright © 2017 Till Junge
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

#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "cell/cell_factory.hh"
#include "cell/cell_split.hh"
#include "common/intersection_octree.hh"

#include <libmufft/fftw_engine.hh>

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>

#include <boost/mpl/list.hpp>
#include <math.h>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(split_cell_octree_error_test);

  BOOST_AUTO_TEST_CASE(area_calculation_diff_resolution) {
    constexpr Dim_t dim{twoD};
    using Mat_t = MaterialLinearElastic1<dim>;

    const Real contrast{10};
    const Real Young_soft{1.0030648180242636},
        Poisson_soft{0.29930675909878679};
    const Real Young_hard{contrast * Young_soft},
        Poisson_hard{0.29930675909878679};
    const Real corkpp_precision{1e-5};

    constexpr Real length{4.2};
    constexpr int high_res{55}, low_res{11};
    DynRcoord_t lengths_split{length, length};
    DynCcoord_t resolutions_split_high_res{high_res, high_res};
    DynCcoord_t resolutions_split_low_res{low_res, low_res};

    auto fft_ptr_split_high_res{std::make_unique<muFFT::FFTWEngine>(
        resolutions_split_high_res)};
    auto fft_ptr_split_low_res{std::make_unique<muFFT::FFTWEngine>(
        resolutions_split_low_res)};

    auto proj_ptr_split_high_res{
        std::make_unique<ProjectionFiniteStrainFast<dim>>(
            std::move(fft_ptr_split_high_res), lengths_split)};
    auto proj_ptr_split_low_res{
        std::make_unique<ProjectionFiniteStrainFast<dim>>(
            std::move(fft_ptr_split_low_res), lengths_split)};

    CellSplit sys_split_high_res(std::move(proj_ptr_split_high_res));
    CellSplit sys_split_low_res(std::move(proj_ptr_split_low_res));

    auto & Material_hard_split_high_res{Mat_t::make(
        sys_split_high_res, "hard high res", Young_hard, Poisson_hard)};
    auto & Material_soft_split_high_res{Mat_t::make(
        sys_split_high_res, "soft high res", Young_soft, Poisson_soft)};

    auto & Material_hard_split_low_res{Mat_t::make(
        sys_split_low_res, "hard low res", Young_hard, Poisson_hard)};
    auto & Material_soft_split_low_res{Mat_t::make(
        sys_split_low_res, "soft low res", Young_soft, Poisson_soft)};

    DynRcoord_t center{2.1, 2.1};
    DynRcoord_t width{-0.65, 0.65};
    DynRcoord_t height{-0.65, 0.65};

    std::vector<DynRcoord_t> precipitate_vertices{};
    precipitate_vertices.push_back(
        {center[0] + width[0], center[1] + height[0]});
    precipitate_vertices.push_back(
        {center[0] + width[1], center[1] + height[0]});
    precipitate_vertices.push_back(
        {center[0] + width[0], center[1] + height[1]});
    precipitate_vertices.push_back(
        {center[0] + width[1], center[1] + height[1]});

    // analyzing the intersection of the preicipitate with the pixels
    RootNode<SplitCell::simple> precipitate_low_res(sys_split_low_res,
                                                    precipitate_vertices);
    // Extracting the intersected pixels and their correspondent intersection
    // ratios:
    auto && precipitate_low_res_intersects{
        precipitate_low_res.get_intersected_pixels()};
    auto && precipitate_low_res_intersection_ratios{
        precipitate_low_res.get_intersection_ratios()};
    auto && precipitate_low_res_intersects_id{
        precipitate_low_res.get_intersected_pixels_id()};

    // assign material to the pixels which have intersection with the
    // precipitate
    for (auto tup : akantu::zip(precipitate_low_res_intersects_id,
                                precipitate_low_res_intersects,
                                precipitate_low_res_intersection_ratios)) {
      auto && pix_id{std::get<0>(tup)};
      auto && ratio{std::get<2>(tup)};
      Material_hard_split_low_res.add_pixel_split(pix_id, ratio);
    }
    Material_hard_split_low_res.initialise();
    // assign material to the rest of the pixels:
    std::vector<Real> assigned_ratio_low_res =
        sys_split_low_res.get_assigned_ratios();
    for (auto && pix_id : sys_split_low_res.get_pixel_indices()) {
      if (assigned_ratio_low_res[pix_id] < 1.0) {
        Material_soft_split_low_res.add_pixel_split(
            pix_id, 1.0 - assigned_ratio_low_res[pix_id]);
      }
    }
    Material_soft_split_low_res.initialise();
    sys_split_low_res.initialise();

    // Calculating the area of the precipitate from the intersected pixels:
    Real area_preticipitate_low_res{0.0};
    Real area_preticipitate_high_res{0.0};
    Real pixel_volume_low_res{muGrid::CcoordOps::compute_pixel_volume(
        sys_split_low_res.get_projection().get_nb_domain_grid_pts(),
        sys_split_low_res.get_projection().get_domain_lengths())};
    for (auto && precipitate_area_low_res :
         precipitate_low_res_intersection_ratios) {
      area_preticipitate_low_res +=
          precipitate_area_low_res * pixel_volume_low_res;
    }

    // analyzing the intersection of the precipitate with the pixels
    RootNode<SplitCell::simple> precipitate_high_res(sys_split_high_res,
                                                     precipitate_vertices);
    // Extracting the intersected pixels and their correspondent intersection
    // ratios:
    auto precipitate_high_res_intersects_id{
        precipitate_high_res.get_intersected_pixels_id()};
    auto precipitate_high_res_intersection_ratios{
        precipitate_high_res.get_intersection_ratios()};

    // assign material to the pixels which have intersection with the
    // precipitate
    for (auto tup : akantu::zip(precipitate_high_res_intersects_id,
                                precipitate_high_res_intersection_ratios)) {
      auto && pix{std::get<0>(tup)};
      auto && ratio{std::get<1>(tup)};
      Material_hard_split_high_res.add_pixel_split(pix, ratio);
    }
    Material_hard_split_high_res.initialise();
    // assign material to the rest of the pixels:
    std::vector<Real> assigned_ratio_high_res{
        sys_split_high_res.get_assigned_ratios()};

    for (auto && pix_id : sys_split_high_res.get_pixel_indices()) {
      if (assigned_ratio_high_res[pix_id] < 1.0) {
        Material_soft_split_high_res.add_pixel_split(
            pix_id, 1.0 - assigned_ratio_high_res[pix_id]);
      }
    }
    Material_soft_split_high_res.initialise();
    sys_split_high_res.initialise();

    auto pixel_volume_high_res{muGrid::CcoordOps::compute_pixel_volume(
        sys_split_high_res.get_projection().get_nb_domain_grid_pts(),
        sys_split_high_res.get_projection().get_domain_lengths())};
    for (auto && precipitate_area_high_res :
         precipitate_high_res_intersection_ratios) {
      area_preticipitate_high_res +=
          precipitate_area_high_res * pixel_volume_high_res;
    }

    BOOST_CHECK_LE(
        (abs(area_preticipitate_high_res - area_preticipitate_low_res) /
         area_preticipitate_low_res),
        corkpp_precision);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
