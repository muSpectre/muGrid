/**
 * @file   split_test_corkpp.cc
 *
 * @author Ali Faslfi <ali.faslfi@epfl.ch>
 *
 * @date   21 Jun 2018
 *
 * @brief  Tests for split cells and octree material assignment
 *
 * Copyright © 2017 Till Ali Faslafi
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

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>

#include <libmufft/fftw_engine.hh>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(split_corkpp_test);

  BOOST_AUTO_TEST_CASE(corkpp_test_3D) {
    constexpr Dim_t dim{threeD};
    DynCcoord_t nb_pixels{1, 1, 1};
    DynRcoord_t lengths{1.0, 1.0, 1.0};
    constexpr Real corkpp_precision{1e-5};
    auto fft_ptr{std::make_unique<muFFT::FFTWEngine>(nb_pixels)};
    auto proj_ptr{std::make_unique<ProjectionFiniteStrainFast<dim>>(
        std::move(fft_ptr), lengths)};
    Cell sys_1(std::move(proj_ptr));
    constexpr Real x{0.646416436};
    constexpr Real y{0.475466546};
    std::vector<DynRcoord_t> precipitate_vertices{};

    // using the class Vectors written in common/intersection_octree.hh
    // instead of std:vector<Eigen::matrix<Real, dim, 1>>
    Vectors_t p_v(dim);

    precipitate_vertices.push_back({0.00, 0.00, 0.00});
    precipitate_vertices.push_back({x, 0.00, 0.00});
    precipitate_vertices.push_back({0.00, x, 0.00});
    precipitate_vertices.push_back({0.00, 0.00, x});
    precipitate_vertices.push_back({y, y, y});

    for (auto && vertex : precipitate_vertices) {
      p_v.push_back(vertex);
    }

    RootNode<SplitCell::laminate> precipitate(sys_1, precipitate_vertices);

    auto && precipitate_intersection_ratios{
        precipitate.get_intersection_ratios()};
    auto && precipitate_intersects_normals{
        precipitate.get_intersection_normals()};

    // Volume of each of the three tetrahedra can be calculated via:
    // Volume = 1/6*abs(dot(cross(P2-P0,P3-P0),P4-P0))
    Real ratio_ref{3.0 * (1.0 / 6.0) *
                   ((p_v.at<dim>(1) - p_v.at<dim>(0))
                        .cross(p_v.at<dim>(2) - p_v.at<dim>(0)))
                       .dot(p_v.at<dim>(4) - p_v.at<dim>(0))};

    // the average normal is the unit vector aligned with {1.0 ,1.0 ,1.0}
    Eigen::Matrix<Real, dim, 1> normal_ref{sqrt(3) / 3, sqrt(3) / 3,
                                           sqrt(3) / 3};

    auto && ratio{precipitate_intersection_ratios[0]};
    auto && normal{precipitate_intersects_normals[0]};

    BOOST_CHECK_LE((normal - normal_ref).norm(), corkpp_precision);
    BOOST_CHECK_LE(abs(ratio - ratio_ref), corkpp_precision);
  }

  BOOST_AUTO_TEST_CASE(corkpp_test_2D) {
    constexpr Dim_t dim{twoD};
    DynCcoord_t nb_pixels{1, 1};
    DynRcoord_t lengths{1.0, 1.0};
    constexpr Real corkpp_precision{1e-4};

    auto fft_ptr_split{std::make_unique<muFFT::FFTWEngine>(nb_pixels)};
    auto proj_ptr_split{std::make_unique<ProjectionFiniteStrainFast<dim>>(
        std::move(fft_ptr_split), lengths)};
    CellSplit sys_split{std::move(proj_ptr_split)};

    constexpr Real x{0.646416436};
    constexpr Real y{0.475466546};
    std::vector<DynRcoord_t> precipitate_vertices{};
    precipitate_vertices.push_back({0.00, 0.00});
    precipitate_vertices.push_back({x, 0.00});
    precipitate_vertices.push_back({0.00, x});
    precipitate_vertices.push_back({y, y});

    RootNode<SplitCell::laminate> precipitate(sys_split, precipitate_vertices);

    auto && precipitate_intersection_ratios{
        precipitate.get_intersection_ratios()};
    auto && precipitate_intersects_normals{
        precipitate.get_intersection_normals()};

    // the area of each triangle of the two is (x * y / 2.0)
    Real ratio_ref{2.0 * (x * y / 2.0)};
    // the average normal is the unit vector aligned with {1.0 ,1.0}
    Eigen::Matrix<Real, dim, 1> normal_ref{sqrt(2) / 2, sqrt(2) / 2};
    auto && ratio{precipitate_intersection_ratios[0]};
    auto && normal{precipitate_intersects_normals[0]};
    BOOST_CHECK_LE((normal - normal_ref).norm(), corkpp_precision);
    BOOST_CHECK_LE(abs(ratio - ratio_ref), corkpp_precision);
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
