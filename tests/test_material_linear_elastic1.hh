/**
 * @file   test_material_linear_elastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Jan 2020
 *
 * @brief  Fixtures for tests using MaterialLinearElastic1
 *
 * Copyright © 2020 Till Junge
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

#ifndef TESTS_TEST_MATERIAL_LINEAR_ELASTIC1_HH_
#define TESTS_TEST_MATERIAL_LINEAR_ELASTIC1_HH_

#include "materials/material_linear_elastic2.hh"
#include "materials/material_linear_elastic1.hh"

namespace muSpectre {

  template <class Mat_t>
  struct MaterialFixture : public Mat_t {
    using Mat = Mat_t;
    constexpr static Index_t NbQuadPts{1};
    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
    constexpr static Real poisson{lambda / (2 * (lambda + mu))};
    constexpr static Index_t mdim{Mat_t::MaterialDimension()};

    MaterialFixture() : Mat("Name", mdim, NbQuadPts, young, poisson) {}
  };
  template <class Mat_t>
  constexpr Real MaterialFixture<Mat_t>::poisson;
  template <class Mat_t>
  constexpr Real MaterialFixture<Mat_t>::young;
  template <class Mat_t>
  constexpr Index_t MaterialFixture<Mat_t>::NbQuadPts;
  template <class Mat_t>
  constexpr Index_t MaterialFixture<Mat_t>::mdim;

  template <class Mat_t>
  struct has_internals {
    constexpr static bool value{false};
  };

  template <Index_t DimM>
  struct has_internals<MaterialLinearElastic2<DimM>> {
    constexpr static bool value{true};
  };

  using mat_list =
      boost::mpl::list<MaterialFixture<MaterialLinearElastic1<twoD>>,
                       MaterialFixture<MaterialLinearElastic1<threeD>>,
                       MaterialFixture<MaterialLinearElastic2<twoD>>,
                       MaterialFixture<MaterialLinearElastic2<threeD>>>;


  /* ---------------------------------------------------------------------- */
  template <class Mat_t>
  struct MaterialFixtureFilled : public MaterialFixture<Mat_t> {
    using Mat = Mat_t;
    constexpr static Index_t box_size{3};
    MaterialFixtureFilled()
        : MaterialFixture<Mat_t>(), mat("Mat Name", Mat::MaterialDimension(),
                                        OneQuadPt, this->young, this->poisson) {
      using Ccoord = Ccoord_t<Mat_t::MaterialDimension()>;
      Ccoord cube{
          muGrid::CcoordOps::get_cube<Mat_t::MaterialDimension()>(box_size)};
      muGrid::CcoordOps::Pixels<Mat_t::MaterialDimension()> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        this->mat.add_pixel(id);
      }
      this->mat.initialise();
    }
    Mat_t mat;
  };

  using mat_fill =
      boost::mpl::list<MaterialFixtureFilled<MaterialLinearElastic1<twoD>>,
                       MaterialFixtureFilled<MaterialLinearElastic1<threeD>>>;


}  // namespace muSpectre

#endif  // TESTS_TEST_MATERIAL_LINEAR_ELASTIC1_HH_
