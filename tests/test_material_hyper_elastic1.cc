/**
 * file   test_material_hyper_elastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   28 Nov 2017
 *
 * @brief  Tests for the large-strain, objective Hooke's law, implemented in
 *         the convenient strategy (i.e., using MaterialMuSpectre)
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <boost/mpl/list.hpp>

#include "materials/material_hyper_elastic1.hh"
#include "tests.hh"
#include "common/test_goodies.hh"
#include "common/field_collection.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elastic_1);

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialFixture
  {
    constexpr static Real lambda{2}, mu{1.5};
    MaterialFixture():mat("Name", lambda, mu){};
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};

    MaterialHyperElastic1<sdim, mdim> mat;
  };

  using mat_list = boost::mpl::list<MaterialFixture<twoD, twoD>,
                                    MaterialFixture<twoD, threeD>,
                                    MaterialFixture<threeD, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mat_list, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.mdim());
  }


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mat_list, Fix) {
    auto & mat{Fix::mat};
    constexpr Dim_t sdim{Fix::sdim};
    testGoodies::RandRange<size_t> rng;;
    const Dim_t nb_pixel{7}, box_size{17};
    using Ccoord = Ccoord_t<sdim>;
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      Ccoord c;
      for (Dim_t j = 0; j < sdim; ++j) {
        c[j] = rng.randval(0, box_size);
      }
      BOOST_CHECK_NO_THROW(mat.add_pixel(c));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialFixtureFilled: public MaterialFixture<DimS, DimM>
  {
    constexpr static Dim_t box_size{3};
    MaterialFixtureFilled():MaterialFixture<DimS, DimM>(){
      using Ccoord = Ccoord_t<DimS>;
      Ccoord sizes{CcoordOps::get_cube<DimS>(box_size)};
      CcoordOps::Pixels<DimS> pix(sizes);
      for (auto pixel: pix) {
        this->mat.add_pixel(pixel);
      }
      this->mat.initialise();
    };
  };

  using mat_fill = boost::mpl::list<MaterialFixtureFilled<twoD, twoD>,
                                    MaterialFixtureFilled<twoD, threeD>,
                                    MaterialFixtureFilled<threeD, threeD>>;
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluale_law, Fix, mat_fill, Fix) {
    const auto cube{CcoordOps::get_cube<Fix::sdim>(Fix::box_size)};
    FieldCollection<Fix::sdim, Fix::mdim> globalfields;
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
