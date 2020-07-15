/**
 * @file   test_material_visco_elastic_damage_ss2.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   29 Apr 2020
 *
 * @brief  test for MaterialViscoElasticDamageSS2
 *
 * Copyright © 2020 Ali Falsafi
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

#include <materials/material_visco_elastic_damage_ss2.hh>

#include <libmugrid/field_collection_global.hh>
#include <libmugrid/mapped_field.hh>
#include <libmugrid/iterators.hh>
#include <libmugrid/tensor_algebra.hh>

#include <type_traits>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_visc_elastic_damage_ss2);

  template <Dim_t Dim>
  struct MaterialFixture {
    using MatVis1 = MaterialViscoElasticDamageSS1<Dim>;
    using MatVis2 = MaterialViscoElasticDamageSS2<Dim>;
    const Real young_inf{1.0e4};
    const Real young_v{1.0e4};
    const Real eta_v{1.0e1};
    const Real poisson{0.3};
    const Real kappa{1.0};
    const Real kappa_var{kappa * 1e-3};
    const Real alpha{0.014};
    const Real beta{0.34};
    const Real dt{1e-8};
    const size_t nb_steps{10000};
    MaterialFixture()
        : mat_vis("VisDam", mdim(), NbQuadPts(), young_inf, young_v, eta_v,
                  poisson, kappa + kappa_var, alpha, beta, dt),
          mat_vis_2("VisDam_2", mdim(), NbQuadPts(), young_inf, young_v, eta_v,
                    poisson, kappa, alpha, beta, dt) {}

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 1; }

    MatVis1 mat_vis;
    MatVis2 mat_vis_2;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("VisDam_2", Fix::mat_vis_2.get_name());
    auto & mat{Fix::mat_vis_2};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_add_pixel, Fix, mats, Fix) {
    auto & mat{Fix::mat_vis_2};
    muGrid::testGoodies::RandRange<size_t> rng;

    const Dim_t nb_pixel{7}, box_size{17};
    for (Dim_t i = 0; i < nb_pixel; ++i) {
      const size_t c{rng.randval(0, box_size)};
      BOOST_CHECK_NO_THROW(mat.add_pixel(c, Fix::kappa_var));
    }

    BOOST_CHECK_NO_THROW(mat.initialise());
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre
