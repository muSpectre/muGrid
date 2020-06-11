/**
 * @file   test_native_stress_storage.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Jan 2020
 *
 * @brief  test evaluation of stress with the `store_native` flag set to yes
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

#include "tests.hh"
#include "materials/material_linear_elastic1.hh"

#include <libmugrid/iterators.hh>
#include "libmugrid/test_goodies.hh"

#include "test_material_linear_elastic1.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(native_stress);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(store_native_stress, Fix, mat_fill, Fix) {
    constexpr auto cube{
        muGrid::CcoordOps::get_cube<Fix::MaterialDimension()>(Fix::box_size)};
    constexpr auto loc{
        muGrid::CcoordOps::get_cube<Fix::MaterialDimension()>(Index_t{0})};

    constexpr Index_t mdim{Fix::MaterialDimension()};
    auto & mat{Fix::mat};

    using FC_t = muGrid::GlobalFieldCollection;
    FC_t globalfields{Fix::MaterialDimension()};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts);
    globalfields.initialise(cube, loc);
    globalfields.register_real_field("Transformation Gradient", mdim * mdim,
                                     QuadPtTag);
    auto & P1 = globalfields.register_real_field(
        "Nominal Stress1", mdim * mdim,
        QuadPtTag);  // to be computed alone
    globalfields.register_real_field(
        "Nominal Stress2", mdim * mdim,
        QuadPtTag);  // to be computed with tangent
    globalfields.register_real_field(
        "Tangent Moduli", muGrid::ipow(mdim, 4),
        QuadPtTag);  // to be computed with tangent
    globalfields.register_real_field("Material stress (PK2) reference",
                                     mdim * mdim, QuadPtTag);

    static_assert(std::is_same<decltype(P1), muGrid::RealField &>::value,
                  "oh oh");
    using traits = MaterialMuSpectre_traits<typename Fix::Mat>;
    {  // block to contain not-constant gradient map
      typename traits::StressMap_t grad_map(
          globalfields.get_field("Transformation Gradient"));
      for (auto F_ : grad_map) {
        F_.setRandom();
      }
      grad_map[0] = grad_map[0].Identity();  // identifiable gradients for debug
      grad_map[1] = 1.2 * grad_map[1].Identity();  // ditto
    }

    // compute stresses using material
    mat.compute_stresses(globalfields.get_field("Transformation Gradient"),
                         globalfields.get_field("Nominal Stress1"),
                         Formulation::finite_strain, SplitCell::no,
                         StoreNativeStress::yes);

    typename traits::StrainMap_t Fmap(
        globalfields.get_field("Transformation Gradient"));
    typename traits::StressMap_t Smap_ref(
        globalfields.get_field("Material stress (PK2) reference"));

    auto && C{mat.get_C()};

    for (auto tup : akantu::zip(Fmap, Smap_ref)) {
      auto & F{std::get<0>(tup)};
      Eigen::Matrix<Real, mdim, mdim> E{0.5 *
                                        (F.transpose() * F - F.Identity())};
      Eigen::Matrix<Real, mdim, mdim> S{Matrices::tensmult(C, E)};
      auto & S_{std::get<1>(tup)};
      S_ = S;
    }
    auto && native_stress{mat.get_mapped_native_stress().get_map()};
    for (auto tup : akantu::zip(Smap_ref, native_stress)) {
      auto S_r{std::get<0>(tup)};
      auto S_1{std::get<1>(tup)};
      Real error{muGrid::testGoodies::rel_error(S_r, S_1)};
      BOOST_CHECK_LT(error, tol);
      if (not(error < tol)) {
        std::cout << "Stored native stored:" << std::endl << S_1 << std::endl;
        std::cout << "Stored native reference:" << std::endl
                  << S_r << std::endl;
      }
    }

    mat.compute_stresses_tangent(
        globalfields.get_field("Transformation Gradient"),
        globalfields.get_field("Nominal Stress2"),
        globalfields.get_field("Tangent Moduli"), Formulation::finite_strain,
        SplitCell::no, StoreNativeStress::yes);

    for (auto tup : akantu::zip(Smap_ref, native_stress)) {
      auto P_r{std::get<0>(tup)};
      auto P_1{std::get<1>(tup)};
      Real error{muGrid::testGoodies::rel_error(P_r, P_1)};
      BOOST_CHECK_LT(error, tol);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
