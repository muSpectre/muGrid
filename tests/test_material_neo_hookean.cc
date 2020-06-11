/**
 * @file   test_material_neo_hookean.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   30 Jan 2020
 *
 * @brief  Testing the MaterialNeoHookean
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

#include "materials/material_neo_hookean_elastic.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

#include <libmugrid/ccoord_operations.hh>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_neo_hookean_elastic)

  template <Index_t Dim>
  struct MaterialFixture {
    using MatNeo = MaterialNeoHookeanElastic<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young{5.165484e4};
    const Real poisson{0.3};
    const Real amp{1e-5};
    MaterialFixture()
        : mat_neo("Neo", mdim(), NbQuadPts(), young, poisson),
          mat_lin("Lin", mdim(), NbQuadPts(), young, poisson) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }

    MatNeo mat_neo;
    MatLin mat_lin;
  };

  template <Index_t Dim>
  struct MaterialFixtureFilled : public MaterialFixture<Dim> {
    using Parent = MaterialFixture<Dim>;
    using MatNeo = typename Parent::MatNeo;
    using MatLin = typename Parent::MatLin;
    constexpr static Index_t box_size{1};
    MaterialFixtureFilled()
        : Parent(), mat_neo("Neo", Parent::mdim(), Parent::NbQuadPts(),
                            Parent::young, Parent::poisson),
          mat_lin("Lin", Parent::mdim(), Parent::NbQuadPts(), Parent::young,
                  Parent::poisson) {
      using Ccoord = Ccoord_t<Dim>;
      Ccoord cube{muGrid::CcoordOps::get_cube<Dim>(box_size)};

      muGrid::CcoordOps::Pixels<Dim> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        this->mat_neo.add_pixel(id);
        this->mat_lin.add_pixel(id);
      }
      this->mat_neo.initialise();
      this->mat_lin.initialise();
    }
    const Real tol{Parent::amp};
    constexpr static Index_t mdim() { return MatNeo::MaterialDimension(); }
    constexpr static Index_t sdim() { return mdim(); }

    MatNeo mat_neo;
    MatLin mat_lin;
  };  // namespace muSpectre

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;
  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>,
                                     MaterialFixtureFilled<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Neo", Fix::mat_neo.get_name());
    auto & mat{Fix::mat_neo};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_pure_volumetric, Fix,
                                   mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_neo{Fix::mat_neo};
    auto & mat_lin{Fix::mat_lin};
    Real amp{Fix::amp};
    Strain_t del_F{Strain_t::Zero()};
    if (Fix::mdim() == twoD) {
      del_F << 0.165165, 0.0, 0.0, 0.25415236;
    } else if (Fix::mdim() == threeD) {
      del_F << 0.1684, 0.0, 0.0, 0.0, 0.00, 0.10, 0.0, 0.0, 0.254132;
    }
    del_F = 1.0e-4 * del_F;
    Strain_t F{Strain_t::Identity() + amp * del_F};
    Strain_t E{Strain_t::Zero()};

    E = 0.5 * (F * F.transpose() - Strain_t::Identity());
    auto && tau_neo{mat_neo.evaluate_stress(F, 0)};
    auto && S_neo{
        MatTB::PK2_stress<StressMeasure::Kirchhoff, StrainMeasure::Gradient>(
            F, tau_neo)};
    auto && S_lin{mat_lin.evaluate_stress(E, 0)};

    Real err{rel_error(S_lin, S_neo)};

    BOOST_CHECK_LT(err, Fix::tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_tangent_pure_volumetric,
                                   Fix, mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_neo{Fix::mat_neo};
    auto & mat_lin{Fix::mat_lin};
    Real amp{Fix::amp};
    Strain_t del_F{Strain_t::Zero()};

    if (Fix::mdim() == twoD) {
      del_F << 0.165165, 0.0, 0.0, 0.25415236;
    } else if (Fix::mdim() == threeD) {
      del_F << 0.1684, 0.0, 0.0, 0.0, 0.00, 0.10, 0.0, 0.0, 0.254132;
    }

    del_F = 1.0e-4 * del_F;
    Strain_t F{Strain_t::Identity() + amp * del_F};
    Strain_t E{Strain_t::Zero()};

    E = 0.5 * (F * F.transpose() - Strain_t::Identity());
    auto && tau_c_neo{mat_neo.evaluate_stress_tangent(F, 0)};

    auto && tau_neo{std::get<0>(tau_c_neo)};
    auto && c_neo{std::get<1>(tau_c_neo)};

    auto && P_K_neo{
        MatTB::PK1_stress<StressMeasure::Kirchhoff, StrainMeasure::Gradient>(
            F, tau_neo, c_neo)};

    auto && P_neo{std::get<0>(P_K_neo)};
    auto && K_neo{std::get<1>(P_K_neo)};

    auto && S_C_lin{mat_lin.evaluate_stress_tangent(E, 0)};

    auto && P_K_lin{
        MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
            F, std::get<0>(S_C_lin), std::get<1>(S_C_lin))};

    auto && P_lin{std::get<0>(P_K_lin)};
    auto && K_lin{std::get<1>(P_K_lin)};

    Real err_S{rel_error(P_lin, P_neo)};
    Real err_C{rel_error(K_lin, K_neo)};

    BOOST_CHECK_LT(err_S, Fix::tol);
    BOOST_CHECK_LT(err_C, Fix::tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_objectivity_test, Fix, mats_fill, Fix) {
    using FC_t = muGrid::GlobalFieldCollection;

    auto & mat{Fix::mat_neo};

    const Index_t nb_pixel{1};
    auto cube{muGrid::CcoordOps::get_cube<Fix::sdim()>(nb_pixel)};

    FC_t globalfields{Fix::mdim()};
    globalfields.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    globalfields.initialise(cube, {});
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        F1_f{"Transformation Gradient 1", globalfields, QuadPtTag};
    muGrid::MappedT2Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        P1_f{"Nominal Stress 1", globalfields, QuadPtTag};
    muGrid::MappedT4Field<Real, Mapping::Mut, Fix::mdim(), IterUnit::SubPt>
        K1_f{"Tangent Moduli 1", globalfields, QuadPtTag};

    BOOST_CHECK_THROW(mat.compute_stresses_tangent(
                          globalfields.get_field("Transformation Gradient 1"),
                          globalfields.get_field("Nominal Stress 1"),
                          globalfields.get_field("Tangent Moduli 1"),
                          Formulation::small_strain),
                      std::runtime_error);

    BOOST_CHECK_THROW(mat.compute_stresses(
                          globalfields.get_field("Transformation Gradient 1"),
                          globalfields.get_field("Nominal Stress 1"),
                          Formulation::small_strain),
                      std::runtime_error);
  }

  BOOST_AUTO_TEST_SUITE_END()

}  // namespace muSpectre
