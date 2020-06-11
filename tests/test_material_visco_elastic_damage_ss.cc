/**
 * @file   test_material_visco_elastic_damage_ss.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   30 Jan 2020
 *
 * @brief  Testing the MaterialViscoElasticDamageSS
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

// #include
// "materials/material_linear_visco_elastic_deviatoric_damage_small_strain.hh"
#include "materials/material_visco_elastic_damage_ss.hh"
#include "materials/material_visco_elastic_ss.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(
      material_linear_visco_elastic_deviatoric_damage_small_strain)

  template <Index_t Dim>
  struct MaterialFixture {
    using MatVisDam = MaterialViscoElasticDamageSS<Dim>;
    using MatVis = MaterialViscoElasticSS<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young_inf{1.0e4};
    const Real young_v{1.0e4};
    const Real eta_v{1.0e1};
    const Real poisson{0.3};
    const Real kappa{1.0};
    const Real alpha{0.014};
    const Real beta{0.34};
    const Real dt{1e-8};
    const size_t nb_steps{10000};
    MaterialFixture()
        : mat_vis_dam("VisDam", mdim(), NbQuadPts(), young_inf, young_v, eta_v,
                      poisson, kappa, alpha, beta, dt),
          mat_vis("Vis", mdim(), NbQuadPts(), young_inf, young_v, eta_v,
                  poisson, dt),
          mat_lin_inf("Lin", mdim(), NbQuadPts(), young_inf, poisson) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }

    MatVisDam mat_vis_dam;
    MatVis mat_vis;
    MatLin mat_lin_inf;
  };

  template <Index_t Dim>
  struct MaterialFixtureFilled : public MaterialFixture<Dim> {
    using Parent = MaterialFixture<Dim>;
    using MatVisDam = typename Parent::MatVisDam;
    using MatVis = typename Parent::MatVis;
    using MatLin = typename Parent::MatLin;
    constexpr static Index_t box_size{1};
    MaterialFixtureFilled()
        : Parent(),
          mat_vis_dam("VisDam", mdim(), Parent::NbQuadPts(), Parent::young_inf,
                      Parent::young_v, Parent::eta_v, Parent::poisson,
                      Parent::kappa, Parent::alpha, Parent::beta, Parent::dt),
          mat_vis("Vis", mdim(), Parent::NbQuadPts(), Parent::young_inf,
                  Parent::young_v, Parent::eta_v, Parent::poisson, Parent::dt),
          mat_lin_inf("Lin", mdim(), Parent::NbQuadPts(), Parent::young_inf,
                      Parent::poisson),
          mat_lin_init("Lin", mdim(), Parent::NbQuadPts(),
                       Parent::young_inf + Parent::young_v, Parent::poisson) {
      using Ccoord = Ccoord_t<Dim>;
      Ccoord cube{muGrid::CcoordOps::get_cube<Dim>(box_size)};
      muGrid::CcoordOps::Pixels<Dim> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        this->mat_vis_dam.add_pixel(id);
        this->mat_lin_inf.add_pixel(id);
        this->mat_lin_init.add_pixel(id);
      }
      this->mat_vis_dam.initialise();
      this->mat_lin_inf.initialise();
      this->mat_lin_init.initialise();
    }
    constexpr static Real tol{1.e-6};
    const Real tol_init{Parent::young_inf * Parent::dt};
    constexpr static Real get_tol() { return tol; }
    Real get_tol_init() { return tol_init; }
    constexpr static Index_t mdim() { return MatVisDam::MaterialDimension(); }
    constexpr static Index_t sdim() { return mdim(); }

    MatVisDam mat_vis_dam;
    MatVis mat_vis;
    MatLin mat_lin_inf;
    MatLin mat_lin_init;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;
  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("VisDam", Fix::mat_vis_dam.get_name());
    auto & mat{Fix::mat_vis_dam};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_pure_volumetric, Fix,
                                   mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_vis_dam{Fix::mat_vis_dam};
    // auto & mat_lin_inf{Fix::mat_lin_inf};
    auto & mat_lin_init{Fix::mat_lin_init};

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt> h_{
        "history intgral ", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_{"Pure elastic stress", coll, QuadPtTag};

    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_{"Kappa", coll, QuadPtTag};

    auto & h_prev{h_.get_map()};
    h_prev[0].current() = Strain_t::Identity();
    auto & s_null_prev{s_null_.get_map()};
    s_null_prev[0].current() = Strain_t::Identity();
    auto & kappa_prev{kappa_.get_map()};
    kappa_prev[0].current() = Fix::kappa;

    Strain_t E_fin{Strain_t::Zero()};
    E_fin(0, 0) = 0.1e-1;
    E_fin(1, 1) = E_fin(0, 0);
    if (Fix::mdim() == 3) {
      E_fin(2, 2) = E_fin(1, 1);
    }

    auto && dE{E_fin * (1.0 / Fix::nb_steps)};

    h_.get_state_field().cycle();
    s_null_.get_state_field().cycle();
    kappa_.get_state_field().cycle();

    Strain_t S_vis_inf{Strain_t::Zero()};

    for (size_t i{0}; i < Fix::nb_steps; ++i) {
      auto && E{i * dE};
      h_.get_state_field().cycle();
      s_null_.get_state_field().cycle();
      kappa_.get_state_field().cycle();
      S_vis_inf = mat_vis_dam.evaluate_stress(E, h_[0], s_null_[0], kappa_[0]);
    }
    auto && S_lin_inf{
        mat_lin_init.evaluate_stress((Fix::nb_steps - 1) * dE, 0)};
    auto && dam_current{
        mat_vis_dam.compute_damage_measure(kappa_[0].current())};

    auto && S_ref_fin{dam_current * S_lin_inf};
    auto && err_fin{rel_error(S_ref_fin, S_vis_inf)};
    BOOST_CHECK_LT(err_fin, Fix::get_tol_init());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_pure_shear, Fix,
                                   mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using muGrid::Matrices::ddot;

    auto & mat_vis_dam{Fix::mat_vis_dam};
    auto & mat_vis{Fix::mat_vis};

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt> h_{
        "history intgral ", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_{"Pure elastic stress ", coll, QuadPtTag};

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        h_none{"history intgral non", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_none{"Pure elastic stress non", coll, QuadPtTag};

    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_{"Kappa", coll, QuadPtTag};

    auto & h_prev{h_.get_map()};
    h_prev[0].current() = Strain_t::Identity();
    auto & s_null_prev{s_null_.get_map()};
    s_null_prev[0].current() = Strain_t::Identity();

    auto & h_prev_non{h_none.get_map()};
    h_prev_non[0].current() = Strain_t::Identity();
    auto & s_null_prev_non{s_null_none.get_map()};
    s_null_prev_non[0].current() = Strain_t::Identity();

    auto & kappa_prev{kappa_.get_map()};
    kappa_prev[0].current() = Fix::kappa;

    // Real && tau_v{Fix::eta_v / Fix::young_v};

    Strain_t E_fin{Strain_t::Zero()};
    E_fin(0, 1) = 1.0e-3;
    E_fin(1, 0) = E_fin(0, 1);
    if (Fix::mdim() == 3) {
      E_fin(1, 2) = E_fin(0, 1);
      E_fin(2, 1) = E_fin(1, 2);
    }
    auto && dE{E_fin * (1.0 / Fix::nb_steps)};

    h_.get_state_field().cycle();
    s_null_.get_state_field().cycle();

    h_none.get_state_field().cycle();
    s_null_none.get_state_field().cycle();

    kappa_.get_state_field().cycle();

    Strain_t S_vis{Strain_t::Zero()};
    Strain_t S_vis_dam{Strain_t::Zero()};

    Strain_t S_vis_old{Strain_t::Zero()};
    Strain_t S_vis_dam_old{Strain_t::Zero()};

    Real energy_dam{0.0};
    Real energy_dam_ref{0.0};

    Real err_energy{0.0};

    for (size_t i{0}; i < Fix::nb_steps; ++i) {
      auto && E{i * dE};
      h_.get_state_field().cycle();
      s_null_.get_state_field().cycle();
      h_none.get_state_field().cycle();
      s_null_none.get_state_field().cycle();
      kappa_.get_state_field().cycle();

      S_vis_dam = mat_vis_dam.evaluate_stress(E, h_[0], s_null_[0], kappa_[0]);

      S_vis = mat_vis.evaluate_stress(E, h_none[0], s_null_none[0]);

      auto && dam_current{
          mat_vis_dam.compute_damage_measure(kappa_[0].current())};

      auto && dam_old{mat_vis_dam.compute_damage_measure(kappa_[0].old())};
      if (i > 1) {
        energy_dam += ddot<sdim>(0.5 * (S_vis_dam + S_vis_dam_old), dE);
        energy_dam_ref +=
            ddot<sdim>(0.5 * (dam_current * S_vis + dam_old * S_vis_old), dE);
        S_vis_old = S_vis;
        S_vis_dam_old = S_vis_dam;
        err_energy = rel_error(energy_dam, energy_dam_ref);
        BOOST_CHECK_LT(err_energy, Fix::get_tol_init());
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END()

}  // namespace muSpectre
