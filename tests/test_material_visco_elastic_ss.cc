/**
 * @file   test_material_visco_elastic_ss.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   30 Jan 2020
 *
 * @brief  Testing the MaterialViscoElasticSS
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

#include "materials/material_visco_elastic_ss.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

#include "libmugrid/tensor_algebra.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_visco_elastic_ss)

  template <Index_t Dim>
  struct MaterialFixture {
    using MatVis = MaterialViscoElasticSS<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young_inf{5.165484e4};
    const Real young_v{1.65156e4};
    const Real young_tot{young_v + young_inf};
    const Real eta_v{1.8435146e1};
    const Real poisson{0.3};
    const Real dt{1e-5};
    const size_t nb_steps{2000};
    MaterialFixture()
        : mat_vis("Vis", mdim(), NbQuadPts(), young_inf, young_v, eta_v,
                  poisson, dt),
          mat_lin_inf("Lin", mdim(), NbQuadPts(), young_inf, poisson) {}
    constexpr static Index_t mdim() { return Dim; }
    constexpr static Index_t sdim() { return mdim(); }
    constexpr static Index_t NbQuadPts() { return 1; }

    MatVis mat_vis;
    MatLin mat_lin_inf;
  };

  template <Index_t Dim>
  struct MaterialFixtureFilled : public MaterialFixture<Dim> {
    using Parent = MaterialFixture<Dim>;
    using MatVis = typename Parent::MatVis;
    using MatLin = typename Parent::MatLin;
    constexpr static Index_t box_size{1};
    MaterialFixtureFilled()
        : Parent(),
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
        this->mat_vis.add_pixel(id);
        this->mat_lin_inf.add_pixel(id);
        this->mat_lin_init.add_pixel(id);
      }
      this->mat_vis.initialise();
      this->mat_lin_inf.initialise();
      this->mat_lin_init.initialise();
    }
    const Real tol_init{60 * Parent::dt};
    const Real tol_fin{10 * Parent::dt};
    Real get_tol_init() { return tol_init; }
    Real get_tol_fin() { return tol_fin; }
    constexpr static Index_t mdim() { return MatVis::MaterialDimension(); }
    constexpr static Index_t sdim() { return mdim(); }

    MatVis mat_vis;  //<<! The viscoelastic material instance to be checked
    MatLin
        mat_lin_inf;  //<<! The linear elastic material (assymptotic as t -> ∞)
    MatLin mat_lin_init;  //<<! The linear elastic material (initial t = 0)
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;
  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>,
                                     MaterialFixtureFilled<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Vis", Fix::mat_vis.get_name());
    auto & mat{Fix::mat_vis};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_pure_volumetric, Fix,
                                   mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_vis{Fix::mat_vis};
    // auto & mat_lin_inf{Fix::mat_lin_inf};
    auto & mat_lin_init{Fix::mat_lin_init};

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt> h_{
        "history intgral", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_{"Pure elastic stress", coll, QuadPtTag};

    auto & h_prev{h_.get_map()};
    h_prev[0].current() = Strain_t::Identity();
    auto & s_null_prev{s_null_.get_map()};
    s_null_prev[0].current() = Strain_t::Identity();

    Strain_t E{Strain_t::Zero()};
    E(0, 0) = 0.1784687231e-3;
    E(1, 1) = E(0, 0);
    if (Fix::mdim() == 3) {
      E(2, 2) = E(1, 1);
    }

    h_.get_state_field().cycle();
    s_null_.get_state_field().cycle();
    auto && S_vis_init{mat_vis.evaluate_stress(E, h_[0], s_null_[0])};

    Strain_t S_vis_fin{Strain_t::Zero()};
    for (size_t i{0}; i < Fix::nb_steps; ++i) {
      h_.get_state_field().cycle();
      s_null_.get_state_field().cycle();
      S_vis_fin = mat_vis.evaluate_stress(E, h_[0], s_null_[0]);
    }
    auto && S_lin_init{mat_lin_init.evaluate_stress(E, 0)};
    // auto && S_lin_fin{mat_lin_inf.evaluate_stress(E, 0)};
    Real err_fin{rel_error(S_lin_init, S_vis_fin)};
    Real err_init{rel_error(S_lin_init, S_vis_init)};

    BOOST_CHECK_LT(err_fin, Fix::get_tol_fin());
    BOOST_CHECK_LT(err_init, Fix::get_tol_init());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress_pure_shear_test, Fix,
                                   mats_fill, Fix) {
    constexpr Index_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using muGrid::Matrices::ddot;

    auto & mat_vis{Fix::mat_vis};
    auto & mat_lin_inf{Fix::mat_lin_inf};
    auto & mat_lin_init{Fix::mat_lin_init};

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt> h_{
        "history intgral", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_{"Pure elastic stress", coll, QuadPtTag};

    auto & h_prev{h_.get_map()};
    h_prev[0].current() = Strain_t::Identity();
    auto & s_null_prev{s_null_.get_map()};
    s_null_prev[0].current() = Strain_t::Identity();

    Strain_t E{Strain_t::Zero()};
    E(0, 1) = 2.382563487e-2;
    E(1, 0) = E(0, 1);
    if (Fix::mdim() == 3) {
      E(1, 2) = 4.21437e-1;
      E(2, 1) = E(1, 2);
    }

    Real && tau_v{Fix::eta_v / Fix::young_v};

    // initial and assymptotic elastic responses
    auto && S_lin_init{mat_lin_init.evaluate_stress(E, 0)};
    auto && S_lin_fin{mat_lin_inf.evaluate_stress(E, 0)};

    Real energy{0.0};
    Real energy_ref{0.0};
    Real err_energy{0.0};

    h_.get_state_field().cycle();
    s_null_.get_state_field().cycle();
    auto && S_vis_init{mat_vis.evaluate_stress(E, h_[0], s_null_[0])};
    Real err_init{rel_error(S_lin_init, S_vis_init)};

    Strain_t S_vis{Strain_t::Zero()};
    for (size_t i{0}; i < Fix::nb_steps; ++i) {
      h_.get_state_field().cycle();
      s_null_.get_state_field().cycle();
      S_vis = mat_vis.evaluate_stress(E, h_[0], s_null_[0]);
      if (i >= 1) {
        energy = 0.5 * ddot<sdim>(S_vis, E);
        energy_ref =
            0.5 * ddot<sdim>(E, E) *
            (Fix::young_inf + Fix::young_v * std::exp(-(i * Fix::dt / tau_v))) /
            (1 + Fix::poisson);
        err_energy = rel_error(energy, energy_ref);
        BOOST_CHECK_LT(err_energy, std::sqrt(Fix::get_tol_init()));
      }
    }

    // The final response of the viscoelastic material
    auto && S_vis_fin{S_vis};
    Real err_fin{rel_error(S_lin_fin, S_vis_fin)};

    BOOST_CHECK_LT(err_init, Fix::get_tol_init());
    BOOST_CHECK_LT(err_fin, Fix::get_tol_fin());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_shear_test, Fix, mats_fill,
                                   Fix) {
    constexpr Index_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_vis{Fix::mat_vis};
    auto & mat_lin_inf{Fix::mat_lin_inf};
    auto & mat_lin_init{Fix::mat_lin_init};

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt> h_{
        "history intgral", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        s_null_{"Pure elastic stress", coll, QuadPtTag};

    auto & h_prev{h_.get_map()};
    h_prev[0].current() = Strain_t::Identity();
    auto & s_null_prev{s_null_.get_map()};
    s_null_prev[0].current() = Strain_t::Identity();

    Strain_t E{Strain_t::Zero()};
    E(0, 1) = 2.382563487e-2;
    E(1, 0) = E(0, 1);
    if (Fix::mdim() == 3) {
      E(1, 2) = 4.21437e-1;
      E(2, 1) = E(1, 2);
    }
    E(0, 0) = 0.1784687231e-3;
    E(1, 1) = E(0, 0);
    if (Fix::mdim() == 3) {
      E(2, 2) = E(1, 1);
    }

    h_.get_state_field().cycle();
    s_null_.get_state_field().cycle();
    auto && S_vis_init{mat_vis.evaluate_stress(E, h_[0], s_null_[0])};

    Strain_t S_vis_fin{Strain_t::Zero()};
    for (size_t i{0}; i < Fix::nb_steps; ++i) {
      h_.get_state_field().cycle();
      s_null_.get_state_field().cycle();
      S_vis_fin = mat_vis.evaluate_stress(E, h_[0], s_null_[0]);
    }
    Strain_t S_lin_init{mat_lin_init.evaluate_stress(E, 0)};
    Strain_t S_lin_fin{mat_lin_inf.evaluate_stress(E, 0)};
    Strain_t S_vis_fin_dev{S_vis_fin};
    Strain_t S_lin_fin_dev{S_lin_fin};
    Strain_t S_vis_init_dev{S_vis_init};
    Strain_t S_lin_init_dev{S_lin_init};

    Strain_t S_vis_fin_vol{S_vis_fin};
    Strain_t S_lin_fin_vol{S_lin_fin};
    Strain_t S_vis_init_vol{S_vis_init};
    Strain_t S_lin_init_vol{S_lin_init};

    for (Index_t i{0}; i < sdim; ++i) {
      S_vis_fin_dev(i, i) = 0.0;
      S_vis_init_dev(i, i) = 0.0;
      S_lin_fin_dev(i, i) = 0.0;
      S_lin_init_dev(i, i) = 0.0;
    }

    for (Index_t i{0}; i < sdim; ++i) {
      for (Index_t j{0}; j < sdim; ++j) {
        if (not(i == j)) {
          S_vis_fin_vol(i, j) = 0.0;
          S_vis_init_vol(i, j) = 0.0;
          S_lin_fin_vol(i, j) = 0.0;
          S_lin_init_vol(i, j) = 0.0;
        }
      }
    }

    Real err_fin_dev{rel_error(S_lin_fin_dev, S_vis_fin_dev)};
    Real err_fin_vol{rel_error(S_lin_init_vol, S_vis_fin_vol)};
    Real err_init{rel_error(S_lin_init, S_vis_init)};

    BOOST_CHECK_LT(err_fin_dev, Fix::get_tol_fin());
    BOOST_CHECK_LT(err_fin_vol, Fix::get_tol_init());
    BOOST_CHECK_LT(err_init, Fix::get_tol_init());
  }

  BOOST_AUTO_TEST_SUITE_END()

}  // namespace muSpectre
