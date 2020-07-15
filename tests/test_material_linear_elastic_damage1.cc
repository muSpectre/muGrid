/**
 * @file   test_material_linear_elastic_damage1.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2020
 *
 * @brief  the test for the MaterialLinearElasticDamage1
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

#include "materials/material_linear_elastic_damage1.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/iterable_proxy.hh"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_linear_elastic_damage1)
  template <Dim_t Dim>
  struct MaterialFixture {
    using MatDam = MaterialLinearElasticDamage1<Dim>;
    using MatLin = MaterialLinearElastic1<Dim>;
    const Real young{1.0e4};
    const Real poisson{0.3};
    const Real kappa{1.0};
    const Real alpha{0.014};
    const Real beta{0.34};

    //! Constructor
    MaterialFixture()
        : mat_dam("Dam", mdim(), NbQuadPts(), young, poisson, kappa, alpha,
                  beta),
          mat_lin("Lin", mdim(), NbQuadPts(), young, poisson) {}

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 1; }

    MatDam mat_dam;
    MatLin mat_lin;
  };

  template <Index_t Dim>
  struct MaterialFixtureFilled : public MaterialFixture<Dim> {
    using Parent = MaterialFixture<Dim>;
    using MatDam = typename Parent::MatDam;
    using MatLin = typename Parent::MatLin;

    constexpr static Index_t box_size{1};
    MaterialFixtureFilled()
        : Parent(),
          mat_dam("Dam", mdim(), Parent::NbQuadPts(), Parent::young,
                  Parent::poisson, Parent::kappa, Parent::alpha, Parent::beta),
          mat_lin("Lin", mdim(), Parent::NbQuadPts(), Parent::young,
                  Parent::poisson) {
      using Ccoord = Ccoord_t<Dim>;
      Ccoord cube{muGrid::CcoordOps::get_cube<Dim>(box_size)};
      muGrid::CcoordOps::Pixels<Dim> pixels(cube);
      for (auto && id_pixel : akantu::enumerate(pixels)) {
        auto && id{std::get<0>(id_pixel)};
        this->mat_dam.add_pixel(id);
        this->mat_lin.add_pixel(id);
      }
      this->mat_dam.initialise();
      this->mat_lin.initialise();
    }

    constexpr static Real tol{1.e-6};
    constexpr static Real get_tol() { return tol; }
    constexpr static Dim_t mdim() { return MatDam::MaterialDimension(); }
    constexpr static Dim_t sdim() { return mdim(); }

    MatDam mat_dam;
    MatLin mat_lin;
  };

  using mats = boost::mpl::list<MaterialFixture<twoD>, MaterialFixture<threeD>>;
  using mats_fill = boost::mpl::list<MaterialFixtureFilled<twoD>,
                                     MaterialFixtureFilled<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Dam", Fix::mat_dam.get_name());
    auto & mat{Fix::mat_dam};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evalaute_stress, Fix, mats_fill, Fix) {
    constexpr Dim_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    auto & mat_dam{Fix::mat_dam};
    auto & mat_lin{Fix::mat_lin};

    // create state fields
    muGrid::LocalFieldCollection::SubPtMap_t map{};
    map[QuadPtTag] = Fix::NbQuadPts();
    muGrid::LocalFieldCollection coll{sdim, map};
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> kappa_{
        "Kappa", coll, QuadPtTag};

    auto & kappa{kappa_.get_map()};
    kappa[0].current() = Fix::kappa;

    Strain_t F{1e-1 * Strain_t::Random() + Strain_t::Identity()};
    Strain_t E{0.5 * ((F * F.transpose()) - Strain_t::Identity())};

    kappa_.get_state_field().cycle();

    Strain_t S_dam{Strain_t::Zero()};
    Strain_t S_lin{Strain_t::Zero()};

    S_dam = mat_dam.evaluate_stress(E, kappa_[0]);
    S_lin = mat_lin.evaluate_stress(E, 0);
    auto && kap_ref{std::sqrt(::muGrid::Matrices::ddot<Fix::mdim()>(S_lin, E))};

    auto && kap{kappa_[0].current()};

    auto && err{rel_error(kap, kap_ref)};

    BOOST_CHECK_LT(err, Fix::get_tol());

    auto && dam{kap > Fix::kappa
                    ? Fix::beta + (1 - Fix::beta) *
                                      ((1.0 - std::exp(-(kap - Fix::kappa) /
                                                       Fix::alpha)) /
                                       ((kap - Fix::kappa) / Fix::alpha))
                    : 1.0};

    auto && S_dam_ref{dam * S_lin};
    err = rel_error(S_dam, S_dam_ref);

    BOOST_CHECK_LT(err, Fix::get_tol());
  }

  BOOST_AUTO_TEST_SUITE_END()
}  // namespace muSpectre
