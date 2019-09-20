/**
 * @file   py_comparison_test_material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   30 Oct 2018
 *
 * @brief  simple wrapper around the MaterialHyperElastoPlastic1 to test it
 *         with arbitrary input
 *
 * Copyright © 2018 Till Junge
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

#include "materials/stress_transformations_Kirchhoff.hh"
#include "materials/material_hyper_elasto_plastic1.hh"
#include "materials/materials_toolbox.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <tuple>

using pybind11::literals::operator""_a;
namespace py = pybind11;

namespace muSpectre {

  template <Dim_t Dim>
  decltype(auto) material_wrapper(Real K, Real mu, Real H, Real tau_y0,
                                  py::EigenDRef<const Eigen::MatrixXd> F,
                                  py::EigenDRef<const Eigen::MatrixXd> F_prev,
                                  py::EigenDRef<const Eigen::MatrixXd> be_prev,
                                  Real eps_prev) {
    const Real Young{MatTB::convert_elastic_modulus<
        ElasticModulus::Young, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    const Real Poisson{MatTB::convert_elastic_modulus<
        ElasticModulus::Poisson, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};

    using Mat_t = MaterialHyperElastoPlastic1<Dim>;
    Mat_t mat("Name", Dim, 1, Young, Poisson, tau_y0, H);

    auto & coll{mat.get_collection()};
    coll.add_pixel({0});
    coll.initialise();

    auto & F_{mat.get_F_prev_field()};
    auto & be_{mat.get_be_prev_field()};
    auto & eps_{mat.get_plast_flow_field()};

    F_.get_map().get_current_static() = F_prev;
    be_.get_map().get_current_static() = be_prev;
    eps_.get_map().get_current_static() = eps_prev;
    mat.save_history_variables();

    return mat.evaluate_stress_tangent(F, 0);
  }

  template <Dim_t Dim>
  py::tuple kirchhoff_fun(Real K, Real mu, Real H, Real tau_y0,
                          py::EigenDRef<const Eigen::MatrixXd> F,
                          py::EigenDRef<const Eigen::MatrixXd> F_prev,
                          py::EigenDRef<const Eigen::MatrixXd> be_prev,
                          Real eps_prev) {
    auto && tup{
        material_wrapper<Dim>(K, mu, H, tau_y0, F, F_prev, be_prev, eps_prev)};
    auto && tau{std::get<0>(tup)};
    auto && C{std::get<1>(tup)};

    return py::make_tuple(std::move(tau), std::move(C));
  }

  template <Dim_t Dim>
  py::tuple PK1_fun(Real K, Real mu, Real H, Real tau_y0,
                    py::EigenDRef<const Eigen::MatrixXd> F,
                    py::EigenDRef<const Eigen::MatrixXd> F_prev,
                    py::EigenDRef<const Eigen::MatrixXd> be_prev,
                    Real eps_prev) {
    auto && tup{
        material_wrapper<Dim>(K, mu, H, tau_y0, F, F_prev, be_prev, eps_prev)};
    auto && tau{std::get<0>(tup)};
    auto && C{std::get<1>(tup)};

    using Mat_t = MaterialHyperElastoPlastic1<Dim>;
    constexpr auto StrainM{Mat_t::traits::strain_measure};
    constexpr auto StressM{Mat_t::traits::stress_measure};

    auto && PK_tup{MatTB::PK1_stress<StressM, StrainM>(
        Eigen::Matrix<Real, Dim, Dim>{F}, tau, C)};
    auto && P{std::get<0>(PK_tup)};
    auto && K_{std::get<1>(PK_tup)};

    return py::make_tuple(std::move(P), std::move(K_));
  }

  PYBIND11_MODULE(material_hyper_elasto_plastic1, mod) {
    mod.doc() = "Comparison provider for MaterialHyperElastoPlastic1";
    auto adder{[&mod](auto name, auto & fun) {
      mod.def(name, &fun, "K"_a, "mu"_a, "H"_a, "tau_y0"_a, "F"_a, "F_prev"_a,
              "be_prev"_a, "eps_prev"_a);
    }};
    adder("kirchhoff_fun_2d", kirchhoff_fun<twoD>);
    adder("kirchhoff_fun_3d", kirchhoff_fun<threeD>);
    adder("PK1_fun_2d", PK1_fun<twoD>);
    adder("PK1_fun_3d", PK1_fun<threeD>);
  }

}  // namespace muSpectre
