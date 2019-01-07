/**
 * @file   py_comparison_test_material_linear_elastic_1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Dec 2018
 *
 * @brief  simple wrapper around the MaterialLinearElastic1 to test it
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
 * General Public License for more details.
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
 */

#include "materials/stress_transformations_PK1.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/materials_toolbox.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <tuple>

using pybind11::literals::operator""_a;
namespace py = pybind11;

namespace muSpectre {
  template <Dim_t Dim>
  std::tuple<Eigen::Matrix<Real, Dim, Dim>,
             Eigen::Matrix<Real, Dim * Dim, Dim * Dim>>
  material_wrapper(Real Young, Real Poisson,
                   py::EigenDRef<const Eigen::MatrixXd> F) {
    using Mat_t = MaterialLinearElastic1<Dim, Dim>;
    Mat_t mat("Name", Young, Poisson);

    auto & coll{mat.get_collection()};
    coll.add_pixel({0});
    coll.initialise();

    Eigen::Matrix<Real, Dim, Dim> F_mat(F);
    Eigen::Matrix<Real, Dim, Dim> E{
        MatTB::convert_strain<StrainMeasure::Gradient,
                              StrainMeasure::GreenLagrange>(F_mat)};
    return mat.evaluate_stress_tangent(std::move(E));
  }

  template <Dim_t Dim>
  py::tuple PK2_fun(Real Young, Real Poisson,
                    py::EigenDRef<const Eigen::MatrixXd> F) {
    auto && tup{material_wrapper<Dim>(Young, Poisson, F)};
    auto && S{std::get<0>(tup)};
    Eigen::MatrixXd C{std::get<1>(tup)};

    return py::make_tuple(Eigen::MatrixXd{S}, C);
  }

  template <Dim_t Dim>
  py::tuple PK1_fun(Real Young, Real Poisson,
                    py::EigenDRef<const Eigen::MatrixXd> F) {
    auto && tup{material_wrapper<Dim>(Young, Poisson, F)};
    auto && S{std::get<0>(tup)};
    auto && C{std::get<1>(tup)};

    using Mat_t = MaterialLinearElastic1<Dim, Dim>;
    constexpr auto StrainM{Mat_t::traits::strain_measure};
    constexpr auto StressM{Mat_t::traits::stress_measure};

    auto && PK_tup{MatTB::PK1_stress<StressM, StrainM>(
        Eigen::Matrix<Real, Dim, Dim>{F}, S, C)};
    auto && P{std::get<0>(PK_tup)};
    auto && K{std::get<1>(PK_tup)};

    return py::make_tuple(std::move(P), std::move(K));
  }

  PYBIND11_MODULE(material_linear_elastic1, mod) {
    mod.doc() = "Comparison provider for MaterialLinearelastic1";
    auto adder{[&mod](auto name, auto & fun) {
      mod.def(name, &fun, "Young"_a, "Poisson"_a, "F"_a);
    }};
    adder("PK2_fun_2d", PK2_fun<twoD>);
    adder("PK2_fun_3d", PK2_fun<threeD>);
    adder("PK1_fun_2d", PK1_fun<twoD>);
    adder("PK1_fun_3d", PK1_fun<threeD>);
  }

}  // namespace muSpectre
