/**
 * file   bind_py_solver.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for the muSpectre solvers
 *
 * @section LICENCE
 *
 * Copyright © 2018 Till Junge
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

#include "common/common.hh"
#include "solver/solvers.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

/**
 * Solvers instanciated for systems with equal spatial and material dimension
 */

template <Dim_t sdim>
void add_newton_cg_helper(py::module & mod) {
  std::stringstream name_stream {};
  name_stream << "newton_cg" << sdim << "d";

  constexpr Dim_t mdim{sdim};
  using sys = SystemBase<sdim, mdim>;
  using grad = Grad_t<sdim>;
  using grad_vec = Grad_t<sdim>;

  mod.def(name_stream.str().c_str(),
          [](sys & s, const grad & g, Real ct, Real nt,
             Uint max, Dim_t verb) -> Dim_t {//typename sys::StrainField_t & {
            newton_cg(s, g, ct, nt, max, verb);
            return -1; //! temporary dummy return
          },
          "system"_a,
          "ΔF₀"_a,
          "cg_tol"_a,
          "newton_tol"_a,
          "maxiter"_a=0,
          "verbose"_a=0);
  mod.def(name_stream.str().c_str(),
          [](sys & s, const grad_vec & g, Real ct, Real nt,
             Uint max, Dim_t verb) -> Dim_t {//typename sys::StrainField_t & {
            newton_cg(s, g, ct, nt, max, verb);
            return -1;
          },
          "system"_a,
          "ΔF₀"_a,
          "cg_tol"_a,
          "newton_tol"_a,
          "maxiter"_a=0,
          "verbose"_a=0);
}

template <Dim_t sdim>
void add_de_geus_helper(py::module & mod) {
  std::stringstream name_stream {};
  name_stream << "de_geus" << sdim << "d";

  constexpr Dim_t mdim{sdim};
  using sys = SystemBase<sdim, mdim>;
  using grad = Grad_t<sdim>;
  using grad_vec = Grad_t<sdim>;

  mod.def(name_stream.str().c_str(),
          [](sys & s, const grad & g, Real ct, Real nt,
             Uint max, Dim_t verb) -> Dim_t {//typename sys::StrainField_t & {
            de_geus(s, g, ct, nt, max, verb);
            return -1;
          },
          "system"_a,
          "ΔF₀"_a,
          "cg_tol"_a,
          "newton_tol"_a,
          "maxiter"_a=0,
          "verbose"_a=0);
  mod.def(name_stream.str().c_str(),
          [](sys & s, const grad_vec & g, Real ct, Real nt,
             Uint max, Dim_t verb) -> Dim_t {//typename sys::StrainField_t & {
            de_geus(s, g, ct, nt, max, verb);
            return -1;
          },
          "system"_a,
          "ΔF₀"_a,
          "cg_tol"_a,
          "newton_tol"_a,
          "maxiter"_a=0,
          "verbose"_a=0);
}

template <Dim_t dim>
void add_solver_helper(py::module & mod) {
  add_newton_cg_helper<dim>(mod);
  add_de_geus_helper  <dim>(mod);
}

void add_solvers(py::module & mod) {
  auto solvers{mod.def_submodule("solvers")};
  solvers.doc() = "bindings for solvers";

  add_solver_helper<twoD  >(solvers);
  add_solver_helper<threeD>(solvers);
}
