/**
 * @file   bind_py_projections.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   18 Jan 2018
 *
 * @brief  Python bindings for the Projection operators
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

#include "fft/projection_small_strain.hh"
#include "fft/projection_finite_strain.hh"
#include "fft/projection_finite_strain_fast.hh"

#include "fft/fftw_engine.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <memory>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
template <Dim_t DimS, Dim_t DimM>
class PyProjectionBase: public ProjectionBase<DimS, DimM> {
public:
  //! base class
  using Parent = ProjectionBase<DimS, DimM>;
  //! field type on which projection is applied
  using Field_t = typename Parent::Field_t;

  void apply_projection(Field_t & field) override {
    PYBIND11_OVERLOAD_PURE
      (void,
       Parent,
       apply_projection,
       field
       );
  }

  Eigen::Map<Eigen::ArrayXXd> get_operator() override {
    PYBIND11_OVERLOAD_PURE
      (Eigen::Map<Eigen::ArrayXXd>,
       Parent,
       get_operator
       );
  }
};

template <class Proj, Dim_t DimS, Dim_t DimM=DimS>
void add_proj_helper(py::module & mod, std::string name_start) {
  using Ccoord = Ccoord_t<DimS>;
  using Rcoord = Rcoord_t<DimS>;
  using Engine = FFTWEngine<DimS, DimM>;
  using Field_t = typename Proj::Field_t;

  static_assert(DimS == DimM,
                "currently only for DimS==DimM");

  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<Proj>(mod, name.str().c_str())
    .def(py::init([](Ccoord res, Rcoord lengths) {
          auto engine = std::make_unique<Engine>(res);
          return Proj(std::move(engine), lengths);
        }))
    .def("initialise", &Proj::initialise,
         "flags"_a=FFT_PlanFlags::estimate,
         "initialises the fft engine (plan the transform)")
    .def("apply_projection",
         [](Proj & proj, py::EigenDRef<Eigen::ArrayXXd> v){
           typename Engine::GFieldCollection_t coll{};
           coll.initialise(proj.get_subdomain_resolutions(),
                           proj.get_subdomain_locations());
           Field_t & temp{make_field<Field_t>("temp_field", coll)};
           temp.eigen() = v;
           proj.apply_projection(temp);
           return Eigen::ArrayXXd{temp.eigen()};
         })
    .def("get_operator", &Proj::get_operator)
    .def("get_formulation", &Proj::get_formulation,
         "return a Formulation enum indicating whether the projection is small"
         " or finite strain");
}

void add_proj_dispatcher(py::module & mod) {
  add_proj_helper<
    ProjectionSmallStrain<  twoD,   twoD>,
    twoD>(mod, "ProjectionSmallStrain");
  add_proj_helper<
    ProjectionSmallStrain<threeD, threeD>,
    threeD>(mod, "ProjectionSmallStrain");

  add_proj_helper<
    ProjectionFiniteStrain<  twoD,   twoD>,
    twoD>(mod, "ProjectionFiniteStrain");
  add_proj_helper<
    ProjectionFiniteStrain<threeD, threeD>,
    threeD>(mod, "ProjectionFiniteStrain");

  add_proj_helper<
    ProjectionFiniteStrainFast<  twoD,   twoD>,
    twoD>(mod, "ProjectionFiniteStrainFast");
  add_proj_helper<
    ProjectionFiniteStrainFast<threeD, threeD>,
    threeD>(mod, "ProjectionFiniteStrainFast");

}

void add_projections(py::module & mod) {
  add_proj_dispatcher(mod);

}
