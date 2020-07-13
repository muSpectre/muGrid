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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <libmugrid/exception.hh>
#include <libmugrid/numpy_tools.hh>

#include "projection/projection_small_strain.hh"
#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_approx_Green_operator.hh"

#include <libmufft/fftw_engine.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

using muFFT::Gradient_t;
using muGrid::DynRcoord_t;
using muGrid::Index_t;
using muGrid::numpy_wrap;
using muGrid::NumpyProxy;
using muGrid::Real;
using muGrid::RuntimeError;
using muSpectre::Formulation;
using muSpectre::MatrixXXc;
using muSpectre::ProjectionBase;
using pybind11::literals::operator""_a;
namespace py = pybind11;

class ProjectionBaseUnclonable : public ProjectionBase {
 public:
  ProjectionBaseUnclonable(const muFFT::FFTEngine_ptr & engine,
                           const DynRcoord_t & domain_lengths,
                           const Index_t & nb_quad_pts,
                           const Index_t & nb_components,
                           const Formulation & form)
      : ProjectionBase(engine, domain_lengths, nb_quad_pts, nb_components,
                       form) {}

  std::unique_ptr<ProjectionBase> clone() const final {
    throw RuntimeError(
        "Python version of the projection operators can't be cloned");
  }
};

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
class PyProjectionBase : public ProjectionBaseUnclonable {
 public:
  //! base class
  using Parent = ProjectionBase;
  //! field type on which projection is applied
  using Field_t = typename Parent::Field_t;
  //! shortcut fo strain shape
  using StrainShape_t = std::array<Index_t, 2>;

  PyProjectionBase(const muFFT::FFTEngine_ptr & engine,
                   const DynRcoord_t & domain_lengths,
                   const Index_t & nb_quad_pts,
                   const Index_t & nb_components, const Formulation & form)
      : ProjectionBaseUnclonable(engine, domain_lengths, nb_quad_pts,
                                 nb_components, form) {}

  void apply_projection(Field_t & field) override {
    PYBIND11_OVERLOAD_PURE(void, Parent, apply_projection, field);
  }

  //  Eigen::Map<MatrixXXc> get_operator() override {
  //    PYBIND11_OVERLOAD_PURE(Eigen::Map<MatrixXXc>, Parent, get_operator);
  //  }

  StrainShape_t get_strain_shape() const override {
    PYBIND11_OVERLOAD_PURE(StrainShape_t, Parent, get_strain_shape);
  }

  Index_t get_nb_dof_per_pixel() const override {
    PYBIND11_OVERLOAD_PURE(Index_t, Parent, get_nb_dof_per_pixel);
  }
};

void add_projection_base(py::module & mod) {
  py::class_<ProjectionBase,                   // class
             std::shared_ptr<ProjectionBase>,  // holder
             PyProjectionBase                  // trampoline base
             >(mod, "ProjectionBase")
      .def(py::init<const muFFT::FFTEngine_ptr &, const DynRcoord_t &,
                    const Index_t &, const Index_t &, const Formulation &>());
}

template <class Proj, Index_t DimS>
void add_proj_helper(py::module & mod, std::string name_start) {
  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<Proj,                   // class
             std::shared_ptr<Proj>,  // holder
             ProjectionBase          // trampoline base
             >(mod, name.str().c_str())
      .def(py::init<muFFT::FFTEngine_ptr, const DynRcoord_t &, Gradient_t>(),
           "fft_engine"_a, "domain_lengths"_a, "gradient"_a)
      .def(py::init<muFFT::FFTEngine_ptr, const DynRcoord_t &>(),
           "fft_engine"_a, "domain_lengths"_a)
      .def("initialise", &Proj::initialise,
           "initialises the projection operator")
      // apply_projection that takes Fields
      .def("apply_projection", &Proj::apply_projection)
      // apply_projection that takes numpy arrays
      .def("apply_projection",
           [](Proj & proj,
              py::array_t<Real, py::array::f_style> & vector_field) {
             py::buffer_info buffer{vector_field.request()};
             py::array_t<Real, py::array::f_style> proj_vector_field(
                 buffer.shape);
             py::buffer_info proj_buffer{proj_vector_field.request()};
             std::copy(static_cast<Real *>(buffer.ptr),
                       static_cast<Real *>(buffer.ptr) + buffer.size,
                       static_cast<Real *>(proj_buffer.ptr));

             auto strain_shape = proj.get_strain_shape();
             NumpyProxy<Real, py::array::f_style> proxy(
                 proj.get_nb_subdomain_grid_pts(),
                 proj.get_subdomain_locations(), proj.get_nb_quad_pts(),
                 {strain_shape[0], strain_shape[1]}, proj_vector_field);
             proj.apply_projection(proxy.get_field());
             return proj_vector_field;
           })
      .def_property_readonly("operator", &Proj::get_operator)
      .def_property_readonly("formulation", &Proj::get_formulation,
                             "return a Formulation enum indicating whether the "
                             "projection is small or finite strain")
      .def_property_readonly("nb_subdomain_grid_pts",
                             &Proj::get_nb_subdomain_grid_pts)
      .def_property_readonly("subdomain_locations",
                             &Proj::get_subdomain_locations)
      .def_property_readonly("nb_domain_grid_pts",
                             &Proj::get_nb_domain_grid_pts)
      .def_property_readonly("domain_lengths", &Proj::get_nb_domain_grid_pts);
}

template <class Proj, Index_t DimS>
void add_green_proj_helper(py::module & mod, std::string name_start) {
  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<Proj,                   // class
             std::shared_ptr<Proj>,  // holder
             ProjectionBase          // trampoline base
             >(mod, name.str().c_str())

      .def(py::init<muFFT::FFTEngine_ptr, const DynRcoord_t &,
                    const Eigen::Ref<Eigen::Matrix<muFFT::Real, Eigen::Dynamic,
                                                   Eigen::Dynamic>> &,
                    Gradient_t>())
      .def(py::init([](muFFT::FFTEngine_ptr fft_engine,
                       const DynRcoord_t & domain_lenghts,
                       py::EigenDRef<Eigen::MatrixXd> C_ref) {
        Eigen::MatrixXd tmp{C_ref};
        return Proj(std::move(fft_engine), domain_lenghts, tmp);
      }))
      .def("initialise", &Proj::initialise,
           "initialises the projection operator")
      // apply_projection that takes Fields
      .def("apply_projection", &Proj::apply_projection)

      .def_property_readonly("operator", &Proj::get_operator)
      .def_property_readonly("formulation", &Proj::get_formulation,
                             "return a Formulation enum indicating whether the "
                             "projection is small or finite strain")
      .def_property_readonly("nb_subdomain_grid_pts",
                             &Proj::get_nb_subdomain_grid_pts)
      .def_property_readonly("subdomain_locations",
                             &Proj::get_subdomain_locations)
      .def_property_readonly("nb_domain_grid_pts",
                             &Proj::get_nb_domain_grid_pts)
      .def_property_readonly("domain_lengths", &Proj::get_nb_domain_grid_pts);
}

void add_projections(py::module & mod) {
  add_projection_base(mod);
  add_proj_helper<muSpectre::ProjectionSmallStrain<muGrid::twoD>, muGrid::twoD>(
      mod, "ProjectionSmallStrain");
  add_proj_helper<muSpectre::ProjectionSmallStrain<muGrid::threeD>,
                  muGrid::threeD>(mod, "ProjectionSmallStrain");

  add_proj_helper<muSpectre::ProjectionFiniteStrain<muGrid::twoD>,
                  muGrid::twoD>(mod, "ProjectionFiniteStrain");
  add_proj_helper<muSpectre::ProjectionFiniteStrain<muGrid::threeD>,
                  muGrid::threeD>(mod, "ProjectionFiniteStrain");

  add_proj_helper<muSpectre::ProjectionFiniteStrainFast<muGrid::twoD>,
                  muGrid::twoD>(mod, "ProjectionFiniteStrainFast");
  add_proj_helper<muSpectre::ProjectionFiniteStrainFast<muGrid::threeD>,
                  muGrid::threeD>(mod, "ProjectionFiniteStrainFast");

  add_proj_helper<
      muSpectre::ProjectionFiniteStrainFast<muGrid::twoD, muGrid::TwoQuadPts>,
      muGrid::twoD>(mod, "ProjectionFiniteStrainFast_2q");
  add_proj_helper<
      muSpectre::ProjectionFiniteStrainFast<muGrid::threeD, muGrid::TwoQuadPts>,
      muGrid::threeD>(mod, "ProjectionFiniteStrainFast_2q");

  add_green_proj_helper<muSpectre::ProjectionApproxGreenOperator<muGrid::twoD>,
                        muGrid::twoD>(mod, "ProjectionApproxGreenOperator");
  add_green_proj_helper<
      muSpectre::ProjectionApproxGreenOperator<muGrid::threeD>, muGrid::threeD>(
      mod, "ProjectionApproxGreenOperator");
}
