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

#include "projection/projection_small_strain.hh"
#include "projection/projection_finite_strain.hh"
#include "projection/projection_finite_strain_fast.hh"

#include <libmufft/fftw_engine.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <memory>

using muFFT::Gradient_t;
using muGrid::Dim_t;
using muGrid::DynRcoord_t;
using muSpectre::MatrixXXc;
using muSpectre::ProjectionBase;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
template <Dim_t DimS>
class PyProjectionBase : public ProjectionBase {
 public:
  //! base class
  using Parent = ProjectionBase;
  //! field type on which projection is applied
  using Field_t = typename Parent::Field_t;

  void apply_projection(Field_t & field) override {
    PYBIND11_OVERLOAD_PURE(void, Parent, apply_projection, field);
  }

  Eigen::Map<MatrixXXc> get_operator() override {
    PYBIND11_OVERLOAD_PURE(Eigen::Map<MatrixXXc>, Parent, get_operator);
  }
};

template <class Proj, Dim_t DimS>
void add_proj_helper(py::module & mod, std::string name_start) {
  using Field_t = typename Proj::Field_t;

#ifdef WITH_MPI
  auto make_proj = [](Ccoord res, Rcoord lengths, const Gradient_t & gradient,
                      const std::string & fft,
                      const muFFT::Communicator & comm) {
    if (fft == "fftw") {
      auto engine = std::make_unique<muFFT::FFTWEngine<DimS>>(
          res, Proj::NbComponents(), comm);
      return Proj(std::move(engine), lengths, gradient);
#ifdef WITH_FFTWMPI
    } else if (fft == "fftwmpi") {
      auto engine = std::make_unique<muFFT::FFTWMPIEngine<DimS>>(
          res, Proj::NbComponents(), comm);
      return Proj(std::move(engine), lengths, gradient);
#endif
#ifdef WITH_PFFT
    } else if (fft == "pfft") {
      auto engine = std::make_unique<muFFT::PFFTEngine<DimS>>(
          res, Proj::NbComponents(), comm);
      return Proj(std::move(engine), lengths, gradient);
#endif
    } else {
      throw std::runtime_error("Unknown FFT engine '" + fft + "' specified.");
    }
  };
#endif

  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<Proj>(mod, name.str().c_str())
#ifdef WITH_MPI
      .def(py::init(make_proj), "nb_grid_pts"_a, "lengths"_a,
           "gradient"_a = muSpectre::make_fourier_gradient<DimS>(),
           "fft"_a = "fftw", "communicator"_a = muFFT::Communicator())
      .def(py::init([make_proj](Ccoord res, Rcoord lengths,
                                const Gradient_t & gradient,
                                const std::string & fft, size_t comm) {
             return make_proj(res, lengths, gradient, fft,
                              std::move(muFFT::Communicator(MPI_Comm(comm))));
           }),
           "nb_grid_pts"_a, "lengths"_a,
           "gradient"_a = muSpectre::make_fourier_gradient<DimS>(),
           "fft"_a = "fftw", "communicator"_a = size_t(MPI_COMM_SELF))
#else
      .def(py::init([](Ccoord res, Rcoord lengths, const Gradient_t & gradient,
                       const std::string & fft) {
             if (fft == "fftw") {
               auto engine = std::make_unique<muFFT::FFTWEngine<DimS>>(
                   res, Proj::NbComponents());
               return Proj(std::move(engine), lengths, gradient);
             } else {
               throw std::runtime_error("Unknown FFT engine '" + fft +
                                        "' specified.");
             }
           }),
           "nb_grid_pts"_a, "lengths"_a,
           "gradient"_a = muSpectre::make_fourier_gradient<DimS>(),
           "fft"_a = "fftw")
#endif
      .def("initialise", &Proj::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate,
           "initialises the fft engine (plan the transform)")
      // apply_projection that takes Fields
      .def("apply_projection", &Proj::apply_projection)
      // apply_projection that takes numpy arrays
      /*
      .def("apply_projection",
           [](Proj & proj, py::EigenDRef<Eigen::ArrayXXd> v) {
             typename muFFT::FFTEngineBase::GFieldCollection_t coll{1};
             Eigen::Index subdomain_size =
                 muGrid::CcoordOps::get_size(proj.get_nb_subdomain_grid_pts());
             if (v.rows() != DimS * DimS || v.cols() != subdomain_size) {
               throw std::runtime_error("Expected input array of shape (" +
                                        std::to_string(DimS * DimS) + ", " +
                                        std::to_string(subdomain_size) +
                                        "), but input array has shape (" +
                                        std::to_string(v.rows()) + ", " +
                                        std::to_string(v.cols()) + ").");
             }
             coll.initialise(proj.get_nb_subdomain_grid_pts(),
                             proj.get_subdomain_locations());
             Field_t & temp{coll.template register_field<
                typename Field_t::Element_t>("temp_field",
                                             proj.get_nb_components())};
             temp.eigen_pixel() = v;
             proj.apply_projection(temp);
             return Eigen::ArrayXXd{temp.eigen_pixel()};
           })
           */
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
  add_proj_helper<muSpectre::ProjectionSmallStrain<muGrid::twoD>,
                  muGrid::twoD>(mod, "ProjectionSmallStrain");
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
}
