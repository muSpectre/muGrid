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
 * General Public License for more details.
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
 */

#include "fft/projection_small_strain.hh"
#include "fft/projection_finite_strain.hh"
#include "fft/projection_finite_strain_fast.hh"

#include "fft/fftw_engine.hh"
#ifdef WITH_FFTWMPI
#include "fft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "fft/pfft_engine.hh"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <memory>

using namespace muSpectre;  // NOLINT // TODO(junge): figure this out
namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT: recommended usage

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
template <Dim_t DimS, Dim_t DimM>
class PyProjectionBase : public ProjectionBase<DimS, DimM> {
 public:
  //! base class
  using Parent = ProjectionBase<DimS, DimM>;
  //! field type on which projection is applied
  using Field_t = typename Parent::Field_t;

  void apply_projection(Field_t &field) override {
    PYBIND11_OVERLOAD_PURE(void, Parent, apply_projection, field);
  }

  Eigen::Map<Eigen::ArrayXXd> get_operator() override {
    PYBIND11_OVERLOAD_PURE(Eigen::Map<Eigen::ArrayXXd>, Parent, get_operator);
  }
};

template <class Proj, Dim_t DimS, Dim_t DimM = DimS>
void add_proj_helper(py::module &mod, std::string name_start) {
  using Ccoord = Ccoord_t<DimS>;
  using Rcoord = Rcoord_t<DimS>;
  using Field_t = typename Proj::Field_t;

  static_assert(DimS == DimM, "currently only for DimS==DimM");

  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<Proj>(mod, name.str().c_str())
#ifdef WITH_MPI
      .def(py::init([](Ccoord res, Rcoord lengths, const std::string &fft,
                       size_t comm) {
             if (fft == "fftw") {
               auto engine = std::make_unique<FFTWEngine<DimS>>(
                   res, Proj::NbComponents(),
                   std::move(Communicator(MPI_Comm(comm))));
               return Proj(std::move(engine), lengths);
#else
      .def(py::init([](Ccoord res, Rcoord lengths, const std::string &fft) {
             if (fft == "fftw") {
               auto engine = std::make_unique<FFTWEngine<DimS>>(
                   res, Proj::NbComponents());
               return Proj(std::move(engine), lengths);
#endif
#ifdef WITH_FFTWMPI
             } else if (fft == "fftwmpi") {
               auto engine = std::make_unique<FFTWMPIEngine<DimS>>(
                   res, Proj::NbComponents(),
                   std::move(Communicator(MPI_Comm(comm))));
               return Proj(std::move(engine), lengths);
#endif
#ifdef WITH_PFFT
             } else if (fft == "pfft") {
               auto engine = std::make_unique<PFFTEngine<DimS>>(
                   res, Proj::NbComponents(),
                   std::move(Communicator(MPI_Comm(comm))));
               return Proj(std::move(engine), lengths);
#endif
             } else {
               throw std::runtime_error("Unknown FFT engine '" + fft +
                                        "' specified.");
             }
           }),
           "resolutions"_a, "lengths"_a,
#ifdef WITH_MPI
           "fft"_a = "fftw", "communicator"_a = size_t(MPI_COMM_SELF))
#else
           "fft"_a = "fftw")
#endif
      .def("initialise", &Proj::initialise, "flags"_a = FFT_PlanFlags::estimate,
           "initialises the fft engine (plan the transform)")
      .def("apply_projection",
           [](Proj &proj, py::EigenDRef<Eigen::ArrayXXd> v) {
             typename FFTEngineBase<DimS>::GFieldCollection_t coll{};
             Eigen::Index subdomain_size =
                 CcoordOps::get_size(proj.get_subdomain_resolutions());
             if (v.rows() != DimS * DimM || v.cols() != subdomain_size) {
               throw std::runtime_error("Expected input array of shape (" +
                                        std::to_string(DimS * DimM) + ", " +
                                        std::to_string(subdomain_size) +
                                        "), but input array has shape (" +
                                        std::to_string(v.rows()) + ", " +
                                        std::to_string(v.cols()) + ").");
             }
             coll.initialise(proj.get_subdomain_resolutions(),
                             proj.get_subdomain_locations());
             Field_t &temp{make_field<Field_t>("temp_field", coll,
                                               proj.get_nb_components())};
             temp.eigen() = v;
             proj.apply_projection(temp);
             return Eigen::ArrayXXd{temp.eigen()};
           })
      .def("get_operator", &Proj::get_operator)
      .def(
          "get_formulation", &Proj::get_formulation,
          "return a Formulation enum indicating whether the projection is small"
          " or finite strain")
      .def("get_subdomain_resolutions", &Proj::get_subdomain_resolutions)
      .def("get_subdomain_locations", &Proj::get_subdomain_locations)
      .def("get_domain_resolutions", &Proj::get_domain_resolutions)
      .def("get_domain_lengths", &Proj::get_domain_resolutions);
}

void add_proj_dispatcher(py::module &mod) {
  add_proj_helper<ProjectionSmallStrain<twoD, twoD>, twoD>(
      mod, "ProjectionSmallStrain");
  add_proj_helper<ProjectionSmallStrain<threeD, threeD>, threeD>(
      mod, "ProjectionSmallStrain");

  add_proj_helper<ProjectionFiniteStrain<twoD, twoD>, twoD>(
      mod, "ProjectionFiniteStrain");
  add_proj_helper<ProjectionFiniteStrain<threeD, threeD>, threeD>(
      mod, "ProjectionFiniteStrain");

  add_proj_helper<ProjectionFiniteStrainFast<twoD, twoD>, twoD>(
      mod, "ProjectionFiniteStrainFast");
  add_proj_helper<ProjectionFiniteStrainFast<threeD, threeD>, threeD>(
      mod, "ProjectionFiniteStrainFast");
}

void add_projections(py::module &mod) { add_proj_dispatcher(mod); }
