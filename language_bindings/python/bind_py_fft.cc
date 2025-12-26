/**
 * @file   bind_py_fft.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Python bindings for FFT classes
 *
 * Copyright © 2024 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include "bind_py_declarations.hh"

#include "fft/fft_engine.hh"
#include "fft/fft_utils.hh"
#include "field/field.hh"
#include "collection/field_collection.hh"
#include "util/python_helpers.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using muGrid::FFTEngineBase;
using muGrid::FFTEngine;
using muGrid::HostSpace;
using muGrid::Communicator;

// Alias for Python binding - use HostSpace specialization
using PyFFTEngine = FFTEngine<HostSpace>;
using muGrid::Int;
using muGrid::Index_t;
using muGrid::DynGridIndex;
using muGrid::Real;
using muGrid::Complex;
using muGrid::Field;
using muGrid::GlobalFieldCollection;
using muGrid::py_coords;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_fft_utils(py::module & mod) {
  // FFT frequency index functions
  mod.def(
      "fft_freqind",
      [](Index_t n) { return muGrid::fft_freqind(n); },
      "n"_a,
      R"(
      Compute the frequency bin indices for a full c2c FFT.

      For n samples, returns indices [0, 1, ..., n/2-1, -n/2, ..., -1] for even n,
      or [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] for odd n.

      This is equivalent to numpy.fft.fftfreq(n) * n.

      Parameters
      ----------
      n : int
          Number of points in the transform

      Returns
      -------
      list of int
          Vector of integer frequency indices
      )");

  mod.def(
      "fft_freq",
      [](Index_t n, Real d) { return muGrid::fft_freq(n, d); },
      "n"_a, "d"_a = 1.0,
      R"(
      Compute the frequency values for a full c2c FFT.

      Returns frequencies in cycles per unit spacing: f = k / (n * d)
      where k is the frequency index from fft_freqind.

      This is equivalent to numpy.fft.fftfreq(n, d).

      Parameters
      ----------
      n : int
          Number of points in the transform
      d : float, optional
          Sample spacing (default 1.0)

      Returns
      -------
      list of float
          Vector of frequency values
      )");

  mod.def(
      "rfft_freqind",
      [](Index_t n) { return muGrid::rfft_freqind(n); },
      "n"_a,
      R"(
      Compute the frequency bin indices for a r2c (half-complex) FFT.

      For n real input samples, the r2c transform produces n/2+1 complex outputs.
      Returns indices [0, 1, ..., n/2].

      This is equivalent to numpy.fft.rfftfreq(n) * n.

      Parameters
      ----------
      n : int
          Number of real input points

      Returns
      -------
      list of int
          Vector of integer frequency indices (length n/2+1)
      )");

  mod.def(
      "rfft_freq",
      [](Index_t n, Real d) { return muGrid::rfft_freq(n, d); },
      "n"_a, "d"_a = 1.0,
      R"(
      Compute the frequency values for a r2c (half-complex) FFT.

      Returns frequencies in cycles per unit spacing: f = k / (n * d)
      where k is the frequency index from rfft_freqind.

      This is equivalent to numpy.fft.rfftfreq(n, d).

      Parameters
      ----------
      n : int
          Number of real input points
      d : float, optional
          Sample spacing (default 1.0)

      Returns
      -------
      list of float
          Vector of frequency values (length n/2+1)
      )");

  mod.def(
      "get_hermitian_grid_pts",
      [](const DynGridIndex & nb_grid_pts, Index_t r2c_axis) {
        return muGrid::get_hermitian_grid_pts(nb_grid_pts, r2c_axis);
      },
      "nb_grid_pts"_a, "r2c_axis"_a = 0,
      R"(
      Compute the Fourier grid dimensions for a half-complex r2c transform.

      For a real-space grid of size [Nx, Ny, Nz], the half-complex Fourier grid
      has size [Nx/2+1, Ny, Nz] (for r2c_axis=0).

      Parameters
      ----------
      nb_grid_pts : list of int
          Real-space grid dimensions
      r2c_axis : int, optional
          Axis along which r2c transform is performed (default 0)

      Returns
      -------
      list of int
          Fourier-space grid dimensions
      )");

  mod.def(
      "fft_normalization",
      [](const DynGridIndex & nb_grid_pts) {
        return muGrid::fft_normalization(nb_grid_pts);
      },
      "nb_grid_pts"_a,
      R"(
      Compute the normalization factor for FFT roundtrip.

      For an unnormalized FFT, ifft(fft(x)) = N * x where N is the total number
      of grid points. This returns 1/N for normalizing the result.

      Parameters
      ----------
      nb_grid_pts : list of int
          Grid dimensions

      Returns
      -------
      float
          Normalization factor (1.0 / total_grid_points)
      )");
}

void add_fft_engine(py::module & mod) {
  using MemoryLocation = muGrid::FieldCollection::MemoryLocation;
  using SubPtMap_t = muGrid::FieldCollection::SubPtMap_t;

  py::class_<PyFFTEngine, muGrid::CartesianDecomposition>(mod, "FFTEngine",
      R"(
      Distributed FFT engine using pencil (2D) decomposition.

      This class provides distributed FFT operations on structured grids with
      MPI parallelization. It uses pencil decomposition which allows efficient
      scaling to large numbers of ranks.

      Key features:
      - Supports 2D and 3D grids
      - Handles arbitrary ghost buffer configurations in real space
      - No ghosts in Fourier space (hard assumption)
      - Supports host and device (GPU) memory
      - Unnormalized transforms (like FFTW)

      The engine owns field collections for both real and Fourier space, and
      work buffers for intermediate results during the distributed FFT.

      Examples
      --------
      >>> engine = FFTEngine([64, 64, 64])
      >>> real_field = engine.real_space_field("displacement", 3)
      >>> fourier_field = engine.fourier_space_field("displacement_k", 3)
      >>> engine.fft(real_field, fourier_field)
      >>> engine.ifft(fourier_field, real_field)
      >>> real_field.s[:] *= engine.normalisation
      )")
      .def(py::init<const DynGridIndex &, const Communicator &, const DynGridIndex &,
                    const DynGridIndex &, const SubPtMap_t &>(),
           "nb_domain_grid_pts"_a,
           "comm"_a = Communicator(),
           "nb_ghosts_left"_a = DynGridIndex{},
           "nb_ghosts_right"_a = DynGridIndex{},
           "nb_sub_pts"_a = SubPtMap_t{},
           R"(
           Construct an FFT engine with pencil decomposition.

           Parameters
           ----------
           nb_domain_grid_pts : list of int
               Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
           comm : Communicator, optional
               MPI communicator (default: serial)
           nb_ghosts_left : list of int, optional
               Ghost cells on low-index side of each dimension
           nb_ghosts_right : list of int, optional
               Ghost cells on high-index side of each dimension
           nb_sub_pts : dict, optional
               Number of sub-points per pixel
           )")

      // Transform operations
      .def("fft",
           [](PyFFTEngine & self, const Field & input, Field & output) {
             self.fft(input, output);
           },
           "input"_a, "output"_a,
           R"(
           Forward FFT: real space -> Fourier space.

           The transform is unnormalized. To recover the original data after
           ifft(fft(x)), multiply by normalisation.

           Parameters
           ----------
           input : Field
               Real-space field (must be in this engine's real collection)
           output : Field
               Fourier-space field (must be in this engine's Fourier collection)
           )")

      .def("ifft",
           [](PyFFTEngine & self, const Field & input, Field & output) {
             self.ifft(input, output);
           },
           "input"_a, "output"_a,
           R"(
           Inverse FFT: Fourier space -> real space.

           The transform is unnormalized. To recover the original data after
           ifft(fft(x)), multiply by normalisation.

           Parameters
           ----------
           input : Field
               Fourier-space field (must be in this engine's Fourier collection)
           output : Field
               Real-space field (must be in this engine's real collection)
           )")

      // Field registration
      .def("real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::register_real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Register a real-space field.

           Parameters
           ----------
           name : str
               Unique field name
           nb_components : int, optional
               Number of components (default 1)

           Returns
           -------
           Field
               Reference to the created field
           )")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::register_fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Register a Fourier-space field.

           Parameters
           ----------
           name : str
               Unique field name
           nb_components : int, optional
               Number of components (default 1)

           Returns
           -------
           Field
               Reference to the created field
           )")

      // Collection access
      .def_property_readonly("real_space_collection",
           py::overload_cast<>(&PyFFTEngine::get_real_space_collection),
           py::return_value_policy::reference_internal,
           "Get the real-space field collection.")

      .def_property_readonly("fourier_space_collection",
           py::overload_cast<>(&PyFFTEngine::get_fourier_space_collection),
           py::return_value_policy::reference_internal,
           "Get the Fourier-space field collection.")

      // Geometry queries
      .def_property_readonly("normalisation", &PyFFTEngine::normalisation,
           R"(
           Get the normalization factor for FFT roundtrip.
           Multiply ifft output by this to recover original values.
           )")

      .def_property_readonly("nb_fourier_grid_pts",
           &PyFFTEngine::get_nb_fourier_grid_pts,
           R"(
           Get the global Fourier grid dimensions.
           For r2c transform: [Nx/2+1, Ny, Nz]
           )")

      .def_property_readonly("nb_fourier_subdomain_grid_pts",
           &PyFFTEngine::get_nb_fourier_subdomain_grid_pts,
           "Get the local Fourier grid dimensions on this rank.")

      .def_property_readonly("fourier_subdomain_locations",
           &PyFFTEngine::get_fourier_subdomain_locations,
           "Get the starting location of this rank's Fourier subdomain.")

      .def_property_readonly("process_grid",
           [](const PyFFTEngine & self) {
             auto grid = self.get_process_grid();
             return py::make_tuple(grid[0], grid[1]);
           },
           "Get the 2D process grid dimensions (P1, P2).")

      .def_property_readonly("process_coords",
           [](const PyFFTEngine & self) {
             auto coords = self.get_process_coords();
             return py::make_tuple(coords[0], coords[1]);
           },
           "Get this rank's coordinates in the process grid (p1, p2).")

      .def_property_readonly("backend_name",
           &PyFFTEngine::get_backend_name,
           "Get the name of the FFT backend being used.");
}

void add_fft_classes(py::module & mod) {
  add_fft_utils(mod);
  add_fft_engine(mod);
}
