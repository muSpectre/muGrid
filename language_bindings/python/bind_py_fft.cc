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
using muGrid::Dim_t;
using muGrid::Index_t;
using muGrid::DynGridIndex;
using muGrid::Real;
using muGrid::Complex;
using muGrid::Field;
using muGrid::GlobalFieldCollection;
using muGrid::py_coords;
using pybind11::literals::operator""_a;

namespace py = pybind11;

/**
 * Generate real-space coordinate array for a distributed FFT engine.
 *
 * Returns an array of shape [dim, local_nx, local_ny, ...] containing
 * the fractional coordinates for each pixel in the local subdomain.
 *
 * For T=Real: coordinates are normalized (in range [0, 1))
 * For T=Int: coordinates are integer indices
 *
 * @tparam T Output type (Real or Int)
 * @tparam with_ghosts Include ghost cells if true
 * @param eng The FFT engine
 * @return numpy array of coordinates
 */
template<typename T, bool with_ghosts>
auto py_fft_coords(const PyFFTEngine & eng) {
    const auto & real_collection = eng.get_real_space_collection();
    const auto & pixels = with_ghosts
        ? real_collection.get_pixels_with_ghosts()
        : real_collection.get_pixels_without_ghosts();
    const auto & nb_domain_grid_pts {eng.get_nb_domain_grid_pts()};
    const Dim_t dim{eng.get_spatial_dim()};

    // Shape: [dim, local_nx, local_ny, ...]
    std::vector<Index_t> shape;
    shape.push_back(dim);
    for (auto && n : pixels.get_nb_subdomain_grid_pts()) {
        shape.push_back(n);
    }

    // Strides: first index (dim) is contiguous
    std::vector<Index_t> strides;
    strides.push_back(sizeof(T));
    for (auto && s : pixels.get_strides()) {
        strides.push_back(s * dim * sizeof(T));
    }

    py::array_t<T> coords(shape, strides);
    T * ptr = static_cast<T *>(coords.request().ptr);

    // Iterate over local pixels and compute coordinates
    for (auto && pix : pixels.coordinates()) {
        for (Index_t i = 0; i < dim; ++i) {
            // pix[i] is the global coordinate (may be negative for ghosts)
            ptr[i] = muGrid::normalize_coord<T>(pix[i], nb_domain_grid_pts[i]);
        }
        ptr += dim;
    }

    return coords;
}

void add_fft_utils(py::module & mod) {
  // Note: fft_freqind, fft_freq, rfft_freqind, rfft_freq are now properties
  // on the FFTEngine class, not standalone functions.

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

      // Field registration - register_* versions throw if field exists
      .def("register_real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::register_real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Register a new real-space field.

           Raises an error if a field with the given name already exists.

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

           Raises
           ------
           RuntimeError
               If a field with the given name already exists
           )")

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::register_fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Register a new Fourier-space field.

           Raises an error if a field with the given name already exists.

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

           Raises
           ------
           RuntimeError
               If a field with the given name already exists
           )")

      // Field access - returns existing field if present
      .def("real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Get or create a real-space field.

           If a field with the given name already exists, returns it.
           Otherwise creates a new field with the specified number of components.

           Parameters
           ----------
           name : str
               Unique field name
           nb_components : int, optional
               Number of components (default 1)

           Returns
           -------
           Field
               Reference to the field
           )")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngine::fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           R"(
           Get or create a Fourier-space field.

           If a field with the given name already exists, returns it.
           Otherwise creates a new field with the specified number of components.

           Parameters
           ----------
           name : str
               Unique field name
           nb_components : int, optional
               Number of components (default 1)

           Returns
           -------
           Field
               Reference to the field
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
           [](const PyFFTEngine & self) {
             return muGrid::to_tuple(self.get_nb_fourier_grid_pts());
           },
           R"(
           Get the global Fourier grid dimensions.
           For r2c transform: (Nx/2+1, Ny, Nz)
           )")

      .def_property_readonly("nb_fourier_subdomain_grid_pts",
           [](const PyFFTEngine & self) {
             return muGrid::to_tuple(self.get_nb_fourier_subdomain_grid_pts());
           },
           "Get the local Fourier grid dimensions on this rank.")

      .def_property_readonly("fourier_subdomain_locations",
           [](const PyFFTEngine & self) {
             return muGrid::to_tuple(self.get_fourier_subdomain_locations());
           },
           "Get the starting location of this rank's Fourier subdomain.")

      .def_property_readonly("nb_subdomain_grid_pts",
           [](const PyFFTEngine & self) {
             return muGrid::to_tuple(
                 self.get_nb_subdomain_grid_pts_without_ghosts());
           },
           "Get the local real-space grid dimensions on this rank (excluding ghosts).")

      .def_property_readonly("subdomain_locations",
           [](const PyFFTEngine & self) {
             return muGrid::to_tuple(
                 self.get_subdomain_locations_without_ghosts());
           },
           "Get the starting location of this rank's real-space subdomain.")

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
           "Get the name of the FFT backend being used.")

      .def_property_readonly("spatial_dim",
           &PyFFTEngine::get_spatial_dim,
           "Get the spatial dimension (2 or 3).")

      // FFT frequency arrays
      .def_property_readonly("fftfreq",
           [](const PyFFTEngine & self) {
             return muGrid::py_fftfreq<Real, PyFFTEngine>(self);
           },
           R"(
           Get array of normalized FFT frequencies for the local Fourier subdomain.

           Returns an array of shape [dim, local_fx, local_fy, ...] where each
           element is the normalized frequency in range [-0.5, 0.5).

           For MPI parallel runs, this returns only the frequencies for the
           local subdomain owned by this rank.

           Returns
           -------
           ndarray
               Array of fractional frequencies
           )")

      .def_property_readonly("ifftfreq",
           [](const PyFFTEngine & self) {
             return muGrid::py_fftfreq<Int, PyFFTEngine>(self);
           },
           R"(
           Get array of integer FFT frequency indices for the local Fourier subdomain.

           Returns an array of shape [dim, local_fx, local_fy, ...] where each
           element is the integer frequency index.

           For MPI parallel runs, this returns only the frequencies for the
           local subdomain owned by this rank.

           Returns
           -------
           ndarray
               Array of integer frequency indices
           )")

      // Real-space coordinate arrays
      .def_property_readonly("coords",
           [](const PyFFTEngine & self) {
             return py_fft_coords<Real, false>(self);
           },
           R"(
           Get array of normalized coordinates for the local real-space subdomain.

           Returns an array of shape [dim, local_nx, local_ny, ...] where each
           element is the normalized coordinate in range [0, 1).

           Ghost cells are excluded.

           Returns
           -------
           ndarray
               Array of fractional coordinates
           )")

      .def_property_readonly("icoords",
           [](const PyFFTEngine & self) {
             return py_fft_coords<Int, false>(self);
           },
           R"(
           Get array of integer coordinate indices for the local real-space subdomain.

           Returns an array of shape [dim, local_nx, local_ny, ...] where each
           element is the integer coordinate index.

           Ghost cells are excluded.

           Returns
           -------
           ndarray
               Array of integer coordinate indices
           )")

      .def_property_readonly("coordsg",
           [](const PyFFTEngine & self) {
             return py_fft_coords<Real, true>(self);
           },
           R"(
           Get array of normalized coordinates including ghost cells.

           Returns an array of shape [dim, local_nx+ghosts, local_ny+ghosts, ...]
           where each element is the normalized coordinate in range [0, 1).

           Returns
           -------
           ndarray
               Array of fractional coordinates including ghosts
           )")

      .def_property_readonly("icoordsg",
           [](const PyFFTEngine & self) {
             return py_fft_coords<Int, true>(self);
           },
           R"(
           Get array of integer coordinate indices including ghost cells.

           Returns an array of shape [dim, local_nx+ghosts, local_ny+ghosts, ...]
           where each element is the integer coordinate index.

           Returns
           -------
           ndarray
               Array of integer coordinate indices including ghosts
           )");
}

void add_fft_classes(py::module & mod) {
  add_fft_utils(mod);
  add_fft_engine(mod);
}
