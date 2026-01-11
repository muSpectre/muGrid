/**
 * @file   bind_py_fft.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2025
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

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

using muGrid::FFTEngineBase;
using muGrid::FFTEngine;
using muGrid::HostSpace;
using muGrid::Communicator;
using muGrid::Device;

// Alias for Python binding - use HostSpace specialization
using PyFFTEngine = FFTEngine<HostSpace>;

#if defined(MUGRID_ENABLE_CUDA)
using muGrid::CUDASpace;
using PyFFTEngineCUDA = FFTEngine<CUDASpace>;
#endif

#if defined(MUGRID_ENABLE_HIP)
using muGrid::ROCmSpace;
using PyFFTEngineROCm = FFTEngine<ROCmSpace>;
#endif

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
 * @tparam EngineType The FFT engine type
 * @param eng The FFT engine
 * @return numpy array of coordinates
 */
template<typename T, bool with_ghosts, typename EngineType>
auto py_fft_coords(const EngineType & eng) {
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

      For a real-space grid of size (Nx, Ny, Nz), the half-complex Fourier grid
      has size (Nx/2+1, Ny, Nz) (for r2c_axis=0).

      Parameters
      ----------
      nb_grid_pts : tuple of int
          Real-space grid dimensions
      r2c_axis : int, optional
          Axis along which r2c transform is performed (default 0)

      Returns
      -------
      tuple of int
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
      nb_grid_pts : tuple of int
          Grid dimensions

      Returns
      -------
      float
          Normalization factor (1.0 / total_grid_points)
      )");
}

void add_fft_engine(py::module & mod) {
  using SubPtMap_t = muGrid::FieldCollection::SubPtMap_t;

  py::class_<PyFFTEngine, muGrid::CartesianDecomposition>(mod, "FFTEngine",
      R"(
      Distributed FFT engine using pencil (2D) decomposition.

      This class provides distributed FFT operations on structured grids with
      MPI parallelization. It uses pencil decomposition which allows efficient
      scaling to large numbers of ranks.

      Key features:
      - Supports 1D, 2D, and 3D grids
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
           nb_domain_grid_pts : tuple of int
               Global grid dimensions (Nx,), (Nx, Ny), or (Nx, Ny, Nz)
           comm : Communicator, optional
               MPI communicator (default: serial)
           nb_ghosts_left : tuple of int, optional
               Ghost cells on low-index side of each dimension
           nb_ghosts_right : tuple of int, optional
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

      .def("register_real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngine::register_real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           R"(
           Register a new real-space field with component shape.

           Raises an error if a field with the given name already exists.

           Parameters
           ----------
           name : str
               Unique field name
           components : tuple of int, optional
               Shape of field components. Default is () for scalar.

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

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngine::register_fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           R"(
           Register a new Fourier-space field with component shape.

           Raises an error if a field with the given name already exists.

           Parameters
           ----------
           name : str
               Unique field name
           components : tuple of int, optional
               Shape of field components. Default is () for scalar.

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

      .def("real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngine::real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           R"(
           Get or create a real-space field with component shape.

           If a field with the given name already exists, returns it.
           Otherwise creates a new field with the specified component shape.

           Parameters
           ----------
           name : str
               Unique field name
           components : tuple of int, optional
               Shape of field components. Default is () for scalar.

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

      .def("fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngine::fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           R"(
           Get or create a Fourier-space field with component shape.

           If a field with the given name already exists, returns it.
           Otherwise creates a new field with the specified component shape.

           Parameters
           ----------
           name : str
               Unique field name
           components : tuple of int, optional
               Shape of field components. Default is () for scalar.

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
             return py_fft_coords<Real, false, PyFFTEngine>(self);
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
             return py_fft_coords<Int, false, PyFFTEngine>(self);
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
             return py_fft_coords<Real, true, PyFFTEngine>(self);
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
             return py_fft_coords<Int, true, PyFFTEngine>(self);
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

#if defined(MUGRID_ENABLE_CUDA)
void add_fft_engine_cuda(py::module & mod) {
  using SubPtMap_t = muGrid::FieldCollection::SubPtMap_t;

  py::class_<PyFFTEngineCUDA, muGrid::CartesianDecomposition>(mod, "FFTEngineCUDA",
      R"(
      GPU-accelerated distributed FFT engine using cuFFT.

      This is the CUDA variant of FFTEngine that performs FFT operations on GPU memory.
      Fields created by this engine are allocated on the specified CUDA device.

      See FFTEngine for full documentation of the interface.
      )")
      .def(py::init([](const DynGridIndex & nb_domain_grid_pts,
                       const Communicator & comm,
                       const DynGridIndex & nb_ghosts_left,
                       const DynGridIndex & nb_ghosts_right,
                       const SubPtMap_t & nb_sub_pts,
                       int device_id) {
             // Set the CUDA device before creating the engine
             cudaSetDevice(device_id);
             // Create device object with the specified device_id
             Device device = Device::cuda(device_id);
             return new PyFFTEngineCUDA(nb_domain_grid_pts, comm,
                                        nb_ghosts_left, nb_ghosts_right,
                                        nb_sub_pts, device);
           }),
           "nb_domain_grid_pts"_a,
           "comm"_a = Communicator(),
           "nb_ghosts_left"_a = DynGridIndex{},
           "nb_ghosts_right"_a = DynGridIndex{},
           "nb_sub_pts"_a = SubPtMap_t{},
           "device_id"_a = 0,
           R"(
           Construct a GPU FFT engine.

           Parameters
           ----------
           nb_domain_grid_pts : tuple of int
               Global grid dimensions (Nx,), (Nx, Ny), or (Nx, Ny, Nz)
           comm : Communicator, optional
               MPI communicator (default: serial)
           nb_ghosts_left : tuple of int, optional
               Ghost cells on low-index side
           nb_ghosts_right : tuple of int, optional
               Ghost cells on high-index side
           nb_sub_pts : dict, optional
               Number of sub-points per pixel
           device_id : int, optional
               CUDA device ID (default: 0)
           )")

      .def("fft",
           [](PyFFTEngineCUDA & self, const Field & input, Field & output) {
             self.fft(input, output);
           },
           "input"_a, "output"_a,
           "Forward FFT: real space -> Fourier space (GPU)")

      .def("ifft",
           [](PyFFTEngineCUDA & self, const Field & input, Field & output) {
             self.ifft(input, output);
           },
           "input"_a, "output"_a,
           "Inverse FFT: Fourier space -> real space (GPU)")

      .def("register_real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineCUDA::register_real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Register a new real-space field on GPU")

      .def("register_real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineCUDA::register_real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Register a new real-space field on GPU with component shape")

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineCUDA::register_fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Register a new Fourier-space field on GPU")

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineCUDA::register_fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Register a new Fourier-space field on GPU with component shape")

      .def("real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineCUDA::real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Get or create a real-space field on GPU")

      .def("real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineCUDA::real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Get or create a real-space field on GPU with component shape")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineCUDA::fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Get or create a Fourier-space field on GPU")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineCUDA::fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Get or create a Fourier-space field on GPU with component shape")

      .def_property_readonly("real_space_collection",
           py::overload_cast<>(&PyFFTEngineCUDA::get_real_space_collection),
           py::return_value_policy::reference_internal)

      .def_property_readonly("fourier_space_collection",
           py::overload_cast<>(&PyFFTEngineCUDA::get_fourier_space_collection),
           py::return_value_policy::reference_internal)

      .def_property_readonly("normalisation", &PyFFTEngineCUDA::normalisation)

      .def_property_readonly("nb_fourier_grid_pts",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::to_tuple(self.get_nb_fourier_grid_pts());
           })

      .def_property_readonly("nb_fourier_subdomain_grid_pts",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::to_tuple(self.get_nb_fourier_subdomain_grid_pts());
           })

      .def_property_readonly("fourier_subdomain_locations",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::to_tuple(self.get_fourier_subdomain_locations());
           })

      .def_property_readonly("nb_subdomain_grid_pts",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::to_tuple(
                 self.get_nb_subdomain_grid_pts_without_ghosts());
           })

      .def_property_readonly("subdomain_locations",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::to_tuple(
                 self.get_subdomain_locations_without_ghosts());
           })

      .def_property_readonly("process_grid",
           [](const PyFFTEngineCUDA & self) {
             auto grid = self.get_process_grid();
             return py::make_tuple(grid[0], grid[1]);
           })

      .def_property_readonly("process_coords",
           [](const PyFFTEngineCUDA & self) {
             auto coords = self.get_process_coords();
             return py::make_tuple(coords[0], coords[1]);
           })

      .def_property_readonly("backend_name", &PyFFTEngineCUDA::get_backend_name)

      .def_property_readonly("spatial_dim", &PyFFTEngineCUDA::get_spatial_dim)

      .def_property_readonly("fftfreq",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::py_fftfreq<Real, PyFFTEngineCUDA>(self);
           })

      .def_property_readonly("ifftfreq",
           [](const PyFFTEngineCUDA & self) {
             return muGrid::py_fftfreq<Int, PyFFTEngineCUDA>(self);
           })

      .def_property_readonly("coords",
           [](const PyFFTEngineCUDA & self) {
             return py_fft_coords<Real, false, PyFFTEngineCUDA>(self);
           })

      .def_property_readonly("icoords",
           [](const PyFFTEngineCUDA & self) {
             return py_fft_coords<Int, false, PyFFTEngineCUDA>(self);
           })

      .def_property_readonly("coordsg",
           [](const PyFFTEngineCUDA & self) {
             return py_fft_coords<Real, true, PyFFTEngineCUDA>(self);
           })

      .def_property_readonly("icoordsg",
           [](const PyFFTEngineCUDA & self) {
             return py_fft_coords<Int, true, PyFFTEngineCUDA>(self);
           });
}
#endif  // MUGRID_ENABLE_CUDA

#if defined(MUGRID_ENABLE_HIP)
void add_fft_engine_hip(py::module & mod) {
  using SubPtMap_t = muGrid::FieldCollection::SubPtMap_t;

  py::class_<PyFFTEngineROCm, muGrid::CartesianDecomposition>(mod, "FFTEngineROCm",
      R"(
      GPU-accelerated distributed FFT engine using rocFFT.

      This is the ROCm variant of FFTEngine that performs FFT operations on GPU memory.
      Fields created by this engine are allocated on the specified AMD GPU device.

      See FFTEngine for full documentation of the interface.
      )")
      .def(py::init([](const DynGridIndex & nb_domain_grid_pts,
                       const Communicator & comm,
                       const DynGridIndex & nb_ghosts_left,
                       const DynGridIndex & nb_ghosts_right,
                       const SubPtMap_t & nb_sub_pts,
                       int device_id) {
             // Set the ROCm device before creating the engine
             hipSetDevice(device_id);
             // Create device object with the specified device_id
             Device device = Device::rocm(device_id);
             return new PyFFTEngineROCm(nb_domain_grid_pts, comm,
                                       nb_ghosts_left, nb_ghosts_right,
                                       nb_sub_pts, device);
           }),
           "nb_domain_grid_pts"_a,
           "comm"_a = Communicator(),
           "nb_ghosts_left"_a = DynGridIndex{},
           "nb_ghosts_right"_a = DynGridIndex{},
           "nb_sub_pts"_a = SubPtMap_t{},
           "device_id"_a = 0,
           R"(
           Construct a GPU FFT engine using ROCm.

           Parameters
           ----------
           nb_domain_grid_pts : tuple of int
               Global grid dimensions (Nx,), (Nx, Ny), or (Nx, Ny, Nz)
           comm : Communicator, optional
               MPI communicator (default: serial)
           nb_ghosts_left : tuple of int, optional
               Ghost cells on low-index side
           nb_ghosts_right : tuple of int, optional
               Ghost cells on high-index side
           nb_sub_pts : dict, optional
               Number of sub-points per pixel
           device_id : int, optional
               ROCm device ID (default: 0)
           )")

      .def("fft",
           [](PyFFTEngineROCm & self, const Field & input, Field & output) {
             self.fft(input, output);
           },
           "input"_a, "output"_a,
           "Forward FFT: real space -> Fourier space (GPU)")

      .def("ifft",
           [](PyFFTEngineROCm & self, const Field & input, Field & output) {
             self.ifft(input, output);
           },
           "input"_a, "output"_a,
           "Inverse FFT: Fourier space -> real space (GPU)")

      .def("register_real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineROCm::register_real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Register a new real-space field on GPU")

      .def("register_real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineROCm::register_real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Register a new real-space field on GPU with component shape")

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineROCm::register_fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Register a new Fourier-space field on GPU")

      .def("register_fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineROCm::register_fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Register a new Fourier-space field on GPU with component shape")

      .def("real_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineROCm::real_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Get or create a real-space field on GPU")

      .def("real_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineROCm::real_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Get or create a real-space field on GPU with component shape")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, Index_t>(
               &PyFFTEngineROCm::fourier_space_field),
           "name"_a, "nb_components"_a = 1,
           py::return_value_policy::reference_internal,
           "Get or create a Fourier-space field on GPU")

      .def("fourier_space_field",
           py::overload_cast<const std::string &, const muGrid::Shape_t &>(
               &PyFFTEngineROCm::fourier_space_field),
           "name"_a, "components"_a = muGrid::Shape_t{},
           py::return_value_policy::reference_internal,
           "Get or create a Fourier-space field on GPU with component shape")

      .def_property_readonly("real_space_collection",
           py::overload_cast<>(&PyFFTEngineROCm::get_real_space_collection),
           py::return_value_policy::reference_internal)

      .def_property_readonly("fourier_space_collection",
           py::overload_cast<>(&PyFFTEngineROCm::get_fourier_space_collection),
           py::return_value_policy::reference_internal)

      .def_property_readonly("normalisation", &PyFFTEngineROCm::normalisation)

      .def_property_readonly("nb_fourier_grid_pts",
           [](const PyFFTEngineROCm & self) {
             return muGrid::to_tuple(self.get_nb_fourier_grid_pts());
           })

      .def_property_readonly("nb_fourier_subdomain_grid_pts",
           [](const PyFFTEngineROCm & self) {
             return muGrid::to_tuple(self.get_nb_fourier_subdomain_grid_pts());
           })

      .def_property_readonly("fourier_subdomain_locations",
           [](const PyFFTEngineROCm & self) {
             return muGrid::to_tuple(self.get_fourier_subdomain_locations());
           })

      .def_property_readonly("nb_subdomain_grid_pts",
           [](const PyFFTEngineROCm & self) {
             return muGrid::to_tuple(
                 self.get_nb_subdomain_grid_pts_without_ghosts());
           })

      .def_property_readonly("subdomain_locations",
           [](const PyFFTEngineROCm & self) {
             return muGrid::to_tuple(
                 self.get_subdomain_locations_without_ghosts());
           })

      .def_property_readonly("process_grid",
           [](const PyFFTEngineROCm & self) {
             auto grid = self.get_process_grid();
             return py::make_tuple(grid[0], grid[1]);
           })

      .def_property_readonly("process_coords",
           [](const PyFFTEngineROCm & self) {
             auto coords = self.get_process_coords();
             return py::make_tuple(coords[0], coords[1]);
           })

      .def_property_readonly("backend_name", &PyFFTEngineROCm::get_backend_name)

      .def_property_readonly("spatial_dim", &PyFFTEngineROCm::get_spatial_dim)

      .def_property_readonly("fftfreq",
           [](const PyFFTEngineROCm & self) {
             return muGrid::py_fftfreq<Real, PyFFTEngineROCm>(self);
           })

      .def_property_readonly("ifftfreq",
           [](const PyFFTEngineROCm & self) {
             return muGrid::py_fftfreq<Int, PyFFTEngineROCm>(self);
           })

      .def_property_readonly("coords",
           [](const PyFFTEngineROCm & self) {
             return py_fft_coords<Real, false, PyFFTEngineROCm>(self);
           })

      .def_property_readonly("icoords",
           [](const PyFFTEngineROCm & self) {
             return py_fft_coords<Int, false, PyFFTEngineROCm>(self);
           })

      .def_property_readonly("coordsg",
           [](const PyFFTEngineROCm & self) {
             return py_fft_coords<Real, true, PyFFTEngineROCm>(self);
           })

      .def_property_readonly("icoordsg",
           [](const PyFFTEngineROCm & self) {
             return py_fft_coords<Int, true, PyFFTEngineROCm>(self);
           });
}
#endif  // MUGRID_ENABLE_HIP

void add_fft_classes(py::module & mod) {
  add_fft_utils(mod);
  add_fft_engine(mod);
#if defined(MUGRID_ENABLE_CUDA)
  add_fft_engine_cuda(mod);
#endif
#if defined(MUGRID_ENABLE_HIP)
  add_fft_engine_hip(mod);
#endif
}
