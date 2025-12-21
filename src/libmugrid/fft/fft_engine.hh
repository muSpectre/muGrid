/**
 * @file   fft/fft_engine.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Distributed FFT engine with pencil decomposition
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

#ifndef SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_
#define SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_

#include "mpi/cartesian_decomposition.hh"
#include "collection/field_collection_global.hh"
#include "mpi/communicator.hh"

#include "fft_1d_backend.hh"
#include "datatype_transpose.hh"
#include "fft_utils.hh"

#include <memory>
#include <array>
#include <string>
#include <map>

namespace muGrid {

/**
 * Distributed FFT engine using pencil (2D) decomposition.
 *
 * This class provides distributed FFT operations on structured grids with
 * MPI parallelization. It uses pencil decomposition which allows efficient
 * scaling to large numbers of ranks.
 *
 * Key features:
 * - Supports 2D and 3D grids
 * - Handles arbitrary ghost buffer configurations in real space
 * - No ghosts in Fourier space (hard assumption)
 * - Supports host and device (GPU) memory
 * - Unnormalized transforms (like FFTW)
 *
 * The engine owns field collections for both real and Fourier space, and
 * work buffers for intermediate results during the distributed FFT.
 *
 * For 2D grids:
 * - Uses 1D decomposition (P ranks)
 * - 1 transpose operation
 * - 2 work buffers
 *
 * For 3D grids:
 * - Uses 2D decomposition (P1 × P2 = P ranks)
 * - 3 transpose operations
 * - 3 work buffers
 */
class FFTEngine : public CartesianDecomposition {
 public:
  using Parent_t = CartesianDecomposition;
  using SubPtMap_t = FieldCollection::SubPtMap_t;
  using MemoryLocation = FieldCollection::MemoryLocation;

  /**
   * Construct an FFT engine with pencil decomposition.
   *
   * @param nb_domain_grid_pts  Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
   * @param comm                MPI communicator (default: serial)
   * @param nb_ghosts_left      Ghost cells on low-index side of each dimension
   * @param nb_ghosts_right     Ghost cells on high-index side of each dimension
   * @param nb_sub_pts          Number of sub-points per pixel (optional)
   * @param memory_location     Where to allocate field memory (Host or Device)
   */
  FFTEngine(const IntCoord_t & nb_domain_grid_pts,
            const Communicator & comm = Communicator(),
            const IntCoord_t & nb_ghosts_left = IntCoord_t{},
            const IntCoord_t & nb_ghosts_right = IntCoord_t{},
            const SubPtMap_t & nb_sub_pts = {},
            MemoryLocation memory_location = MemoryLocation::Host);

  FFTEngine() = delete;
  FFTEngine(const FFTEngine &) = delete;
  FFTEngine(FFTEngine &&) = delete;
  ~FFTEngine() override = default;

  FFTEngine & operator=(const FFTEngine &) = delete;
  FFTEngine & operator=(FFTEngine &&) = delete;

  // === Transform operations ===

  /**
   * Forward FFT: real space → Fourier space.
   *
   * The transform is unnormalized. To recover the original data after
   * ifft(fft(x)), multiply by normalisation().
   *
   * @param input   Real-space field (must be in this engine's real collection)
   * @param output  Fourier-space field (must be in this engine's Fourier
   * collection)
   */
  void fft(const Field & input, Field & output);

  /**
   * Inverse FFT: Fourier space → real space.
   *
   * The transform is unnormalized. To recover the original data after
   * ifft(fft(x)), multiply by normalisation().
   *
   * @param input   Fourier-space field (must be in this engine's Fourier
   * collection)
   * @param output  Real-space field (must be in this engine's real collection)
   */
  void ifft(const Field & input, Field & output);

  // === Field registration ===

  /**
   * Register a real-space scalar field.
   *
   * @param name        Unique field name
   * @param nb_components  Number of components (default 1)
   * @return Reference to the created field
   */
  Field & register_real_space_field(const std::string & name,
                                    Index_t nb_components = 1);

  /**
   * Register a real-space field with shaped components.
   *
   * @param name        Unique field name
   * @param components  Component shape (e.g., {3,3} for a tensor)
   * @return Reference to the created field
   */
  Field & register_real_space_field(const std::string & name,
                                    const Shape_t & components);

  /**
   * Register a Fourier-space field.
   *
   * @param name        Unique field name
   * @param nb_components  Number of components (default 1)
   * @return Reference to the created field
   */
  Field & register_fourier_space_field(const std::string & name,
                                       Index_t nb_components = 1);

  /**
   * Register a Fourier-space field with shaped components.
   *
   * @param name        Unique field name
   * @param components  Component shape (e.g., {3,3} for a tensor)
   * @return Reference to the created field
   */
  Field & register_fourier_space_field(const std::string & name,
                                       const Shape_t & components);

  // === Collection access ===

  /**
   * Get the real-space field collection.
   * This is the same as get_collection() from CartesianDecomposition.
   */
  GlobalFieldCollection & get_real_space_collection();
  const GlobalFieldCollection & get_real_space_collection() const;

  /**
   * Get the Fourier-space field collection.
   */
  GlobalFieldCollection & get_fourier_space_collection();
  const GlobalFieldCollection & get_fourier_space_collection() const;

  // === Geometry queries ===

  /**
   * Get the normalization factor for FFT roundtrip.
   * Multiply ifft output by this to recover original values.
   */
  Real normalisation() const { return this->norm_factor; }

  /**
   * Get the global Fourier grid dimensions.
   * For r2c transform: [Nx/2+1, Ny, Nz]
   */
  const IntCoord_t & get_nb_fourier_grid_pts() const {
    return this->nb_fourier_grid_pts;
  }

  /**
   * Get the local Fourier grid dimensions on this rank.
   */
  const IntCoord_t & get_nb_fourier_subdomain_grid_pts() const {
    return this->nb_fourier_subdomain_grid_pts;
  }

  /**
   * Get the starting location of this rank's Fourier subdomain.
   */
  const IntCoord_t & get_fourier_subdomain_locations() const {
    return this->fourier_subdomain_locations;
  }

  /**
   * Get the 2D process grid dimensions [P1, P2].
   * For 2D problems, P1=1.
   */
  std::array<int, 2> get_process_grid() const {
    return {this->proc_grid_p1, this->proc_grid_p2};
  }

  /**
   * Get this rank's coordinates in the process grid [p1, p2].
   */
  std::array<int, 2> get_process_coords() const {
    return {this->proc_coord_p1, this->proc_coord_p2};
  }

  /**
   * Get the name of the FFT backend being used.
   */
  const char * get_backend_name() const;

 protected:
  /**
   * Initialize the FFT engine after construction.
   * Sets up work buffers and transpose operations.
   */
  void initialise_fft();

  /**
   * Perform 2D forward FFT.
   */
  void fft_2d(const Field & input, Field & output);

  /**
   * Perform 3D forward FFT.
   */
  void fft_3d(const Field & input, Field & output);

  /**
   * Perform 2D inverse FFT.
   */
  void ifft_2d(const Field & input, Field & output);

  /**
   * Perform 3D inverse FFT.
   */
  void ifft_3d(const Field & input, Field & output);

  /**
   * Select the appropriate FFT backend based on field memory location.
   */
  FFT1DBackend * select_backend(bool is_device_memory) const;

  // === Process grid ===
  int proc_grid_p1{1};   //!< First dimension of process grid
  int proc_grid_p2{1};   //!< Second dimension of process grid
  int proc_coord_p1{0};  //!< This rank's p1 coordinate
  int proc_coord_p2{0};  //!< This rank's p2 coordinate

  // === Subcommunicators ===
#ifdef WITH_MPI
  Communicator row_comm;  //!< Communicator for ranks with same p1 (3D only)
  Communicator col_comm;  //!< Communicator for ranks with same p2
#endif

  // === FFT backends ===
  std::unique_ptr<FFT1DBackend> host_backend;    //!< Host (CPU) FFT backend
  std::unique_ptr<FFT1DBackend> device_backend;  //!< Device (GPU) FFT backend

  // === Transpose operations ===
  /**
   * Configuration for creating transposes with different nb_components.
   */
  struct TransposeConfig {
    IntCoord_t local_in;
    IntCoord_t local_out;
    Index_t global_in;
    Index_t global_out;
    Index_t axis_in;
    Index_t axis_out;
    bool use_row_comm;  //!< true = row_comm, false = col_comm
  };

  /**
   * Get or create a transpose for the given nb_components.
   */
  DatatypeTranspose * get_transpose_xz(Index_t nb_components);
  DatatypeTranspose * get_transpose_yz_forward(Index_t nb_components);
  DatatypeTranspose * get_transpose_yz_backward(Index_t nb_components);

  //! Configuration for X↔Z transpose (2D: Y↔X)
  TransposeConfig transpose_xz_config;

  //! Configuration for Y↔Z forward transpose (3D only)
  TransposeConfig transpose_yz_fwd_config;

  //! Configuration for Y↔Z backward transpose (3D only)
  TransposeConfig transpose_yz_bwd_config;

  //! Cached transposes by nb_components
  std::map<Index_t, std::unique_ptr<DatatypeTranspose>> transpose_xz_cache;
  std::map<Index_t, std::unique_ptr<DatatypeTranspose>> transpose_yz_fwd_cache;
  std::map<Index_t, std::unique_ptr<DatatypeTranspose>> transpose_yz_bwd_cache;

  //! Flag indicating if transpose is needed (comm.size() > 1)
  bool need_transpose_xz{false};
  bool need_transpose_yz{false};

  // === Work buffers ===
  //! Fourier-space field collection (final X-pencil layout)
  std::unique_ptr<GlobalFieldCollection> fourier_collection;

  //! Work buffer after first FFT (Z-pencils, no ghosts)
  std::unique_ptr<GlobalFieldCollection> work_zpencil;

  //! Work buffer after Y-FFT (3D only, Y-pencils)
  std::unique_ptr<GlobalFieldCollection> work_ypencil;

  // === Geometry ===
  IntCoord_t nb_fourier_grid_pts;           //!< Global Fourier grid size
  IntCoord_t nb_fourier_subdomain_grid_pts; //!< Local Fourier grid size
  IntCoord_t fourier_subdomain_locations;   //!< Fourier subdomain start
  Real norm_factor;                          //!< 1 / (Nx * Ny * Nz)
  Dim_t spatial_dim;                         //!< 2 or 3
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_
