/**
 * @file   fft/fft_engine_base.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  Non-templated base class for distributed FFT engine
 *
 * Copyright (c) 2024 Lars Pastewka
 *
 * muGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * muGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with muGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_
#define SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_

#include "mpi/cartesian_decomposition.hh"
#include "collection/field_collection_global.hh"
#include "mpi/communicator.hh"
#include "fft_1d_backend.hh"
#include "transpose.hh"
#include "fft_utils.hh"

#include <memory>
#include <array>
#include <string>
#include <map>

namespace muGrid {

/**
 * Non-templated base class for FFTEngine.
 *
 * This class contains all the domain decomposition logic, MPI setup,
 * transpose configurations, and field collection management. It is
 * inherited by the templated FFTEngine<MemorySpace> which adds the
 * memory-space-specific FFT execution logic.
 */
class FFTEngineBase : public CartesianDecomposition {
 public:
  using Parent_t = CartesianDecomposition;
  using SubPtMap_t = FieldCollection::SubPtMap_t;

  /**
   * Construct an FFT engine base with pencil decomposition.
   *
   * @param nb_domain_grid_pts  Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
   * @param comm                MPI communicator (default: serial)
   * @param nb_ghosts_left      Ghost cells on low-index side of each dimension
   * @param nb_ghosts_right     Ghost cells on high-index side of each dimension
   * @param nb_sub_pts          Number of sub-points per pixel (optional)
   * @param device              Where to allocate field memory
   */
  FFTEngineBase(const DynGridIndex & nb_domain_grid_pts,
                const Communicator & comm = Communicator(),
                const DynGridIndex & nb_ghosts_left = DynGridIndex{},
                const DynGridIndex & nb_ghosts_right = DynGridIndex{},
                const SubPtMap_t & nb_sub_pts = {},
                Device device = Device::cpu());

  FFTEngineBase() = delete;
  FFTEngineBase(const FFTEngineBase &) = delete;
  FFTEngineBase(FFTEngineBase &&) = delete;
  ~FFTEngineBase() override = default;

  FFTEngineBase & operator=(const FFTEngineBase &) = delete;
  FFTEngineBase & operator=(FFTEngineBase &&) = delete;

  // === Transform operations (virtual, implemented by derived class) ===

  /**
   * Forward FFT: real space -> Fourier space.
   */
  virtual void fft(const Field & input, Field & output) = 0;

  /**
   * Inverse FFT: Fourier space -> real space.
   */
  virtual void ifft(const Field & input, Field & output) = 0;

  // === Field registration ===

  /**
   * Register a new real-space field. Throws if field already exists.
   */
  Field & register_real_space_field(const std::string & name,
                                    Index_t nb_components = 1);

  Field & register_real_space_field(const std::string & name,
                                    const Shape_t & components);

  /**
   * Register a new Fourier-space field. Throws if field already exists.
   */
  Field & register_fourier_space_field(const std::string & name,
                                       Index_t nb_components = 1);

  Field & register_fourier_space_field(const std::string & name,
                                       const Shape_t & components);

  /**
   * Get or create a real-space field. Returns existing field if present.
   */
  Field & real_space_field(const std::string & name,
                           Index_t nb_components = 1);

  Field & real_space_field(const std::string & name,
                           const Shape_t & components);

  /**
   * Get or create a Fourier-space field. Returns existing field if present.
   */
  Field & fourier_space_field(const std::string & name,
                              Index_t nb_components = 1);

  Field & fourier_space_field(const std::string & name,
                              const Shape_t & components);

  // === Collection access ===

  GlobalFieldCollection & get_real_space_collection();
  const GlobalFieldCollection & get_real_space_collection() const;

  GlobalFieldCollection & get_fourier_space_collection();
  const GlobalFieldCollection & get_fourier_space_collection() const;

  // === Geometry queries ===

  Real normalisation() const { return this->norm_factor; }

  const DynGridIndex & get_nb_fourier_grid_pts() const {
    return this->nb_fourier_grid_pts;
  }

  const DynGridIndex & get_nb_fourier_subdomain_grid_pts() const {
    return this->nb_fourier_subdomain_grid_pts;
  }

  const DynGridIndex & get_fourier_subdomain_locations() const {
    return this->fourier_subdomain_locations;
  }

  std::array<int, 2> get_process_grid() const {
    return {this->proc_grid_p1, this->proc_grid_p2};
  }

  std::array<int, 2> get_process_coords() const {
    return {this->proc_coord_p1, this->proc_coord_p2};
  }

  /**
   * Get the name of the FFT backend being used.
   */
  virtual const char * get_backend_name() const = 0;

 protected:
  /**
   * Initialize the FFT infrastructure after construction.
   * Sets up work buffer collections and transpose configurations.
   */
  void initialise_fft_base();

  /**
   * Configuration for creating transposes with different nb_components.
   */
  struct TransposeConfig {
    DynGridIndex local_in;
    DynGridIndex local_out;
    Index_t global_in;
    Index_t global_out;
    Index_t axis_in;
    Index_t axis_out;
    bool use_row_comm;  //!< true = row_comm, false = col_comm
  };

  /**
   * Get or create a transpose for the given nb_components.
   */
  Transpose * get_transpose_xz(Index_t nb_components);
  Transpose * get_transpose_yz_forward(Index_t nb_components);
  Transpose * get_transpose_yz_backward(Index_t nb_components);

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

  // === Transpose operations ===
  TransposeConfig transpose_xz_config;
  TransposeConfig transpose_yz_fwd_config;
  TransposeConfig transpose_yz_bwd_config;

  std::map<Index_t, std::unique_ptr<Transpose>> transpose_xz_cache;
  std::map<Index_t, std::unique_ptr<Transpose>> transpose_yz_fwd_cache;
  std::map<Index_t, std::unique_ptr<Transpose>> transpose_yz_bwd_cache;

  bool need_transpose_xz{false};
  bool need_transpose_yz{false};

  // === Work buffer collections ===
  //! Fourier-space field collection (final X-pencil layout)
  std::unique_ptr<GlobalFieldCollection> fourier_collection;

  //! Work buffer collection for Z-pencils
  std::unique_ptr<GlobalFieldCollection> work_zpencil;

  //! Work buffer collection for Y-pencils (3D only)
  std::unique_ptr<GlobalFieldCollection> work_ypencil;

  // === Geometry ===
  DynGridIndex nb_fourier_grid_pts;
  DynGridIndex nb_fourier_subdomain_grid_pts;
  DynGridIndex fourier_subdomain_locations;
  Real norm_factor;
  Dim_t spatial_dim;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_ENGINE_BASE_HH_
