/**
 * @file   pfft_engine.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   06 Mar 2017
 *
 * @brief  implements the MPI-parallel pfft engine
 *
 * Copyright © 2017 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/exception.hh>

#include "pfft_engine.hh"

using RuntimeError;

namespace muFFT {

  template <Dim_t DimsS>
  int PFFTEngine<DimsS>::nb_engines{0};

  template <Dim_t DimS>
  PFFTEngine<DimS>::PFFTEngine(Ccoord nb_grid_pts, Dim_t nb_dof_per_pixel,
                               Communicator comm)
      : Parent{nb_grid_pts, nb_dof_per_pixel, comm},
        mpi_comm{comm.get_mpi_comm()}, plan_fft{nullptr},
        plan_ifft{nullptr}, real_workspace{nullptr} {
    if (!this->nb_engines)
      pfft_init();
    this->nb_engines++;

    int size{comm.size()};
    int dim_x{size};
    int dim_y{1};
    // Note: All TODOs below don't affect the function of the PFFT engine. It
    // presently uses slab decompositions, the TODOs are what needs to be done
    // to get stripe decomposition to work - but it does not work yet. Slab
    // vs stripe decomposition may have an impact on how the code scales.
    // TODO(pastewka): Enable this to enable 2d process mesh. This does not pass
    // tests.
    // if (DimS > 2) {
    if (false) {
      dim_y = static_cast<int>(sqrt(size));
      while ((size / dim_y) * dim_y != size)
        dim_y--;
      dim_x = size / dim_y;
    }

    // TODO(pastewka): Enable this to enable 2d process mesh. This does not pass
    // tests.  if (DimS > 2) {
    if (false) {
      if (pfft_create_procmesh_2d(this->comm.get_mpi_comm(), dim_x, dim_y,
                                  &this->mpi_comm)) {
        throw RuntimeError("Failed to create 2d PFFT process mesh.");
      }
    }

    std::array<ptrdiff_t, DimS> narr;
    for (Dim_t i = 0; i < DimS; ++i) {
      narr[i] = this->nb_domain_grid_pts[DimS - 1 - i];
    }
    ptrdiff_t res[DimS], loc[DimS], fres[DimS], floc[DimS];
    this->workspace_size = pfft_local_size_many_dft_r2c(
        DimS, narr.data(), narr.data(), narr.data(), this->nb_dof_per_pixel,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, this->mpi_comm,
        PFFT_TRANSPOSED_OUT, res, loc, fres, floc);
    for (Dim_t i = 0; i < DimS; ++i) {
      this->nb_subdomain_grid_pts[DimS-1-i] = res[i];
      this->subdomain_locations[DimS-1-i] = loc[i];
      this->nb_fourier_grid_pts[DimS-1-i] = fres[i];
      this->fourier_locations[DimS-1-i] = floc[i];
    }
    // TODO(pastewka): Enable this to enable 2d process mesh. This does not pass
    // tests.  for (int i = 0; i < DimS-1; ++i) {
    if (DimS > 1) {
      std::swap(this->nb_fourier_grid_pts[DimS - 2],
                this->nb_fourier_grid_pts[DimS - 1]);
      std::swap(this->fourier_locations[DimS - 2],
                this->fourier_locations[DimS - 1]);
    }

    for (auto & n : this->nb_subdomain_grid_pts) {
      if (n == 0) {
        throw RuntimeError("PFFT planning returned zero grid points. "
                                 "You may need to run on fewer processes.");
      }
    }
    for (auto & n : this->nb_fourier_grid_pts) {
      if (n == 0) {
        throw RuntimeError("PFFT planning returned zero Fourier "
                                 "grid_points. You may need to run on fewer "
                                 "processes.");
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void PFFTEngine<DimS>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw RuntimeError("double initialisation, will leak memory");
    }

    /*
     * Initialize parent after the local number of grid points in each direction
     * have been determined and work space has been initialized
     */
    Parent::initialise(plan_flags);

    this->real_workspace = pfft_alloc_real(2 * this->workspace_size);
    /*
     * We need to check whether the workspace provided by our field is large
     * enough. PFFT may request a workspace size larger than the nominal size of
     * the complex buffer.
     */
    if (static_cast<int>(this->work.size() * this->nb_dof_per_pixel) <
        this->workspace_size) {
      this->work.set_pad_size(this->workspace_size -
                              this->nb_dof_per_pixel * this->work.size());
    }

    unsigned int flags;
    switch (plan_flags) {
    case FFT_PlanFlags::estimate: {
      flags = PFFT_ESTIMATE;
      break;
    }
    case FFT_PlanFlags::measure: {
      flags = PFFT_MEASURE;
      break;
    }
    case FFT_PlanFlags::patient: {
      flags = PFFT_PATIENT;
      break;
    }
    default:
      throw RuntimeError("unknown planner flag type");
      break;
    }

    std::array<ptrdiff_t, DimS> narr;
    for (Dim_t i = 0; i < DimS; ++i) {
      narr[i] = this->nb_domain_grid_pts[DimS - 1 - i];
    }
    Real * in{this->real_workspace};
    pfft_complex * out{reinterpret_cast<pfft_complex *>(this->work.data())};
    this->plan_fft = pfft_plan_many_dft_r2c(
        DimS, narr.data(), narr.data(), narr.data(), this->nb_dof_per_pixel,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, in, out, this->mpi_comm,
        PFFT_FORWARD, PFFT_TRANSPOSED_OUT | flags);
    if (this->plan_fft == nullptr) {
      throw RuntimeError("r2c plan failed");
    }

    pfft_complex * i_in{reinterpret_cast<pfft_complex *>(this->work.data())};
    Real * i_out{this->real_workspace};

    this->plan_ifft = pfft_plan_many_dft_c2r(
        DimS, narr.data(), narr.data(), narr.data(), this->nb_dof_per_pixel,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, i_in, i_out, this->mpi_comm,
        PFFT_BACKWARD, PFFT_TRANSPOSED_IN | flags);
    if (this->plan_ifft == nullptr) {
      throw RuntimeError("c2r plan failed");
    }
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  PFFTEngine<DimS>::~PFFTEngine<DimS>() noexcept {
    if (this->real_workspace != nullptr)
      pfft_free(this->real_workspace);
    if (this->plan_fft != nullptr)
      pfft_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr)
      pfft_destroy_plan(this->plan_ifft);
    if (this->mpi_comm != this->comm.get_mpi_comm()) {
      MPI_Comm_free(&this->mpi_comm);
    }
    // TODO(Till): We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    // this->nb_engines--;
    // if (!this->nb_engines) pfft_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  typename PFFTEngine<DimS>::Workspace_t &
  PFFTEngine<DimS>::fft(Field_t & field) {
    if (!this->plan_fft) {
      throw RuntimeError("fft plan not allocated");
    }
    if (field.size() !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The size of the field passed to the forward FFT is "
            << field.size() << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by PFFTEngine.";
      throw RuntimeError(error.str());
    }
    // Copy field data to workspace buffer. This is necessary because workspace
    // buffer is larger than field buffer.
    std::copy(field.data(),
              field.data() + this->nb_dof_per_pixel * field.size(),
              this->real_workspace);
    pfft_execute_dft_r2c(this->plan_fft, this->real_workspace,
                         reinterpret_cast<pfft_complex *>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void PFFTEngine<DimS>::ifft(Field_t & field) const {
    if (!this->plan_ifft) {
      throw RuntimeError("ifft plan not allocated");
    }
    if (field.size() !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The size of the field passed to the inverse FFT is "
            << field.size() << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by PFFTEngine.";
      throw RuntimeError(error.str());
    }
    pfft_execute_dft_c2r(this->plan_ifft,
                         reinterpret_cast<pfft_complex *>(this->work.data()),
                         this->real_workspace);
    std::copy(this->real_workspace,
              this->real_workspace + this->nb_dof_per_pixel * field.size(),
              field.data());
  }

  template class PFFTEngine<oneD>;
  template class PFFTEngine<twoD>;
  template class PFFTEngine<threeD>;
}  // namespace muFFT
