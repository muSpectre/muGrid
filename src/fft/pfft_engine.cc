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

#include "common/ccoord_operations.hh"
#include "fft/pfft_engine.hh"

namespace muSpectre {

  template <Dim_t DimsS, Dim_t DimM>
  int PFFTEngine<DimsS, DimM>::nb_engines{0};

  template <Dim_t DimS, Dim_t DimM>
  PFFTEngine<DimS, DimM>::PFFTEngine(Ccoord resolutions, Communicator comm)
    :Parent{resolutions, comm}, mpi_comm{comm.get_mpi_comm()}
  {
    if (!this->nb_engines) pfft_init();
    this->nb_engines++;

    int size{comm.size()};
    int dim_x{size};
    int dim_y{1};
    // Note: All TODOs below don't affect the function of the PFFT engine. It
    // presently uses slab decompositions, the TODOs are what needs to be done
    // to get stripe decomposition to work - but it does not work yet. Slab
    // vs stripe decomposition may have an impact on how the code scales.
    // TODO: Enable this to enable 2d process mesh. This does not pass tests.
    //if (DimS > 2) {
    if (false) {
      dim_y = int(sqrt(size));
      while ((size/dim_y)*dim_y != size)  dim_y--;
      dim_x = size/dim_y;
    }

    // TODO: Enable this to enable 2d process mesh. This does not pass tests.
    //if (DimS > 2) {
    if (false) {

      if (pfft_create_procmesh_2d(this->comm.get_mpi_comm(), dim_x, dim_y,
                                  &this->mpi_comm)) {
        throw std::runtime_error("Failed to create 2d PFFT process mesh.");
      }
    }

    std::array<ptrdiff_t, DimS> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    ptrdiff_t res[DimS], loc[DimS], fres[DimS], floc[DimS];
    this->workspace_size =
      pfft_local_size_many_dft_r2c(DimS, narr.data(), narr.data(), narr.data(),
                                   Field_t::nb_components,
                                   PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS,
                                   this->mpi_comm,
                                   PFFT_TRANSPOSED_OUT,
                                   res, loc, fres, floc);
    std::copy(res, res+DimS, this->subdomain_resolutions.begin());
    std::copy(loc, loc+DimS, this->subdomain_locations.begin());
    std::copy(fres, fres+DimS, this->fourier_resolutions.begin());
    std::copy(floc, floc+DimS, this->fourier_locations.begin());
    // TODO: Enable this to enable 2d process mesh. This does not pass tests.
    //for (int i = 0; i < DimS-1; ++i) {
    for (int i = 0; i < 1; ++i) {
      std::swap(this->fourier_resolutions[i], this->fourier_resolutions[i+1]);
      std::swap(this->fourier_locations[i], this->fourier_locations[i+1]);
    }
    
    for (auto & n: this->subdomain_resolutions) {
      if (n == 0) {
        throw std::runtime_error("PFFT planning returned zero resolution. "
                                 "You may need to run on fewer processes.");
      }
    }
    for (auto & n: this->fourier_resolutions) {
      if (n == 0) {
        throw std::runtime_error("PFFT planning returned zero Fourier "
                                 "resolution. You may need to run on fewer "
                                 "processes.");
      }
    }

    for (auto && pixel:
         std::conditional_t<
         DimS==2,
         CcoordOps::Pixels<DimS, 1, 0>,
         // TODO: This should be the correct order of dimension for a 2d
         // process mesh, but tests don't pass.
         //CcoordOps::Pixels<DimS, 1, 2, 0>
         CcoordOps::Pixels<DimS, 1, 0, 2>
         >(this->fourier_resolutions, this->fourier_locations)) {
      this->work_space_container.add_pixel(pixel);
    }
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void PFFTEngine<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }

    // Initialize parent after local resolutions have been determined and
    // work space has been initialized
    Parent::initialise(plan_flags);

    this->real_workspace = pfft_alloc_real(2*this->workspace_size);
    // We need to check whether the workspace provided by our field is large
    // enough. PFFT may request a workspace size larger than the nominal size
    // of the complex buffer.
    if (long(this->work.size()*Field_t::nb_components) < this->workspace_size) {
      this->work.set_pad_size(this->workspace_size -
                              Field_t::nb_components*this->work.size());
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
      throw std::runtime_error("unknown planner flag type");
      break;
    }

    std::array<ptrdiff_t, DimS> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    Real * in{this->real_workspace};
    pfft_complex * out{reinterpret_cast<pfft_complex*>(this->work.data())};
    this->plan_fft = pfft_plan_many_dft_r2c(DimS, narr.data(), narr.data(),
                                            narr.data(), Field_t::nb_components,
                                            PFFT_DEFAULT_BLOCKS,
                                            PFFT_DEFAULT_BLOCKS,
                                            in, out, this->mpi_comm,
                                            PFFT_FORWARD,
                                            PFFT_TRANSPOSED_OUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("r2c plan failed");
    }

    pfft_complex * i_in{reinterpret_cast<pfft_complex*>(this->work.data())};
    Real * i_out{this->real_workspace};

    this->plan_ifft = pfft_plan_many_dft_c2r(DimS, narr.data(), narr.data(),
                                             narr.data(),
                                             Field_t::nb_components,
                                             PFFT_DEFAULT_BLOCKS,
                                             PFFT_DEFAULT_BLOCKS,
                                             i_in, i_out,
                                             this->mpi_comm,
                                             PFFT_BACKWARD,
                                             PFFT_TRANSPOSED_IN | flags);
    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("c2r plan failed");
    }
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  PFFTEngine<DimS, DimM>::~PFFTEngine<DimS, DimM>() noexcept {
    if (this->real_workspace != nullptr) pfft_free(this->real_workspace);
    if (this->plan_fft != nullptr) pfft_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr) pfft_destroy_plan(this->plan_ifft);
    if (this->mpi_comm != this->comm.get_mpi_comm()) {
      MPI_Comm_free(&this->mpi_comm);
    }
    // TODO: We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    //this->nb_engines--;
    //if (!this->nb_engines) pfft_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename PFFTEngine<DimS, DimM>::Workspace_t &
  PFFTEngine<DimS, DimM>::fft (Field_t & field) {
    if (!this->plan_fft) {
        throw std::runtime_error("fft plan not allocated");
    }
    if (field.size() != CcoordOps::get_size(this->subdomain_resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    // Copy field data to workspace buffer. This is necessary because workspace
    // buffer is larger than field buffer.
    std::copy(field.data(), field.data()+Field_t::nb_components*field.size(),
              this->real_workspace);
    pfft_execute_dft_r2c(
      this->plan_fft, this->real_workspace,
      reinterpret_cast<pfft_complex*>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  PFFTEngine<DimS, DimM>::ifft (Field_t & field) const {
    if (!this->plan_ifft) {
        throw std::runtime_error("ifft plan not allocated");
    }
    if (field.size() != CcoordOps::get_size(this->subdomain_resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    pfft_execute_dft_c2r(
      this->plan_ifft, reinterpret_cast<pfft_complex*>(this->work.data()),
      this->real_workspace);
    std::copy(this->real_workspace,
              this->real_workspace+Field_t::nb_components*field.size(),
              field.data());
  }

  template class PFFTEngine<twoD, twoD>;
  template class PFFTEngine<twoD, threeD>;
  template class PFFTEngine<threeD, threeD>;
}  // muSpectre
