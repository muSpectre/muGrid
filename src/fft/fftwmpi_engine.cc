/**
 * @file   fftwmpi_engine.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   06 Mar 2017
 *
 * @brief  implements the MPI-parallel fftw engine
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
#include "fft/fftwmpi_engine.hh"

namespace muSpectre {

  template <Dim_t DimsS, Dim_t DimM>
  int FFTWMPIEngine<DimsS, DimM>::nb_engines{0};

  template <Dim_t DimS, Dim_t DimM>
  FFTWMPIEngine<DimS, DimM>::FFTWMPIEngine(Ccoord resolutions, Rcoord lengths,
                                           Communicator comm)
    :Parent{resolutions, lengths}, comm{comm}, real_workspace{nullptr}
  {
    if (!this->nb_engines) fftw_mpi_init();
    this->nb_engines++;
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void FFTWMPIEngine<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }
    const int & rank = DimS;
    ptrdiff_t howmany = Field_t::nb_components;

    std::array<ptrdiff_t, DimS> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    narr[DimS-1] = this->domain_resolutions[DimS-1]/2+1;
    ptrdiff_t res_x, loc_x;//, res_y, loc_y;
    this->workspace_size = fftw_mpi_local_size_many(
        rank, narr.data(), howmany, //FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, this->comm.get_mpi_comm(),
                                                    &res_x, &loc_x);//, &res_y, &loc_y);
    //this->fourier_resolutions[1] = this->resolutions[0];
    //this->fourier_locations[1] = this->locations[0];
    this->resolutions[0] = res_x;
    this->locations[0] = loc_x;
    //this->fourier_resolutions[0] = res_y;
    //this->fourier_locations[0] = loc_y;
    this->fourier_resolutions[0] = res_x;
    this->fourier_locations[0] = loc_x;

    for (auto && pixel: CcoordOps::Pixels<DimS>(this->fourier_resolutions)) {
      this->work_space_container.add_pixel(pixel);
    }
    Parent::initialise(plan_flags);

    this->real_workspace = fftw_alloc_real(2*this->workspace_size);
    Real * in = this->real_workspace;
    fftw_complex * out = reinterpret_cast<fftw_complex*>(this->work.data());

    unsigned int flags;
    switch (plan_flags) {
    case FFT_PlanFlags::estimate: {
      flags = FFTW_ESTIMATE;
      break;
    }
    case FFT_PlanFlags::measure: {
      flags = FFTW_MEASURE;
      break;
    }
    case FFT_PlanFlags::patient: {
      flags = FFTW_PATIENT;
      break;
    }
    default:
      throw std::runtime_error("unknown planner flag type");
      break;
    }

    narr[DimS-1] = this->domain_resolutions[DimS-1];
    this->plan_fft = fftw_mpi_plan_many_dft_r2c(
      rank, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, in, out, this->comm.get_mpi_comm(),
      flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("r2c plan failed");
    }

    fftw_mpi_execute_dft_r2c(
      this->plan_fft, in, reinterpret_cast<fftw_complex*>(this->work.data()));

    fftw_complex * i_in = reinterpret_cast<fftw_complex*>(this->work.data());
    Real * i_out = this->real_workspace;

    this->plan_ifft = fftw_mpi_plan_many_dft_c2r(
      rank, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, i_in, i_out, this->comm.get_mpi_comm(),
      flags);
    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("c2r plan failed");
    }
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  FFTWMPIEngine<DimS, DimM>::~FFTWMPIEngine<DimS, DimM>() noexcept {
    if (this->real_workspace != nullptr) fftw_free(this->real_workspace);
    if (this->plan_fft != nullptr) fftw_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr) fftw_destroy_plan(this->plan_ifft);
    this->nb_engines--;
    if (!this->nb_engines) fftw_mpi_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename FFTWMPIEngine<DimS, DimM>::Workspace_t &
  FFTWMPIEngine<DimS, DimM>::fft (Field_t & field) {
    // Copy non-padded field to padded real_workspace
    ptrdiff_t fstride = Field_t::nb_components*this->resolutions[DimS-1];
    ptrdiff_t wstride = Field_t::nb_components*this->fourier_resolutions[DimS-1];
    ptrdiff_t n = this->workspace_size/wstride;
    wstride *= 2;

    auto fdata = field.data();
    auto wdata = this->real_workspace;
    for (int i = 0; i < n; i++) {
      std::copy(fdata, fdata+fstride, wdata);
      fdata += fstride;
      wdata += wstride;
    }
    // Compute FFT
    fftw_mpi_execute_dft_r2c(
      this->plan_fft, this->real_workspace,
      reinterpret_cast<fftw_complex*>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  FFTWMPIEngine<DimS, DimM>::ifft (Field_t & field) const {
    if (field.size() != CcoordOps::get_size(this->resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    // Compute inverse FFT
    fftw_mpi_execute_dft_c2r(
      this->plan_ifft, reinterpret_cast<fftw_complex*>(this->work.data()),
      this->real_workspace);
    // Copy non-padded field to padded real_workspace
    ptrdiff_t fstride = Field_t::nb_components*this->resolutions[DimS-1];
    ptrdiff_t wstride = Field_t::nb_components*this->fourier_resolutions[DimS-1];
    ptrdiff_t n = this->workspace_size/wstride;
    wstride *= 2;
  
    auto fdata = field.data();
    auto wdata = this->real_workspace;
    for (int i = 0; i < n; i++) {
      std::copy(wdata, wdata+fstride, fdata);
      fdata += fstride;
      wdata += wstride;
    }
  }

  template class FFTWMPIEngine<twoD, twoD>;
  template class FFTWMPIEngine<twoD, threeD>;
  template class FFTWMPIEngine<threeD, threeD>;
}  // muSpectre
