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
  FFTWMPIEngine<DimS, DimM>::FFTWMPIEngine(Ccoord resolutions,
                                           Communicator comm)
    :Parent{resolutions, comm}
  {
    if (!this->nb_engines) fftw_mpi_init();
    this->nb_engines++;

    std::array<ptrdiff_t, DimS> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    narr[DimS-1] = this->domain_resolutions[DimS-1]/2+1;
    ptrdiff_t res_x, loc_x, res_y, loc_y;
    this->workspace_size =
      fftw_mpi_local_size_many_transposed(DimS, narr.data(),
                                          Field_t::nb_components,
                                          FFTW_MPI_DEFAULT_BLOCK,
                                          FFTW_MPI_DEFAULT_BLOCK,
                                          this->comm.get_mpi_comm(),
                                          &res_x, &loc_x, &res_y, &loc_y);
    this->fourier_resolutions[1] = this->fourier_resolutions[0];
    this->fourier_locations[1] = this->fourier_locations[0];
    this->subdomain_resolutions[0] = res_x;
    this->subdomain_locations[0] = loc_x;
    this->fourier_resolutions[0] = res_y;
    this->fourier_locations[0] = loc_y;

    for (auto & n: this->subdomain_resolutions) {
      if (n == 0) {
        throw std::runtime_error("FFTW MPI planning returned zero resolution. "
                                 "You may need to run on fewer processes.");
      }
    }
    for (auto & n: this->fourier_resolutions) {
      if (n == 0) {
        throw std::runtime_error("FFTW MPI planning returned zero Fourier "
                                 "resolution. You may need to run on fewer "
                                 "processes.");
      }
    }

    for (auto && pixel:
         std::conditional_t<
           DimS==2,
           CcoordOps::Pixels<DimS, 1, 0>,
           CcoordOps::Pixels<DimS, 1, 0, 2>
         >(this->fourier_resolutions, this->fourier_locations)) {
           this->work_space_container.add_pixel(pixel);
    }
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void FFTWMPIEngine<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }

    // Initialize parent after local resolutions have been determined and
    // work space has been initialized
    Parent::initialise(plan_flags);

    this->real_workspace = fftw_alloc_real(2*this->workspace_size);
    // We need to check whether the workspace provided by our field is large
    // enough. MPI parallel FFTW may request a workspace size larger than the
    // nominal size of the complex buffer.
    if (long(this->work.size()*Field_t::nb_components) < this->workspace_size) {
      this->work.set_pad_size(this->workspace_size -
                              Field_t::nb_components*this->work.size());
    }

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

    std::array<ptrdiff_t, DimS> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    Real * in{this->real_workspace};
    fftw_complex * out{reinterpret_cast<fftw_complex*>(this->work.data())};
    this->plan_fft = fftw_mpi_plan_many_dft_r2c(
      DimS, narr.data(), Field_t::nb_components, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, in, out, this->comm.get_mpi_comm(),
      FFTW_MPI_TRANSPOSED_OUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("r2c plan failed");
    }

    fftw_complex * i_in = reinterpret_cast<fftw_complex*>(this->work.data());
    Real * i_out = this->real_workspace;

    this->plan_ifft = fftw_mpi_plan_many_dft_c2r(
      DimS, narr.data(), Field_t::nb_components, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, i_in, i_out, this->comm.get_mpi_comm(),
      FFTW_MPI_TRANSPOSED_IN | flags);
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
    // TODO: We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    //this->nb_engines--;
    //if (!this->nb_engines) fftw_mpi_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename FFTWMPIEngine<DimS, DimM>::Workspace_t &
  FFTWMPIEngine<DimS, DimM>::fft (Field_t & field) {
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("fft plan not initialised");
    }
    if (field.size() != CcoordOps::get_size(this->subdomain_resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride = (Field_t::nb_components*
                         this->subdomain_resolutions[DimS-1]);
    ptrdiff_t wstride = (Field_t::nb_components*2*
                         (this->subdomain_resolutions[DimS-1]/2+1));
    ptrdiff_t n = field.size()/this->subdomain_resolutions[DimS-1];

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
    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("ifft plan not initialised");
    }
    if (field.size() != CcoordOps::get_size(this->subdomain_resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    // Compute inverse FFT
    fftw_mpi_execute_dft_c2r(
      this->plan_ifft, reinterpret_cast<fftw_complex*>(this->work.data()),
      this->real_workspace);
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride{
      Field_t::nb_components*this->subdomain_resolutions[DimS-1]
    };
    ptrdiff_t wstride{
      Field_t::nb_components*2*(this->subdomain_resolutions[DimS-1]/2+1)
    };
    ptrdiff_t n(field.size()/this->subdomain_resolutions[DimS-1]);

    auto fdata{field.data()};
    auto wdata{this->real_workspace};
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
