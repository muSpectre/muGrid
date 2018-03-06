/**
 * @file   fftw_engine_mpi.cc
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
#include "fft/fftw_engine_mpi.hh"

namespace muSpectre {

  template <Dim_t DimsS, Dim_t DimM>
  int FFTWEngineMPI<DimsS, DimM>::nb_engines{0};

  template <Dim_t DimS, Dim_t DimM>
  FFTWEngineMPI<DimS, DimM>::FFTWEngineMPI(Ccoord resolutions, Rcoord lengths)
    :Parent{resolutions, lengths},
     hermitian_resolutions{CcoordOps::get_hermitian_sizes(resolutions)}
  {
    if (!this->nb_engines) fftw_mpi_init();
    this->nb_engines++;
    for (auto && pixel: CcoordOps::Pixels<DimS>(this->hermitian_resolutions)) {
      this->work_space_container.add_pixel(pixel);
    }
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void FFTWEngineMPI<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }
    Parent::initialise(plan_flags);

    const int & rank = DimS;
    int howmany = Field_t::nb_components;

    std::array<ptrdiff_t, 3> narr;
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr.begin());
    narr[2] = this->domain_resolutions[2]/2+1;
    ptrdiff_t loc_res_x, loc_loc_x;
    ptrdiff_t loc_alloc_size{
      fftw_mpi_local_size_many(
        rank, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD,
        &loc_res_x, &loc_loc_x)};
    this->resolutions[0] = loc_res_x;
    this->locations[0] = loc_loc_x;

    //std::array<int, DimS> narr;
    //const int * const n = &narr[0];
    //std::copy(this->resolutions.begin(), this->resolutions.end(), narr.begin());
    //temporary buffer for plan
    Real * r_work_space = fftw_alloc_real(2*loc_alloc_size);
    Real * in = r_work_space;
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

    narr[2] = this->domain_resolutions[2];
    this->plan_fft = fftw_mpi_plan_many_dft_r2c(
      rank, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, in, out, MPI_COMM_WORLD,
      FFTW_PRESERVE_INPUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("r2c plan failed");
    }
    fftw_mpi_execute_dft_r2c(
      this->plan_fft, in, reinterpret_cast<fftw_complex*>(this->work.data()));

    fftw_complex * i_in = reinterpret_cast<fftw_complex*>(this->work.data());
    Real * i_out = r_work_space;

    this->plan_ifft = fftw_mpi_plan_many_dft_c2r(
      rank, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
      FFTW_MPI_DEFAULT_BLOCK, i_in, i_out, MPI_COMM_WORLD,
      FFTW_PRESERVE_INPUT | flags);

    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("c2r plan failed");
    }
    fftw_free(r_work_space);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  FFTWEngineMPI<DimS, DimM>::~FFTWEngineMPI<DimS, DimM>() noexcept {
    fftw_destroy_plan(this->plan_fft);
    fftw_destroy_plan(this->plan_ifft);
    this->nb_engines--;
    if (!this->nb_engines) fftw_mpi_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename FFTWEngineMPI<DimS, DimM>::Workspace_t &
  FFTWEngineMPI<DimS, DimM>::fft (Field_t & field) {
    fftw_mpi_execute_dft_r2c(
      this->plan_fft, field.data(),
      reinterpret_cast<fftw_complex*>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  FFTWEngineMPI<DimS, DimM>::ifft (Field_t & field) const {
    if (field.size() != CcoordOps::get_size(this->resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    fftw_mpi_execute_dft_c2r(
      this->plan_ifft, reinterpret_cast<fftw_complex*>(this->work.data()),
      field.data());
  }

  template class FFTWEngineMPI<twoD, twoD>;
  template class FFTWEngineMPI<twoD, threeD>;
  template class FFTWEngineMPI<threeD, threeD>;
}  // muSpectre
