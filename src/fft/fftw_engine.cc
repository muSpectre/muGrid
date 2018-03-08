/**
 * @file   fftw_engine.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  implements the fftw engine
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
#include "fft/fftw_engine.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  FFTWEngine<DimS, DimM>::FFTWEngine(Ccoord resolutions, Rcoord lengths)
    :Parent{resolutions, lengths},
     hermitian_resolutions{CcoordOps::get_hermitian_sizes(resolutions)}
  {
    for (auto && pixel: CcoordOps::Pixels<DimS>(this->hermitian_resolutions)) {
      this->work_space_container.add_pixel(pixel);
    }
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void FFTWEngine<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }
    Parent::initialise(plan_flags);

    const int & rank = DimS;
    std::array<int, DimS> narr;
    const int * const n = &narr[0];
    std::copy(this->resolutions.begin(), this->resolutions.end(), narr.begin());
    int howmany = Field_t::nb_components;
    //temporary buffer for plan
    size_t alloc_size = CcoordOps::get_size(this->resolutions) *howmany;
    Real * r_work_space = fftw_alloc_real(alloc_size);
    Real * in = r_work_space;
    const int * const inembed = nullptr;//nembed are tricky: they refer to physical layout
    int  istride = howmany;
    int idist = 1;
    fftw_complex * out = reinterpret_cast<fftw_complex*>(this->work.data());
    const int * const onembed = nullptr;
    int ostride = howmany;
    int odist = idist;

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

    this->plan_fft = fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                            idist, out, onembed, ostride, odist,
                                            FFTW_PRESERVE_INPUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("Plan failed");
    }
    fftw_execute_dft_r2c(this->plan_fft,
                         in,
                         reinterpret_cast<fftw_complex*>(this->work.data()));

    fftw_complex * i_in = reinterpret_cast<fftw_complex*>(this->work.data());
    Real * i_out = r_work_space;

    this->plan_ifft = fftw_plan_many_dft_c2r(rank, n, howmany, i_in, inembed,
                                             istride, idist, i_out, onembed,
                                             ostride, odist, flags);

    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("Plan failed");
    }
    fftw_free(r_work_space);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  FFTWEngine<DimS, DimM>::~FFTWEngine<DimS, DimM>() noexcept {
    fftw_destroy_plan(this->plan_fft);
    fftw_destroy_plan(this->plan_ifft);
    fftw_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename FFTWEngine<DimS, DimM>::Workspace_t &
  FFTWEngine<DimS, DimM>::fft (Field_t & field) {
    fftw_execute_dft_r2c(this->plan_fft,
                         field.data(),
                         reinterpret_cast<fftw_complex*>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  FFTWEngine<DimS, DimM>::ifft (Field_t & field) const {
    if (field.size() != CcoordOps::get_size(this->resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    fftw_execute_dft_c2r(this->plan_ifft,
                         reinterpret_cast<fftw_complex*>(this->work.data()),
                         field.data());
  }

  template class FFTWEngine<twoD, twoD>;
  template class FFTWEngine<twoD, threeD>;
  template class FFTWEngine<threeD, threeD>;
}  // muSpectre
