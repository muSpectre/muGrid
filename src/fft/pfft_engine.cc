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
  PFFTEngine<DimS, DimM>::PFFTEngine(Ccoord resolutions, Rcoord lengths,
                                     Communicator comm)
    :Parent{resolutions, lengths}, comm{comm}
  {
    if (!this->nb_engines) pfft_init();
    this->nb_engines++;
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void PFFTEngine<DimS, DimM>::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }
    const int & rank = DimS;
    ptrdiff_t howmany = Field_t::nb_components;

    ptrdiff_t narr[3];
    std::copy(this->domain_resolutions.begin(), this->domain_resolutions.end(),
              narr);
    ptrdiff_t res[3], loc[3], fres[3], floc[3];
    ptrdiff_t loc_alloc_size{
      pfft_local_size_many_dft_r2c(
        rank, narr, narr, narr, howmany, PFFT_DEFAULT_BLOCK,
        PFFT_DEFAULT_BLOCKS, this->comm.get_mpi_comm(), PFFT_TRANSPOSED_OUT,
        res, loc, fres, floc)};
    std::copy(res, res+3, this->resolutions.begin());
    std::copy(loc, loc+3, this->locations.begin());
    std::copy(fres, fres+3, this->fourier_resolutions.begin());
    std::copy(floc, floc+3, this->fourier_locations.begin());

    std::cout << "Real space: " << this->locations << " " << this->resolutions << std::endl;
    std::cout << "Fourier space: " << this->fourier_locations << " " << this->fourier_resolutions << std::endl;

    for (auto && pixel: CcoordOps::Pixels<DimS>(this->fourier_resolutions)) {
      this->work_space_container.add_pixel(pixel);
    }
    Parent::initialise(plan_flags);

    Real * r_work_space = pfft_alloc_real(2*loc_alloc_size);
    Real * in = r_work_space;
    pfft_complex * out = reinterpret_cast<pfft_complex*>(this->work.data());

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

    this->plan_fft = pfft_plan_many_dft_r2c(
      rank, narr, narr, narr, howmany, PFFT_DEFAULT_BLOCK, PFFT_DEFAULT_BLOCKS,
      in, out, this->comm.get_mpi_comm(), PFFT_FORWARD,
      PFFT_PRESERVE_INPUT | PFFT_TRANSPOSED_OUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("r2c plan failed");
    }
    pfft_execute_dft_r2c(
      this->plan_fft, in, reinterpret_cast<pfft_complex*>(this->work.data()));

    pfft_complex * i_in = reinterpret_cast<pfft_complex*>(this->work.data());
    Real * i_out = r_work_space;

    this->plan_ifft = pfft_plan_many_dft_c2r(
      rank, narr, narr, narr, howmany, PFFT_DEFAULT_BLOCK, PFFT_DEFAULT_BLOCKS,
      i_in, i_out, this->comm.get_mpi_comm(), PFFT_BACKWARD,
      PFFT_PRESERVE_INPUT | PFFT_TRANSPOSED_IN | flags);
    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("c2r plan failed");
    }
    pfft_free(r_work_space);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  PFFTEngine<DimS, DimM>::~PFFTEngine<DimS, DimM>() noexcept {
    pfft_destroy_plan(this->plan_fft);
    pfft_destroy_plan(this->plan_ifft);
    this->nb_engines--;
    if (!this->nb_engines) pfft_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  typename PFFTEngine<DimS, DimM>::Workspace_t &
  PFFTEngine<DimS, DimM>::fft (Field_t & field) {
    pfft_execute_dft_r2c(
      this->plan_fft, field.data(),
      reinterpret_cast<pfft_complex*>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  PFFTEngine<DimS, DimM>::ifft (Field_t & field) const {
    if (field.size() != CcoordOps::get_size(this->resolutions)) {
      throw std::runtime_error("size mismatch");
    }
    pfft_execute_dft_c2r(
      this->plan_ifft, reinterpret_cast<pfft_complex*>(this->work.data()),
      field.data());
  }

  template class PFFTEngine<twoD, twoD>;
  template class PFFTEngine<twoD, threeD>;
  template class PFFTEngine<threeD, threeD>;
}  // muSpectre
