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

#include <sstream>

#include "fftw_engine.hh"
#include <libmugrid/ccoord_operations.hh>

namespace muFFT {

  template <Dim_t Dim>
  FFTWEngine<Dim>::FFTWEngine(Ccoord nb_grid_pts, Dim_t nb_components,
                              Communicator comm)
      : Parent{nb_grid_pts, nb_components, comm}, plan_fft{nullptr},
        plan_ifft{nullptr} {
    for (auto && pixel :
         muGrid::CcoordOps::Pixels<Dim>(this->nb_fourier_grid_pts)) {
      this->work_space_container.add_pixel(pixel);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  void FFTWEngine<Dim>::initialise(FFT_PlanFlags plan_flags) {
    if (this->comm.size() > 1) {
      std::stringstream error;
      error << "FFTW engine does not support MPI parallel execution, but a "
            << "communicator of size " << this->comm.size() << " was passed "
            << "during construction";
      throw std::runtime_error(error.str());
    }
    if (this->initialised) {
      throw std::runtime_error("double initialisation, will leak memory");
    }
    Parent::initialise(plan_flags);

    const int & rank = Dim;
    std::array<int, Dim> narr;
    const int * const n = &narr[0];
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    for (Dim_t i = 0; i < Dim; ++i) {
      narr[i] = this->nb_subdomain_grid_pts[Dim - 1 - i];
    }
    int howmany = this->nb_components;
    // temporary buffer for plan
    size_t alloc_size =
        (muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts) * howmany);
    Real * r_work_space = fftw_alloc_real(alloc_size);
    Real * in = r_work_space;
    const int * const inembed =
        nullptr;  // nembed are tricky: they refer to physical layout
    int istride = howmany;
    int idist = 1;
    fftw_complex * out = reinterpret_cast<fftw_complex *>(this->work.data());
    const int * const onembed = nullptr;
    int ostride = istride;
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

    this->plan_fft = fftw_plan_many_dft_r2c(
        rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
        odist, FFTW_PRESERVE_INPUT | flags);
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("Plan failed");
    }

    fftw_complex * i_in = reinterpret_cast<fftw_complex *>(this->work.data());
    Real * i_out = r_work_space;

    this->plan_ifft =
        fftw_plan_many_dft_c2r(rank, n, howmany, i_in, inembed, istride, idist,
                               i_out, onembed, ostride, odist, flags);

    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("Plan failed");
    }
    fftw_free(r_work_space);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  FFTWEngine<Dim>::~FFTWEngine<Dim>() noexcept {
    if (this->plan_fft != nullptr)
      fftw_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr)
      fftw_destroy_plan(this->plan_ifft);
    // TODO(Till): We cannot run fftw_cleanup since subsequent FFTW calls will
    // fail but multiple FFT engines can be active at the same time.
    // fftw_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  typename FFTWEngine<Dim>::Workspace_t &
  FFTWEngine<Dim>::fft(Field_t & field) {
    if (this->plan_fft == nullptr) {
      throw std::runtime_error("fft plan not initialised");
    }
    if (field.size() !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      throw std::runtime_error("size mismatch");
    }
    fftw_execute_dft_r2c(this->plan_fft, field.data(),
                         reinterpret_cast<fftw_complex *>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t Dim>
  void FFTWEngine<Dim>::ifft(Field_t & field) const {
    if (this->plan_ifft == nullptr) {
      throw std::runtime_error("ifft plan not initialised");
    }
    if (field.size() !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      throw std::runtime_error("size mismatch");
    }
    fftw_execute_dft_c2r(this->plan_ifft,
                         reinterpret_cast<fftw_complex *>(this->work.data()),
                         field.data());
  }

  template class FFTWEngine<muGrid::oneD>;
  template class FFTWEngine<muGrid::twoD>;
  template class FFTWEngine<muGrid::threeD>;
}  // namespace muFFT
