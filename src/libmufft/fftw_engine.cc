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

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/exception.hh>

#include "fftw_engine.hh"

namespace muFFT {

  FFTWEngine::FFTWEngine(const DynCcoord_t & nb_grid_pts, Communicator comm,
                         const FFT_PlanFlags & plan_flags,
                         bool allow_temporary_buffer, bool allow_destroy_input)
      : Parent{nb_grid_pts, comm, plan_flags, allow_temporary_buffer,
               allow_destroy_input} {
    this->initialise_field_collections();
    ++this->nb_engines;
  }

  /* ---------------------------------------------------------------------- */
  FFTWEngine::FFTWEngine(const DynCcoord_t & nb_grid_pts,
                         const FFT_PlanFlags & plan_flags,
                         bool allow_temporary_buffer, bool allow_destroy_input)
      : FFTWEngine{nb_grid_pts, Communicator(), plan_flags,
                   allow_temporary_buffer, allow_destroy_input} {
    ++this->nb_engines;
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::create_plan(const Index_t & nb_dof_per_pixel) {
    if (this->has_plan_for(nb_dof_per_pixel)) {
      // plan already exists, we can bail
      return;
    }
    if (this->comm.size() > 1) {
      std::stringstream error;
      error << "FFTW engine does not support MPI parallel execution, but a "
            << "communicator of size " << this->comm.size() << " was passed "
            << "during construction";
      throw FFTEngineError(error.str());
    }

    const int & rank{this->nb_subdomain_grid_pts.get_dim()};
    std::vector<int> narr(rank);
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    for (Index_t i{0}; i < rank; ++i) {
      narr[i] = this->nb_subdomain_grid_pts[rank - 1 - i];
    }
    const int * const n{&narr[0]};
    int howmany{static_cast<int>(nb_dof_per_pixel)};
    // temporary buffer for plan
    size_t alloc_size{muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts) *
                      howmany};
    Real * r_work_space{fftw_alloc_real(alloc_size)};
    if (r_work_space == nullptr) {
      throw FFTEngineError("'r_work_space' allocation failed");
    }
    Real * in{r_work_space};
    // nembed are tricky: they refer to physical layout
    const int * const inembed{nullptr};
    int istride{howmany};
    int idist{1};
    auto && nb_fft_pts{[](const DynCcoord_t & grid_pts) {
      int retval{1};
      for (auto && nb : grid_pts) {
        retval *= nb;
      }
      return retval;
    }(this->get_nb_fourier_grid_pts())};
    fftw_complex * out{fftw_alloc_complex(howmany * nb_fft_pts)};
    if (out == nullptr) {
      throw FFTEngineError("'out' allocation failed");
    }
    const int * const onembed{nullptr};
    int ostride{istride};
    int odist{idist};

    unsigned int flags;
    switch (this->plan_flags) {
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
      throw FFTEngineError("unknown planner flag type");
      break;
    }

    this->fft_plans[nb_dof_per_pixel] = fftw_plan_many_dft_r2c(
        rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
        odist,
        (this->allow_destroy_input ? FFTW_DESTROY_INPUT : FFTW_PRESERVE_INPUT) |
            flags);
    if (this->fft_plans.at(nb_dof_per_pixel) == nullptr) {
      throw FFTEngineError("Plan failed");
    }

    fftw_complex * i_in{out};
    Real * i_out{r_work_space};

    this->ifft_plans[nb_dof_per_pixel] =
        fftw_plan_many_dft_c2r(rank, n, howmany, i_in, inembed, istride, idist,
                               i_out, onembed, ostride, odist, flags);

    if (this->ifft_plans.at(nb_dof_per_pixel) == nullptr) {
      throw FFTEngineError("Plan failed");
    }

    // r2hc_plans

    Real * r_work_space_2{fftw_alloc_real(alloc_size)};
    if (r_work_space_2 == nullptr) {
      throw FFTEngineError("'r_work_space_2' allocation failed");
    }
    Real * r2hc_out{r_work_space_2};

    std::vector<fftw_r2r_kind> fft_kinds(rank);
    for (auto && k : fft_kinds) {
      k = FFTW_R2HC;
    }
    std::vector<fftw_r2r_kind> ifft_kinds(rank);
    for (auto && k : ifft_kinds) {
      k = FFTW_HC2R;
    }

    this->hcfft_plans[nb_dof_per_pixel] = fftw_plan_many_r2r(
        rank, n, howmany, in, inembed, istride, idist, r2hc_out, onembed,
        ostride, odist, fft_kinds.data(),
        (this->allow_destroy_input ? FFTW_DESTROY_INPUT : FFTW_PRESERVE_INPUT) |
            flags);
    if (this->hcfft_plans.at(nb_dof_per_pixel) == nullptr) {
      throw FFTEngineError("Plan failed");
    }

    Real * ir2hc_in{r_work_space_2};
    this->ihcfft_plans[nb_dof_per_pixel] = fftw_plan_many_r2r(
        rank, n, howmany, ir2hc_in, inembed, istride, idist, i_out, onembed,
        ostride, odist, ifft_kinds.data(), flags);

    if (this->ihcfft_plans.at(nb_dof_per_pixel) == nullptr) {
      throw FFTEngineError("Plan failed");
    }

    fftw_free(r_work_space);
    fftw_free(r_work_space_2);
    fftw_free(out);
    this->planned_nb_dofs.insert(nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  FFTWEngine::~FFTWEngine() noexcept {
    for (auto && nb_dof_per_pixel : this->planned_nb_dofs) {
      auto && fft_plan{this->fft_plans.at(nb_dof_per_pixel)};
      if (fft_plan != nullptr) {
        fftw_destroy_plan(fft_plan);
      }
      auto && ifft_plan{this->ifft_plans.at(nb_dof_per_pixel)};
      if (ifft_plan != nullptr) {
        fftw_destroy_plan(ifft_plan);
      }
      auto && hcfft_plan{this->hcfft_plans.at(nb_dof_per_pixel)};
      if (hcfft_plan != nullptr) {
        fftw_destroy_plan(hcfft_plan);
      }
      auto && ihcfft_plan{this->ihcfft_plans.at(nb_dof_per_pixel)};
      if (ihcfft_plan != nullptr) {
        fftw_destroy_plan(ihcfft_plan);
      }
    }
    --this->nb_engines;
    if (this->nb_engines == 0) {
      // TODO(Pastewka): Check with Lars why this crashes the sequential(!) MPI
      // tests
#ifndef WITH_MPI
       fftw_cleanup();
#endif
    }
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::compute_fft(const RealField_t & input_field,
                               FourierField_t & output_field) const {
    fftw_execute_dft_r2c(this->fft_plans.at(input_field.get_nb_dof_per_pixel()),
                         input_field.data(),
                         reinterpret_cast<fftw_complex *>(output_field.data()));
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::compute_ifft(const FourierField_t & input_field,
                                RealField_t & output_field) const {
    fftw_execute_dft_c2r(
        this->ifft_plans.at(input_field.get_nb_dof_per_pixel()),
        reinterpret_cast<fftw_complex *>(input_field.data()),
        output_field.data());
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::compute_hcfft(const RealField_t & input_field,
                                 RealField_t & output_field) const {
    fftw_execute_r2r(this->hcfft_plans.at(input_field.get_nb_dof_per_pixel()),
                     input_field.data(), output_field.data());
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::compute_ihcfft(const RealField_t & input_field,
                                  RealField_t & output_field) const {
    fftw_execute_r2r(this->ihcfft_plans.at(input_field.get_nb_dof_per_pixel()),
                     input_field.data(), output_field.data());
  }

  /* ---------------------------------------------------------------------- */
  std::unique_ptr<FFTEngineBase> FFTWEngine::clone() const {
    return std::make_unique<FFTWEngine>(
        this->get_nb_domain_grid_pts(), this->get_communicator(),
        this->plan_flags, this->allow_temporary_buffer,
        this->allow_destroy_input);
  }

  /* ---------------------------------------------------------------------- */
  Index_t FFTWEngine::nb_engines{0};

}  // namespace muFFT
