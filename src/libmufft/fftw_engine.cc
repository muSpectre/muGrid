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

using muGrid::RuntimeError;

namespace muFFT {

  FFTWEngine::FFTWEngine(const DynCcoord_t & nb_grid_pts,
                         Dim_t nb_dof_per_pixel, Communicator comm)
      : Parent{nb_grid_pts, nb_dof_per_pixel, comm}, plan_fft{nullptr},
        plan_ifft{nullptr} {}

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::initialise(FFT_PlanFlags plan_flags) {
    if (this->comm.size() > 1) {
      std::stringstream error;
      error << "FFTW engine does not support MPI parallel execution, but a "
            << "communicator of size " << this->comm.size() << " was passed "
            << "during construction";
      throw RuntimeError(error.str());
    }
    if (this->initialised) {
      throw RuntimeError("double initialisation, will leak memory");
    }
    Parent::initialise(plan_flags);

    const int & rank = this->nb_subdomain_grid_pts.get_dim();
    std::vector<int> narr(rank);
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    for (Dim_t i = 0; i < rank; ++i) {
      narr[i] = this->nb_subdomain_grid_pts[rank - 1 - i];
    }
    const int * const n = &narr[0];
    int howmany = this->nb_dof_per_pixel;
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
      throw RuntimeError("unknown planner flag type");
      break;
    }

    this->plan_fft = fftw_plan_many_dft_r2c(
        rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride,
        odist, FFTW_PRESERVE_INPUT | flags);
    if (this->plan_fft == nullptr) {
      throw RuntimeError("Plan failed");
    }

    fftw_complex * i_in = reinterpret_cast<fftw_complex *>(this->work.data());
    Real * i_out = r_work_space;

    this->plan_ifft =
        fftw_plan_many_dft_c2r(rank, n, howmany, i_in, inembed, istride, idist,
                               i_out, onembed, ostride, odist, flags);

    if (this->plan_ifft == nullptr) {
      throw RuntimeError("Plan failed");
    }
    fftw_free(r_work_space);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  FFTWEngine::~FFTWEngine() noexcept {
    if (this->plan_fft != nullptr)
      fftw_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr)
      fftw_destroy_plan(this->plan_ifft);
    // TODO(Till): We cannot run fftw_cleanup since subsequent FFTW calls will
    // fail but multiple FFT engines can be active at the same time.
    // fftw_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  typename FFTWEngine::Workspace_t & FFTWEngine::fft(Field_t & field) {
    if (this->plan_fft == nullptr) {
      throw RuntimeError("fft plan not initialised");
    }
    if (static_cast<size_t>(field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error{};
      error << "The number of pixels of the field '" << field.get_name()
            << "' passed to the forward FFT is " << field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by FFTWEngine.";
      throw std::runtime_error(error.str());
    }
    if (field.get_nb_dof_per_quad_pt() * field.get_nb_quad_pts() !=
        this->get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The field reports " << field.get_nb_dof_per_quad_pt() << " "
            << "components per quadrature point and " << field.get_nb_quad_pts()
            << " quadrature points, while this FFT engine was set up to handle "
            << this->get_nb_dof_per_pixel() << " DOFs per pixel.";
      throw RuntimeError(error.str());
    }
    fftw_execute_dft_r2c(this->plan_fft, field.data(),
                         reinterpret_cast<fftw_complex *>(this->work.data()));
    return this->work;
  }

  /* ---------------------------------------------------------------------- */
  void FFTWEngine::ifft(Field_t & field) const {
    if (this->plan_ifft == nullptr) {
      throw RuntimeError("ifft plan not initialised");
    }
    if (static_cast<size_t>(field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << "The number of pixels of the field '" << field.get_name()
            << "' passed to the inverse FFT is " << field.get_nb_pixels()
            << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by FFTWEngine.";
      throw std::runtime_error(error.str());
    }
    if (field.get_nb_dof_per_quad_pt() * field.get_nb_quad_pts() !=
        this->get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The field reports " << field.get_nb_dof_per_quad_pt() << " "
            << "components per quadrature point and " << field.get_nb_quad_pts()
            << " quadrature points, while this FFT engine was set up to handle "
            << this->get_nb_dof_per_pixel() << " DOFs per pixel.";
      throw RuntimeError(error.str());
    }
    fftw_execute_dft_c2r(this->plan_ifft,
                         reinterpret_cast<fftw_complex *>(this->work.data()),
                         field.data());
  }

  std::unique_ptr<FFTEngineBase> FFTWEngine::clone() const {
    return std::make_unique<FFTWEngine>(this->get_nb_domain_grid_pts(),
                                        this->get_nb_dof_per_pixel(),
                                        this->get_communicator());
  }

}  // namespace muFFT
