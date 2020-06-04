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

#include "fftwmpi_engine.hh"

#include "fft_utils.hh"

using muGrid::RuntimeError;

namespace muFFT {

  int FFTWMPIEngine::nb_engines{0};

  FFTWMPIEngine::FFTWMPIEngine(DynCcoord_t nb_grid_pts, Dim_t nb_dof_per_pixel,
                               Communicator comm)
      : Parent{nb_grid_pts, nb_dof_per_pixel, comm}, plan_fft{nullptr},
        plan_ifft{nullptr}, real_workspace{nullptr} {
    if (!this->nb_engines)
      fftw_mpi_init();
    this->nb_engines++;

    int dim = this->nb_fourier_grid_pts.get_dim();
    std::vector<ptrdiff_t> narr(dim);
    for (Dim_t i = 0; i < dim; ++i) {
      narr[i] = this->nb_fourier_grid_pts[dim - 1 - i];
    }
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    ptrdiff_t res0{}, loc0{}, res1{}, loc1{};
    this->workspace_size = fftw_mpi_local_size_many_transposed(
        dim, narr.data(), this->nb_dof_per_pixel, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, this->comm.get_mpi_comm(), &res0, &loc0, &res1,
        &loc1);
    // A factor of two is required because we are using the c2r/r2c DFTs.
    // See:
    // http://www.fftw.org/fftw3_doc/Multi_002ddimensional-MPI-DFTs-of-Real-Data.html
    this->workspace_size *= 2;
    // res0, loc0 describe the decomposition of the first dimension of the input
    // data. Since FFTW expect row-major grid but muFFT uses column-major grids,
    // this is actually the last dimension.
    this->nb_subdomain_grid_pts[dim - 1] = res0;
    this->subdomain_locations[dim - 1] = loc0;
    // res1, loc1 describe the decomposition of the second dimension of the
    // output data. (Second since we are using transposed output.)
    if (dim > 1) {
      this->nb_fourier_grid_pts[dim - 2] = res1;
      this->fourier_locations[dim - 2] = loc1;
    } else {
      this->nb_fourier_grid_pts[dim - 1] = res1;
      this->fourier_locations[dim - 1] = loc1;
    }
    // Set the strides for the real space domain. The real space data needs to
    // contain an additional padding region if the size if odd numbered. See
    // http://www.fftw.org/fftw3_doc/Multi_002ddimensional-MPI-DFTs-of-Real-Data.html
    // Note: This information is presently not used
    if (dim > 1) {
      this->subdomain_strides[1] = 2 * this->nb_fourier_grid_pts[0];
      for (Dim_t i = 2; i < dim; ++i) {
        this->subdomain_strides[i] =
            this->subdomain_strides[i - 1] * this->nb_subdomain_grid_pts[i - 1];
      }
    }
    // Set the strides for the Fourier domain. Since we are omitted the last
    // transpose, these strides reflect that the grid in the Fourier domain is
    // transposed. This transpose operation is transparently handled by the
    // Pixels class, which iterates consecutively of grid locations but returns
    // the proper (transposed or untransposed) coordinates.
    if (dim > 1) {
      // cumulative strides over first dimensions
      Dim_t s = this->fourier_strides[dim - 2];
      this->fourier_strides[dim - 1] = s;
      this->fourier_strides[dim - 2] = this->nb_fourier_grid_pts[dim - 1] * s;
    }

    for (auto & n : this->nb_subdomain_grid_pts) {
      if (n == 0) {
        this->active = false;
      }
    }
    for (auto & n : this->nb_fourier_grid_pts) {
      if (n == 0) {
        this->active = false;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void FFTWMPIEngine::initialise(FFT_PlanFlags plan_flags) {
    if (this->initialised) {
      throw RuntimeError("double initialisation, will leak memory");
    }

    /*
     * Initialise parent after the local number of grid points in each direction
     * have been determined and work space has been initialized
     */
    Parent::initialise(plan_flags);

    if (not this->is_active()) {
      return;
    }

    this->real_workspace = fftw_alloc_real(this->workspace_size);
    /*
     * We need to check whether the workspace provided by our field is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    if (static_cast<int>(this->fourier_field.size() * this->nb_dof_per_pixel) <
        this->workspace_size) {
      this->fourier_field.set_pad_size(this->workspace_size -
                                       this->nb_dof_per_pixel *
                                           this->fourier_field.size());
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
      throw RuntimeError("unknown planner flag type");
      break;
    }

    int dim = this->nb_domain_grid_pts.get_dim();
    std::vector<ptrdiff_t> narr(dim);
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    for (Dim_t i = 0; i < dim; ++i) {
      narr[i] = this->nb_domain_grid_pts[dim - 1 - i];
    }
    Real * in{this->real_workspace};
    fftw_complex * out{
        reinterpret_cast<fftw_complex *>(this->fourier_field.data())};
    this->plan_fft = fftw_mpi_plan_many_dft_r2c(
        dim, narr.data(), this->nb_dof_per_pixel, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, in, out, this->comm.get_mpi_comm(),
        FFTW_MPI_TRANSPOSED_OUT | flags);
    if (this->plan_fft == nullptr) {
      if (dim == 1) {
        throw RuntimeError("r2c plan failed; MPI parallel FFTW does not "
                           "support 1D r2c FFTs");
      } else {
        std::stringstream message{};
        message << "Rank " << this->comm.rank() << ": r2c plan failed. "
                << "nb_subdomain_grid_pts = "
                << this->get_nb_subdomain_grid_pts()
                << ", nb_domain_grid_pts = " << this->get_nb_domain_grid_pts();
        throw RuntimeError{message.str()};
      }
    }

    fftw_complex * i_in =
        reinterpret_cast<fftw_complex *>(this->fourier_field.data());
    Real * i_out = this->real_workspace;

    this->plan_ifft = fftw_mpi_plan_many_dft_c2r(
        dim, narr.data(), this->nb_dof_per_pixel, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, i_in, i_out, this->comm.get_mpi_comm(),
        FFTW_MPI_TRANSPOSED_IN | flags);
    if (this->plan_ifft == nullptr) {
      if (dim == 1) {
        throw RuntimeError("c2r plan failed; MPI parallel FFTW does not "
                           "support 1D c2r FFTs");
      } else {
        throw RuntimeError("c2r plan failed");
      }
    }
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  FFTWMPIEngine::~FFTWMPIEngine() noexcept {
    if (this->real_workspace != nullptr)
      fftw_free(this->real_workspace);
    if (this->plan_fft != nullptr)
      fftw_destroy_plan(this->plan_fft);
    if (this->plan_ifft != nullptr)
      fftw_destroy_plan(this->plan_ifft);
    // TODO(junge): We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    // this->nb_engines--;
    // if (!this->nb_engines) fftw_mpi_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  typename FFTWMPIEngine::FourierField_t &
  FFTWMPIEngine::fft(RealField_t & field) {
    if (not this->is_active()) {
      return this->fourier_field;
    }

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
            << " of the (sub)domain handled by FFTWMPIEngine.";
      throw RuntimeError(error.str());
    }
    if (field.get_nb_dof_per_sub_pt() * field.get_nb_sub_pts() !=
        this->get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The field reports " << field.get_nb_dof_per_sub_pt() << " "
            << "components per quadrature point and " << field.get_nb_sub_pts()
            << " quadrature points, while this FFT engine was set up to handle "
            << this->get_nb_dof_per_pixel() << " DOFs per pixel.";
      throw RuntimeError(error.str());
    }
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride =
        (this->nb_dof_per_pixel * this->nb_subdomain_grid_pts[0]);
    ptrdiff_t wstride =
        (this->nb_dof_per_pixel * 2 * (this->nb_subdomain_grid_pts[0] / 2 + 1));
    ptrdiff_t n = field.size() / this->nb_subdomain_grid_pts[0];

    auto fdata = field.data();
    auto wdata = this->real_workspace;
    for (int i = 0; i < n; i++) {
      std::copy(fdata, fdata + fstride, wdata);
      fdata += fstride;
      wdata += wstride;
    }
    // Compute FFT
    fftw_mpi_execute_dft_r2c(
        this->plan_fft, this->real_workspace,
        reinterpret_cast<fftw_complex *>(this->fourier_field.data()));
    return this->fourier_field;
  }

  /* ---------------------------------------------------------------------- */
  void FFTWMPIEngine::ifft(RealField_t & field) const {
    if (not this->is_active()) {
      return;
    }
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
      throw RuntimeError(error.str());
    }
    if (field.get_nb_dof_per_sub_pt() * field.get_nb_sub_pts() !=
        this->get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "The field reports " << field.get_nb_dof_per_sub_pt() << " "
            << "components per quadrature point and " << field.get_nb_sub_pts()
            << " quadrature points, while this FFT engine was set up to handle "
            << this->get_nb_dof_per_pixel() << " DOFs per pixel.";
      throw RuntimeError(error.str());
    }
    // Compute inverse FFT
    fftw_mpi_execute_dft_c2r(
        this->plan_ifft,
        reinterpret_cast<fftw_complex *>(this->fourier_field.data()),
        this->real_workspace);
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride{this->nb_dof_per_pixel * this->nb_subdomain_grid_pts[0]};
    ptrdiff_t wstride{this->nb_dof_per_pixel * 2 *
                      (this->nb_subdomain_grid_pts[0] / 2 + 1)};
    ptrdiff_t n(field.size() / this->nb_subdomain_grid_pts[0]);

    auto fdata{field.data()};
    auto wdata{this->real_workspace};
    for (int i = 0; i < n; i++) {
      std::copy(wdata, wdata + fstride, fdata);
      fdata += fstride;
      wdata += wstride;
    }
  }

  std::unique_ptr<FFTEngineBase> FFTWMPIEngine::clone() const {
    return std::make_unique<FFTWMPIEngine>(this->get_nb_domain_grid_pts(),
                                           this->get_nb_dof_per_pixel(),
                                           this->get_communicator());
  }

}  // namespace muFFT
