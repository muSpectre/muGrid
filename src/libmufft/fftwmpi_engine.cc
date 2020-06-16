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

namespace muFFT {

  int FFTWMPIEngine::nb_engines{0};

  FFTWMPIEngine::FFTWMPIEngine(const DynCcoord_t & nb_grid_pts,
                               Communicator comm)
      : Parent{nb_grid_pts, comm} {
    if (!this->nb_engines) {
      fftw_mpi_init();
    }
    this->nb_engines++;

    const int dim{this->nb_fourier_grid_pts.get_dim()};
    this->nb_fourier_non_transposed.resize(dim);
    for (Index_t i = 0; i < dim; ++i) {
      this->nb_fourier_non_transposed[i] =
          this->nb_fourier_grid_pts[dim - 1 - i];
    }
    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    const int nb_dof{1};
    ptrdiff_t res0{}, loc0{}, res1{}, loc1{};
    this->workspace_size = fftw_mpi_local_size_many_transposed(
        dim, this->nb_fourier_non_transposed.data(), nb_dof,
        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
        this->comm.get_mpi_comm(), &res0, &loc0, &res1, &loc1);
    // A factor of two is required because we are using the c2r/r2c DFTs.
    // See:
    // http://www.fftw.org/fftw3_doc/Multi_002ddimensional-MPI-DFTs-of-Real-Data.html
    this->workspace_size *= 2;
    // Allocate workspace buffer.
    this->real_workspace = fftw_alloc_real(this->workspace_size);
    if (this->real_workspace == nullptr) {
      throw FFTEngineError("allocation of workspace buffer failed");
    }
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
      for (Index_t i = 2; i < dim; ++i) {
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
      Index_t s = this->fourier_strides[dim - 2];
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
    this->fourier_field_collection.initialise(this->nb_fourier_grid_pts,
                                              this->fourier_locations,
                                              this->fourier_strides);
  }

  /* ---------------------------------------------------------------------- */
  void FFTWMPIEngine::initialise(const Index_t & nb_dof_per_pixel,
                                 const FFT_PlanFlags & plan_flags) {
    if (this->has_plan_for(nb_dof_per_pixel)) {
      // plan already exists, we can bail
      return;
    }

    const int dim{this->nb_fourier_grid_pts.get_dim()};


    if (not this->is_active()) {
      return;
    }

    int howmany{static_cast<int>(nb_dof_per_pixel)};
    ptrdiff_t res0{}, loc0{}, res1{}, loc1{};
    // find how large a workspace this transform needs
    // this needs the fourier grid points as input
    auto required_workspace_size{fftw_mpi_local_size_many_transposed(
        dim, this->nb_fourier_non_transposed.data(), nb_dof_per_pixel,
        FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
        this->comm.get_mpi_comm(), &res0, &loc0, &res1, &loc1)};
    // A factor of two is required because we are using the c2r/r2c DFTs.
    // See:
    // http://www.fftw.org/fftw3_doc/Multi_002ddimensional-MPI-DFTs-of-Real-Data.html


    required_workspace_size *= 2;

    this->required_workspace_sizes[nb_dof_per_pixel] = required_workspace_size;

    // check whether the current workspace is large enough
    if (this->workspace_size < required_workspace_size) {
      if (this->real_workspace != nullptr) {
        fftw_free(this->real_workspace);
      }
      this->real_workspace = fftw_alloc_real(required_workspace_size);
      if (this->real_workspace == nullptr) {
        throw FFTEngineError("allocation of workspace buffer failed");
      }
      this->workspace_size = required_workspace_size;
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

    Real * in{this->real_workspace};
    fftw_complex * out{fftw_alloc_complex(required_workspace_size/2)};

    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    std::vector<ptrdiff_t> narr(dim);
    for (Index_t i = 0; i < dim; ++i) {
      narr[i] = this->nb_domain_grid_pts[dim - 1 - i];
    }
    // this needs the domain grid points as input narr
    this->fft_plans[nb_dof_per_pixel] = fftw_mpi_plan_many_dft_r2c(
        dim, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, in, out, this->comm.get_mpi_comm(),
        FFTW_MPI_TRANSPOSED_OUT | flags);

    if (this->fft_plans[nb_dof_per_pixel] == nullptr) {
      if (dim == 1) {
        throw FFTEngineError("r2c plan failed; MPI parallel FFTW does not "
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

    fftw_complex * i_in{out};
    Real * i_out{this->real_workspace};

    this->ifft_plans[nb_dof_per_pixel] = fftw_mpi_plan_many_dft_c2r(
        dim, narr.data(), howmany, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, i_in, i_out, this->comm.get_mpi_comm(),
        FFTW_MPI_TRANSPOSED_IN | flags);
    if (this->ifft_plans[nb_dof_per_pixel] == nullptr) {
      if (dim == 1)
        throw FFTEngineError("c2r plan failed; MPI parallel FFTW does not "
                             "support 1D c2r FFTs");
      else
        throw FFTEngineError("c2r plan failed");
    }
    fftw_free(out);
    this->planned_nb_dofs.insert(nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  FFTWMPIEngine::~FFTWMPIEngine() noexcept {
    if (this->real_workspace != nullptr)
      fftw_free(this->real_workspace);
    for (auto && nb_dof_per_pixel : this->planned_nb_dofs) {
      if (this->fft_plans[nb_dof_per_pixel] != nullptr)
        fftw_destroy_plan(this->fft_plans[nb_dof_per_pixel]);
      if (this->ifft_plans[nb_dof_per_pixel] != nullptr)
        fftw_destroy_plan(this->ifft_plans[nb_dof_per_pixel]);
    }
    // TODO(junge): We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    // this->nb_engines--;
    // if (!this->nb_engines) fftw_mpi_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  void FFTWMPIEngine::fft(const RealField_t & input_field,
                          FourierField_t & output_field) const {
    if (not this->is_active()) {
      return;
    }
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (nb_dof_per_pixel != output_field.get_nb_dof_per_pixel()) {
      std::stringstream message{};
      message << "The input and output fields are not compatible: the input "
                 "field has  "
              << nb_dof_per_pixel
              << " degrees of freedom per pixel, while the output field has "
              << output_field.get_nb_dof_per_pixel();
      throw FFTEngineError{message.str()};
    }
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel. Use "
                 "`muFFT::FFTEngineBase::initialise` to prepare a plan.";
      throw FFTEngineError{message.str()};
    }
    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error{};
      error << std::endl
            << "rank " << this->get_communicator().rank()
            << ": The number of pixels of the input field '"
            << input_field.get_name() << "' passed to the forward FFT is "
            << input_field.get_nb_pixels() << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by FFTWMPIEngine.";
      throw RuntimeError(error.str());
    }
    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)) {
      std::stringstream error{};
      error << std::endl
            << "rank " << this->get_communicator().rank()
            << ": The number of pixels of the Fourier space output field '"
            << output_field.get_name() << "' passed to the forward FFT is "
            << output_field.get_nb_pixels() << " and doesn't match the size "
            << muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)
            << " of the (sub)domain handled by this FFTWMPIEngine." << std::endl
            << "Domain shape:         " << this->nb_domain_grid_pts << ","
            << std::endl
            << "Subdomain shape:      " << this->nb_subdomain_grid_pts
            << ", locations: " << this->subdomain_locations << std::endl
            << "Fourier domain shape: " << this->nb_fourier_grid_pts
            << ", locations: " << this->fourier_locations;
      throw RuntimeError(error.str());
    }
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    if (static_cast<int>(output_field.size() * nb_dof_per_pixel) <
        required_workspace_size) {
      output_field.set_pad_size(this->get_required_pad_size(nb_dof_per_pixel));
    }
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride = (nb_dof_per_pixel * this->nb_subdomain_grid_pts[0]);
    ptrdiff_t wstride =
        (nb_dof_per_pixel * 2 * (this->nb_subdomain_grid_pts[0] / 2 + 1));
    ptrdiff_t n = input_field.size() / this->nb_subdomain_grid_pts[0];

    auto fdata{input_field.data()};
    auto wdata{this->real_workspace};
    for (int i = 0; i < n; i++) {
      std::copy(fdata, fdata + fstride, wdata);
      fdata += fstride;
      wdata += wstride;
    }

    // Compute FFT
    fftw_mpi_execute_dft_r2c(
        this->fft_plans.at(nb_dof_per_pixel), this->real_workspace,
        reinterpret_cast<fftw_complex *>(output_field.data()));
  }

  /* ---------------------------------------------------------------------- */
  void FFTWMPIEngine::ifft(const FourierField_t & input_field,
                           RealField_t & output_field) const {
    if (not this->is_active()) {
      return;
    }
    auto && nb_dof_per_pixel{input_field.get_nb_dof_per_pixel()};
    if (nb_dof_per_pixel != output_field.get_nb_dof_per_pixel()) {
      std::stringstream message{};
      message << "The input and output fields are not compatible: the input "
                 "field has  "
              << nb_dof_per_pixel
              << " degrees of freedom per pixel, while the output field has "
              << output_field.get_nb_dof_per_pixel();
      throw FFTEngineError{message.str()};
    }
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel. Use "
                 "`muFFT::FFTEngineBase::initialise` to prepare a plan.";
      throw FFTEngineError{message.str()};
    }

    if (static_cast<size_t>(output_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)) {
      std::stringstream error;
      error << std::endl
            << "rank " << this->get_communicator().rank()
            << ": The number of pixels of the output_field '"
            << output_field.get_name() << "' passed to the inverse FFT is "
            << output_field.get_nb_pixels() << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by FFTWEngine.";
      throw RuntimeError(error.str());
    }

    if (static_cast<size_t>(input_field.get_nb_pixels()) !=
        muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts)) {
      std::stringstream error;
      error << std::endl
            << "rank " << this->get_communicator().rank()
            << ": The number of pixels of the Fourier space input_field '"
            << input_field.get_name() << "' passed to the inverse FFT is "
            << input_field.get_nb_pixels() << " and does not match the size "
            << muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts)
            << " of the (sub)domain handled by this FFTWEngine." << std::endl
            << "Domain shape:         " << this->nb_domain_grid_pts << ","
            << std::endl
            << "Subdomain shape:      " << this->nb_subdomain_grid_pts
            << ", locations: " << this->subdomain_locations << std::endl
            << "Fourier domain shape: " << this->nb_fourier_grid_pts
            << ", locations: " << this->fourier_locations;
      throw RuntimeError(error.str());
    }

    // Compute inverse FFT
    fftw_mpi_execute_dft_c2r(
        this->ifft_plans.at(nb_dof_per_pixel),
        reinterpret_cast<fftw_complex *>(input_field.data()),
        this->real_workspace);
    // Copy non-padded field to padded real_workspace.
    // Transposed output of M x N x L transform for >= 3 dimensions is padded
    // M x N x 2*(L/2+1).
    ptrdiff_t fstride{nb_dof_per_pixel * this->nb_subdomain_grid_pts[0]};
    ptrdiff_t wstride{nb_dof_per_pixel * 2 *
                      (this->nb_subdomain_grid_pts[0] / 2 + 1)};
    ptrdiff_t n(output_field.size() / this->nb_subdomain_grid_pts[0]);

    auto fdata{output_field.data()};
    auto wdata{this->real_workspace};
    for (int i = 0; i < n; i++) {
      std::copy(wdata, wdata + fstride, fdata);
      fdata += fstride;
      wdata += wstride;
    }
  }

  /* ---------------------------------------------------------------------- */
  std::unique_ptr<FFTEngineBase> FFTWMPIEngine::clone() const {
    return std::make_unique<FFTWMPIEngine>(this->get_nb_domain_grid_pts(),
                                           this->get_communicator());
  }

  /* ---------------------------------------------------------------------- */
  auto
  FFTWMPIEngine::register_fourier_space_field(const std::string & unique_name,
                                              const Index_t & nb_dof_per_pixel)
      -> FourierField_t & {
    if (not this->has_plan_for(nb_dof_per_pixel)) {
      std::stringstream message{};
      message << "No plan has been created for " << nb_dof_per_pixel
              << " degrees of freedom per pixel. Use "
                 "`muFFT::FFTEngineBase::initialise` to prepare a plan.";
      throw FFTEngineError{message.str()};
    }
    auto & field{
        Parent::register_fourier_space_field(unique_name, nb_dof_per_pixel)};
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    if (static_cast<int>(field.size() * nb_dof_per_pixel) <
        required_workspace_size) {
      field.set_pad_size(required_workspace_size -
                         nb_dof_per_pixel * field.size());
    }
    return field;
  }

  /* ---------------------------------------------------------------------- */
  Index_t
  FFTWMPIEngine::get_required_pad_size(const Index_t & nb_dof_per_pixel) const {
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    return required_workspace_size - nb_dof_per_pixel * this->fourier_size();
  }

}  // namespace muFFT
