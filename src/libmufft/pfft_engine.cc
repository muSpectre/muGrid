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

#include "pfft_engine.hh"

using muGrid::CcoordOps::get_col_major_strides;

namespace muFFT {

  int PFFTEngine::nb_engines{0};

  PFFTEngine::PFFTEngine(const DynCcoord_t & nb_grid_pts,
                         Communicator comm,
                         const FFT_PlanFlags & plan_flags,
                         bool allow_temporary_buffer,
                         bool allow_destroy_input)
      : Parent{nb_grid_pts, comm, plan_flags, allow_temporary_buffer,
               allow_destroy_input}, mpi_comm{comm.get_mpi_comm()} {
    if (!this->nb_engines)
      pfft_init();
    this->nb_engines++;

    int size{comm.size()};
    int dim_x{size};
    int dim_y{1};
    const int dim{this->nb_fourier_grid_pts.get_dim()};

    // Determine process mesh for pencil decomposition.
    if (dim > 2) {
      dim_y = static_cast<int>(sqrt(size));
      while ((size / dim_y) * dim_y != size)
        dim_y--;
      dim_x = size / dim_y;
    }

    if (dim > 2) {
      if (pfft_create_procmesh_2d(this->comm.get_mpi_comm(), dim_x, dim_y,
                                  &this->mpi_comm)) {
        throw FFTEngineError("Failed to create 2d PFFT process mesh.");
      }
    }

    std::vector<ptrdiff_t> narr(dim);
    for (Dim_t i = 0; i < dim; ++i) {
      narr[i] = this->nb_domain_grid_pts[dim - 1 - i];
    }
    const int nb_dof{1};
    std::vector<ptrdiff_t> res(dim), loc(dim), fres(dim), floc(dim);
    pfft_local_size_many_dft_r2c(
        dim, narr.data(), narr.data(), narr.data(), nb_dof,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, this->mpi_comm,
        PFFT_PADDED_R2C | PFFT_TRANSPOSED_OUT, res.data(), loc.data(),
        fres.data(), floc.data());
    for (Dim_t i = 0; i < dim; ++i) {
      this->nb_subdomain_grid_pts[dim-1-i] = res[i];
      this->subdomain_locations[dim-1-i] = loc[i];
      this->nb_fourier_grid_pts[dim-1-i] = fres[i];
      this->fourier_locations[dim-1-i] = floc[i];
    }

    // Set the strides for the real domain which is padded.
    this->subdomain_strides = get_col_major_strides(
        this->nb_subdomain_grid_pts);

    // Set the strides for the Fourier domain (which is not padded but
    // transposed). Since we are omitting the last transpose, these strides
    // reflect that the grid in the Fourier domain is transposed. This
    // transpose operation is transparently handled by the Pixels class, which
    // iterates consecutively of grid locations but returns the proper
    // (transposed or untransposed) coordinates.
    if (dim > 1) {
      // cumulative strides over first dimensions
      this->fourier_strides[dim - 1] = 1;
      this->fourier_strides[0] = this->nb_fourier_grid_pts[dim - 1];
      for (Dim_t i = 1; i < dim - 1; ++i) {
        this->fourier_strides[i] = this->fourier_strides[i - 1] *
                                   this->nb_fourier_grid_pts[i - 1];
      }
    }

    // pfft_local_size_many_dft_r2c returns the *padded* size, not the real size
    this->nb_subdomain_grid_pts[0] = this->nb_domain_grid_pts[0];

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
    this->real_field_collection.initialise(
        this->nb_domain_grid_pts, this->nb_subdomain_grid_pts,
        this->subdomain_locations, this->subdomain_strides);
    this->fourier_field_collection.initialise(
        this->nb_domain_grid_pts, this->nb_fourier_grid_pts,
        this->fourier_locations, this->fourier_strides);
  }

  /* ---------------------------------------------------------------------- */
  void PFFTEngine::create_plan(const Index_t & nb_dof_per_pixel) {
    if (this->has_plan_for(nb_dof_per_pixel)) {
      // plan already exists, we can bail
      return;
    }

    const int dim{this->nb_fourier_grid_pts.get_dim()};

    // Reverse the order of the array dimensions, because FFTW expects a
    // row-major array and the arrays used in muSpectre are column-major
    std::vector<ptrdiff_t> narr(dim);
    for (Dim_t i = 0; i < dim; ++i) {
      narr[i] = this->nb_domain_grid_pts[dim - 1 - i];
    }
    int howmany{static_cast<int>(nb_dof_per_pixel)};
    std::vector<ptrdiff_t> res(dim), loc(dim), fres(dim), floc(dim);
    auto required_workspace_size{pfft_local_size_many_dft_r2c(
        dim, narr.data(), narr.data(), narr.data(), howmany,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, this->mpi_comm,
        PFFT_PADDED_R2C | PFFT_TRANSPOSED_OUT, res.data(), loc.data(),
        fres.data(), floc.data())};

    required_workspace_size *= 2;

    this->required_workspace_sizes[nb_dof_per_pixel] = required_workspace_size;

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
      throw FFTEngineError("unknown planner flag type");
      break;
    }

    Real * in{pfft_alloc_real(required_workspace_size)};
    if (in == nullptr) {
      throw FFTEngineError("'in' allocation failed");
    }
    pfft_complex * out{pfft_alloc_complex(required_workspace_size / 2)};
    if (out == nullptr) {
      throw FFTEngineError("'out' allocation failed");
    }

    this->fft_plans[nb_dof_per_pixel] = pfft_plan_many_dft_r2c(
        dim, narr.data(), narr.data(), narr.data(), howmany,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, in, out, this->mpi_comm,
        PFFT_FORWARD, PFFT_PADDED_R2C | PFFT_TRANSPOSED_OUT | flags);
    if (this->fft_plans[nb_dof_per_pixel] == nullptr) {
      throw FFTEngineError("r2c plan failed");
    }

    pfft_complex * i_in{out};
    Real * i_out{in};

    this->ifft_plans[nb_dof_per_pixel] = pfft_plan_many_dft_c2r(
        dim, narr.data(), narr.data(), narr.data(), howmany,
        PFFT_DEFAULT_BLOCKS, PFFT_DEFAULT_BLOCKS, i_in, i_out, this->mpi_comm,
        PFFT_BACKWARD, PFFT_PADDED_R2C | PFFT_TRANSPOSED_IN | flags);
    if (this->ifft_plans[nb_dof_per_pixel] == nullptr) {
      throw FFTEngineError("c2r plan failed");
    }

    pfft_free(in);
    pfft_free(out);
    this->planned_nb_dofs.insert(nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  PFFTEngine::~PFFTEngine() noexcept {
    for (auto && nb_dof_per_pixel : this->planned_nb_dofs) {
      if (this->fft_plans[nb_dof_per_pixel] != nullptr)
        pfft_destroy_plan(this->fft_plans[nb_dof_per_pixel]);
      if (this->ifft_plans[nb_dof_per_pixel] != nullptr)
        pfft_destroy_plan(this->ifft_plans[nb_dof_per_pixel]);
    }
    if (this->mpi_comm != this->comm.get_mpi_comm()) {
      MPI_Comm_free(&this->mpi_comm);
    }
    // TODO(Till): We cannot run fftw_mpi_cleanup since also calls fftw_cleanup
    // and any running FFTWEngine will fail afterwards.
    // this->nb_engines--;
    // if (!this->nb_engines) pfft_cleanup();
  }

  /* ---------------------------------------------------------------------- */
  void PFFTEngine::compute_fft(const RealField_t & input_field,
                               FourierField_t & output_field) const {
    // Compute FFT
    pfft_execute_dft_r2c(
        this->fft_plans.at(input_field.get_nb_dof_per_pixel()),
        input_field.data(),
        reinterpret_cast<pfft_complex *>(output_field.data()));
  }

  /* ---------------------------------------------------------------------- */
  void PFFTEngine::compute_ifft(const FourierField_t & input_field,
                                   RealField_t & output_field) const {
    pfft_execute_dft_c2r(
        this->ifft_plans.at(input_field.get_nb_dof_per_pixel()),
        reinterpret_cast<pfft_complex *>(input_field.data()),
        output_field.data());
  }

  /* ---------------------------------------------------------------------- */
  std::unique_ptr<FFTEngineBase> PFFTEngine::clone() const {
    return std::make_unique<PFFTEngine>(
        this->get_nb_domain_grid_pts(), this->get_communicator(),
        this->plan_flags, this->allow_temporary_buffer,
        this->allow_destroy_input);
  }

  /* ---------------------------------------------------------------------- */
  auto
  PFFTEngine::register_real_space_field(const std::string & unique_name,
                                           const Index_t & nb_dof_per_pixel)
  -> RealField_t & {
    this->create_plan(nb_dof_per_pixel);
    auto & field{
        Parent::register_real_space_field(unique_name, nb_dof_per_pixel)};
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto && required_workspace_size{
        2 * this->required_workspace_sizes.at(nb_dof_per_pixel)};
    field.set_pad_size(required_workspace_size -
                       nb_dof_per_pixel * field.get_nb_buffer_pixels());
    return field;
  }

  /* ---------------------------------------------------------------------- */
  auto PFFTEngine::register_real_space_field(const std::string & unique_name,
                                                const Shape_t & shape)
  -> RealField_t & {
    auto & field{Parent::register_real_space_field(unique_name, shape)};
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto nb_dof_per_pixel{std::accumulate(shape.begin(), shape.end(), 1,
                                          std::multiplies<Index_t>())};
    auto && required_workspace_size{
        2 * this->required_workspace_sizes.at(nb_dof_per_pixel)};
    field.set_pad_size(required_workspace_size -
                       nb_dof_per_pixel * field.get_nb_buffer_pixels());
    return field;
  }

  /* ---------------------------------------------------------------------- */
  muGrid::ComplexField & PFFTEngine::register_fourier_space_field(
      const std::string & unique_name, const Index_t & nb_dof_per_pixel) {
    auto & field{
        Parent::register_fourier_space_field(unique_name, nb_dof_per_pixel)};
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    if (static_cast<int>(field.get_nb_entries() * nb_dof_per_pixel) <
        required_workspace_size) {
      auto pad_size{required_workspace_size -
                    nb_dof_per_pixel * field.get_nb_buffer_pixels()};
      field.set_pad_size(std::max(0L, pad_size));
    }
    return field;
  }

  /* ---------------------------------------------------------------------- */
  muGrid::ComplexField &
  PFFTEngine::register_fourier_space_field(const std::string & unique_name,
                                              const Shape_t & shape) {
    auto & field{Parent::register_fourier_space_field(unique_name, shape)};
    /*
     * We need to check whether the fourier field provided is large
     * enough. MPI parallel FFTW may request a workspace size larger than the
     * nominal size of the complex buffer.
     */
    auto nb_dof_per_pixel{std::accumulate(shape.begin(), shape.end(), 1,
                                          std::multiplies<Index_t>())};
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    if (static_cast<int>(field.get_nb_entries() * nb_dof_per_pixel) <
        required_workspace_size) {
      auto pad_size{required_workspace_size -
                    nb_dof_per_pixel * field.get_nb_buffer_pixels()};
      field.set_pad_size(std::max(0L, pad_size));
    }
    return field;
  }

  /* ---------------------------------------------------------------------- */
  bool PFFTEngine::check_real_space_field(const RealField_t & field) const {
    auto nb_dof_per_pixel{field.get_nb_dof_per_pixel()};
    auto && required_workspace_size{
        2 * this->required_workspace_sizes.at(nb_dof_per_pixel)};
    auto required_pad_size{required_workspace_size -
                           nb_dof_per_pixel * field.get_nb_buffer_pixels()};
    return Parent::check_real_space_field(field) and
           static_cast<Index_t>(field.get_pad_size()) >= required_pad_size;
  }

  /* ---------------------------------------------------------------------- */
  bool
  PFFTEngine::check_fourier_space_field(const FourierField_t & field) const {
    auto nb_dof_per_pixel{field.get_nb_dof_per_pixel()};
    auto && required_workspace_size{
        this->required_workspace_sizes.at(nb_dof_per_pixel)};
    auto required_pad_size{required_workspace_size -
                           nb_dof_per_pixel * field.get_nb_buffer_pixels()};
    return Parent::check_fourier_space_field(field) and
           static_cast<Index_t>(field.get_pad_size()) >= required_pad_size;
  }

}  // namespace muFFT
