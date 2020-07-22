/**
 * @file   fftwmpi_engine.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   06 Mar 2017
 *
 * @brief  FFT engine using MPI-parallel FFTW
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

#ifndef SRC_LIBMUFFT_FFTWMPI_ENGINE_HH_
#define SRC_LIBMUFFT_FFTWMPI_ENGINE_HH_

#include "fft_engine_base.hh"

#include <fftw3-mpi.h>

namespace muFFT {

  /**
   * implements the `muFFT::FFTEngineBase` interface using the
   * FFTW library
   */
  class FFTWMPIEngine : public FFTEngineBase {
   public:
    using Parent = FFTEngineBase;  //!< base class
    //! field for Fourier transform of second-order tensor
    using FourierField_t = typename Parent::FourierField_t;
    //! real-valued second-order tensor
    using RealField_t = typename Parent::RealField_t;
    //! Default constructor
    FFTWMPIEngine() = delete;

    /**
     * Constructor with the domain's number of grid points in each direction and
     * the communicator
     * @param nb_grid_pts number of grid points of the global grid
     * @param allow_temporary_buffer allow the creation of temporary buffers
     *        if the input buffer has the wrong memory layout
     * @param allow_destroy_input allow that the input buffers are invalidated
     *        during the FFT
     * @comm MPI communicator object
     */
    FFTWMPIEngine(const DynCcoord_t & nb_grid_pts,
                  Communicator comm = Communicator(),
                  const FFT_PlanFlags & plan_flags = FFT_PlanFlags::estimate,
                  bool allow_temporary_buffer = true,
                  bool allow_destroy_input = false);

    //! Copy constructor
    FFTWMPIEngine(const FFTWMPIEngine & other) = delete;

    //! Move constructor
    FFTWMPIEngine(FFTWMPIEngine && other) = delete;

    //! Destructor
    virtual ~FFTWMPIEngine() noexcept;

    //! Copy assignment operator
    FFTWMPIEngine & operator=(const FFTWMPIEngine & other) = delete;

    //! Move assignment operator
    FFTWMPIEngine & operator=(FFTWMPIEngine && other) = delete;

    // compute the plan, etc
    void create_plan(const Index_t & nb_dof_per_pixel) override;

    /**
     * return whether this engine is active (an engine is active if it has more
     * than zero grid points. FFTWMPI sometimes assigns zero grid points)
     */
    bool is_active() const override { return this->active; }

    //! perform a deep copy of the engine (this should never be necessary in
    //! c++)
    std::unique_ptr<FFTEngineBase> clone() const final;

    /**
     * need to override this method here, since FFTWMPI requires field padding
     */
    RealField_t &
    register_real_space_field(const std::string & unique_name,
                              const Index_t & nb_dof_per_pixel) final;

    /**
     * need to override this method here, since FFTWMPI requires field padding
     */
    RealField_t &
    register_real_space_field(const std::string & unique_name,
                              const Shape_t & shape) final;

    /**
     * need to override this method here, since FFTWMPI requires field padding
     */
    FourierField_t &
    register_fourier_space_field(const std::string & unique_name,
                                 const Index_t & nb_dof_per_pixel) final;

    /**
     * need to override this method here, since FFTWMPI requires field padding
     */
    FourierField_t &
    register_fourier_space_field(const std::string & unique_name,
                                 const Shape_t & shape) final;

   protected:
    //! forward transform
    void compute_fft(const RealField_t & field,
                     FourierField_t & output_field) const override;

    //! inverse transform
    void compute_ifft(const FourierField_t & input_field,
                      RealField_t & output_field) const override;

    //! check whether real-space buffer has the correct memory layout
    bool check_real_space_field(const RealField_t & field) const final;

    //! check whether Fourier-space buffer has the correct memory layout
    bool check_fourier_space_field(const FourierField_t & field) const final;

    static int nb_engines;  //!< number of times this engine has
                            //!< been instatiated
    //! holds the plans for forward fourier transforms
    std::map<Index_t, fftw_plan> fft_plans{};
    //! holds the plans for inversefourier transforms
    std::map<Index_t, fftw_plan> ifft_plans{};
    //! holds the fourier field sizes including padding for different transforms
    std::map<Index_t, Index_t> required_workspace_sizes{};
    bool active{true};  //!< FFTWMPI sometimes assigns zero grid points
    //! Input to local_size_many_transposed
    std::vector<ptrdiff_t> nb_fourier_non_transposed{};
  };
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFTWMPI_ENGINE_HH_
