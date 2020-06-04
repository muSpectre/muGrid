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
     */
    FFTWMPIEngine(const DynCcoord_t & nb_grid_pts,
                  Communicator comm = Communicator());

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
    void initialise(const Index_t & nb_dof_per_pixel,
                    const FFT_PlanFlags & plan_flags) override;

    //! forward transform
    void fft(const RealField_t & field,
             FourierField_t & output_field) const override;

    //! inverse transform
    void ifft(const FourierField_t & input_field,
              RealField_t & output_field) const override;

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
    FourierField_t &
    register_fourier_space_field(const std::string & unique_name,
                                 const Index_t & nb_dof_per_pixel) final;

    /**
     * Returns the required pad size. Helpful when calling fftwmpi with wrapped
     * fields
     */
    Index_t get_required_pad_size(const Index_t & nb_dof_per_pixel) const final;

   protected:
    static int nb_engines;  //!< number of times this engine has
                            //!< been instatiated
    //! holds the plans for forward fourier transforms
    std::map<Index_t, fftw_plan> fft_plans{};
    //! holds the plans for inversefourier transforms
    std::map<Index_t, fftw_plan> ifft_plans{};
    //! holds the fourier field sizes including padding for different transforms
    std::map<Index_t, Index_t> required_workspace_sizes{};
    //! maximum size of workspace buffer (returned by planner)
    ptrdiff_t workspace_size{0};
    //! temporary real workspace for correctly padded copy of real input
    Real * real_workspace{nullptr};
    bool active{true};  //!< FFTWMPI sometimes assigns zero grid points
    //! Input to local_size_many_transposed
    std::vector<ptrdiff_t> nb_fourier_non_transposed{};
  };
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFTWMPI_ENGINE_HH_
