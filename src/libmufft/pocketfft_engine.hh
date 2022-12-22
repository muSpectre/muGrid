/**
 * @file   pocketfft_engine.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   20 Nov 2022
 *
 * @brief  FFT engine using PocketFFT
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

#ifndef SRC_LIBMUFFT_POCKETFFT_ENGINE_HH_
#define SRC_LIBMUFFT_POCKETFFT_ENGINE_HH_

#include "fft_engine_base.hh"

namespace muFFT {

  /**
   * implements the `muFFT::FftEngine_Base` interface using
   * PocketFFT (that is shipped as part of this code).
   * See pocketfft/LICENSE.md for PocketFFT's license.
   */
  class PocketFFTEngine : public FFTEngineBase {
   public:
    using Parent = FFTEngineBase;  //!< base class
    //! field for Fourier transform of second-order tensor
    using FourierField_t = typename Parent::FourierField_t;
    //! real-valued second-order tensor
    using RealField_t = typename Parent::RealField_t;
    //! Default constructor
    PocketFFTEngine() = delete;

    /**
     * Constructor with the domain's number of grid points in each direction,
     * the communicator, and fft planner flags
     * @param nb_grid_pts number of grid points of the global grid
     * @param allow_temporary_buffer allow the creation of temporary buffers
     *        if the input buffer has the wrong memory layout
     * @param allow_destroy_input allow that the input buffers are invalidated
     *        during the FFT
     * @comm MPI communicator object
     */
    PocketFFTEngine(const DynCcoord_t & nb_grid_pts,
                    Communicator comm = Communicator(),
                    const FFT_PlanFlags & plan_flags = FFT_PlanFlags::estimate,
                    bool allow_temporary_buffer = true,
                    bool allow_destroy_input = false);
    /**
     * Constructor with the domain's number of grid points in each direction and
     * the fft planner flags
     * @param nb_grid_pts number of grid points of the global grid
     * @param allow_temporary_buffer allow the creation of temporary buffers
     *        if the input buffer has the wrong memory layout
     * @param allow_destroy_input allow that the input buffers are invalidated
     *        during the FFT
     * @comm MPI communicator object
     */
    PocketFFTEngine(const DynCcoord_t & nb_grid_pts,
                    const FFT_PlanFlags & plan_flags,
                    bool allow_temporary_buffer = true,
                    bool allow_destroy_input = false);

    //! Copy constructor
    PocketFFTEngine(const PocketFFTEngine & other) = delete;

    //! Move constructor
    PocketFFTEngine(PocketFFTEngine && other) = delete;

    //! Destructor
    virtual ~PocketFFTEngine() noexcept;

    //! Copy assignment operator
    PocketFFTEngine & operator=(const PocketFFTEngine & other) = delete;

    //! Move assignment operator
    PocketFFTEngine & operator=(PocketFFTEngine && other) = delete;

    // compute the plan, etc
    void create_plan(const Index_t & nb_dof_per_pixel) override;

    //! perform a deep copy of the engine (this should never be necessary in
    //! c++)
    std::unique_ptr<FFTEngineBase> clone() const final;

   protected:
    //! forward transform
    void compute_fft(const RealField_t & input_field,
                     FourierField_t & output_field) override;

    //! inverse transform
    void compute_ifft(const FourierField_t & input_field,
                      RealField_t & output_field) override;
  };

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_POCKETFFT_ENGINE_HH_
