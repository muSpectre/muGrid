/**
 * @file   pfft_engine.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   06 Mar 2017
 *
 * @brief  FFT engine using MPI-parallel PFFT
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

#ifndef SRC_LIBMUFFT_PFFT_ENGINE_HH_
#define SRC_LIBMUFFT_PFFT_ENGINE_HH_

#include "fft_engine_base.hh"

#include <pfft.h>

namespace muFFT {

  /**
   * implements the `muFFT::FFTEngineBase` interface using the
   * FFTW library
   */
  template <Dim_t DimS>
  class PFFTEngine : public FFTEngineBase<DimS> {
   public:
    using Parent = FFTEngineBase<DimS>;  //!< base class
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    //! field for Fourier transform of second-order tensor
    using Workspace_t = typename Parent::Workspace_t;
    //! real-valued second-order tensor
    using Field_t = typename Parent::Field_t;
    //! Default constructor
    PFFTEngine() = delete;

    /**
     * Constructor with the domain's number of grid points in each direciton,
     * the number of components to transform, and the communicator
     */
    PFFTEngine(Ccoord nb_grid_pts, Dim_t nb_dof_per_pixel,
               Communicator comm = Communicator());

    //! Copy constructor
    PFFTEngine(const PFFTEngine & other) = delete;

    //! Move constructor
    PFFTEngine(PFFTEngine && other) = default;

    //! Destructor
    virtual ~PFFTEngine() noexcept;

    //! Copy assignment operator
    PFFTEngine & operator=(const PFFTEngine & other) = delete;

    //! Move assignment operator
    PFFTEngine & operator=(PFFTEngine && other) = default;

    // compute the plan, etc
    void initialise(FFT_PlanFlags plan_flags) override;

    //! forward transform
    Workspace_t & fft(Field_t & field) override;

    //! inverse transform
    void ifft(Field_t & field) const override;

   protected:
    MPI_Comm mpi_comm;  //!< MPI communicator
    static int
        nb_engines;        //!< number of times this engine has been instatiated
    pfft_plan plan_fft{};  //!< holds the plan for forward fourier transform
    pfft_plan plan_ifft{};  //!< holds the plan for inverse fourier transform
    ptrdiff_t
        workspace_size{};     //!< size of workspace buffer returned by planner
    Real * real_workspace{};  //!< temporary real workspace that is correctly
                              //!< padded
  };

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_PFFT_ENGINE_HH_
