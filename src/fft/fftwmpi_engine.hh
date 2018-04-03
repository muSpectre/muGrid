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
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef FFTWMPI_ENGINE_H
#define FFTWMPI_ENGINE_H

#include "fft/fft_engine_base.hh"

#include <fftw3-mpi.h>

namespace muSpectre {

  /**
   * implements the `muSpectre::FFTEngineBase` interface using the
   * FFTW library
   */
  template <Dim_t DimS, Dim_t DimM>
  class FFTWMPIEngine: public FFTEngineBase<DimS, DimM>
  {
  public:
    using Parent = FFTEngineBase<DimS, DimM>; //!< base class
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    //! field for Fourier transform of second-order tensor
    using Workspace_t = typename Parent::Workspace_t;
    //! real-valued second-order tensor
    using Field_t = typename Parent::Field_t;
    //! Default constructor
    FFTWMPIEngine() = delete;

    //! Constructor with system resolutions
    FFTWMPIEngine(Ccoord resolutions, Communicator comm=Communicator());

    //! Copy constructor
    FFTWMPIEngine(const FFTWMPIEngine &other) = delete;

    //! Move constructor
    FFTWMPIEngine(FFTWMPIEngine &&other) = default;

    //! Destructor
    virtual ~FFTWMPIEngine() noexcept;

    //! Copy assignment operator
    FFTWMPIEngine& operator=(const FFTWMPIEngine &other) = delete;

    //! Move assignment operator
    FFTWMPIEngine& operator=(FFTWMPIEngine &&other) = default;

    // compute the plan, etc
    virtual void initialise(FFT_PlanFlags plan_flags) override;

    //! forward transform
    virtual Workspace_t & fft(Field_t & field) override;

    //! inverse transform
    virtual void ifft(Field_t & field) const override;

  protected:
    static int nb_engines; //!< number of times this engine has been instatiated
    fftw_plan plan_fft{}; //!< holds the plan for forward fourier transform
    fftw_plan plan_ifft{}; //!< holds the plan for inverse fourier transform
    ptrdiff_t workspace_size{}; //!< size of workspace buffer returned by planner
    Real *real_workspace{}; //!< temporary real workspace that is correctly padded
    bool initialised{false}; //!< to prevent double initialisation
  private:
  };

}  // muSpectre

#endif /* FFTWMPI_ENGINE_H */
