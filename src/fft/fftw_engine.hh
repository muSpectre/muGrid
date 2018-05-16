/**
 * @file   fftw_engine.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  FFT engine using FFTW
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

#ifndef FFTW_ENGINE_H
#define FFTW_ENGINE_H

#include "fft/fft_engine_base.hh"

#include <fftw3.h>

namespace muSpectre {

  /**
   * implements the `muSpectre::FftEngine_Base` interface using the
   * FFTW library
   */
  template <Dim_t Dim>
  class FFTWEngine: public FFTEngineBase<Dim>
  {
  public:
    using Parent = FFTEngineBase<Dim>; //!< base class
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    //! field for Fourier transform of second-order tensor
    using Workspace_t = typename Parent::Workspace_t;
    //! real-valued second-order tensor
    using Field_t = typename Parent::Field_t;
    //! Default constructor
    FFTWEngine() = delete;

    //! Constructor with cell resolutions
    FFTWEngine(Ccoord resolutions, Dim_t nb_components,
               Communicator comm=Communicator());

    //! Copy constructor
    FFTWEngine(const FFTWEngine &other) = delete;

    //! Move constructor
    FFTWEngine(FFTWEngine &&other) = default;

    //! Destructor
    virtual ~FFTWEngine() noexcept;

    //! Copy assignment operator
    FFTWEngine& operator=(const FFTWEngine &other) = delete;

    //! Move assignment operator
    FFTWEngine& operator=(FFTWEngine &&other) = default;

    // compute the plan, etc
    virtual void initialise(FFT_PlanFlags plan_flags) override;

    //! forward transform
    virtual Workspace_t & fft(Field_t & field) override;

    //! inverse transform
    virtual void ifft(Field_t & field) const override;

  protected:
    fftw_plan plan_fft{}; //!< holds the plan for forward fourier transform
    fftw_plan plan_ifft{}; //!< holds the plan for inverse fourier transform
    bool initialised{false}; //!< to prevent double initialisation
  private:
  };

}  // muSpectre

#endif /* FFTW_ENGINE_H */
