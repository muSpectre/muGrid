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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef PFFT_ENGINE_H
#define PFFT_ENGINE_H

#include "common/communicator.hh"

#include "fft/fft_engine_base.hh"

#include <pfft.h>

namespace muSpectre {

  /**
   * implements the `muSpectre::FFTEngineBase` interface using the
   * FFTW library
   */
  template <Dim_t DimS>
  class PFFTEngine: public FFTEngineBase<DimS>
  {
  public:
    using Parent = FFTEngineBase<DimS>; //!< base class
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    //! field for Fourier transform of second-order tensor
    using Workspace_t = typename Parent::Workspace_t;
    //! real-valued second-order tensor
    using Field_t = typename Parent::Field_t;
    //! Default constructor
    PFFTEngine() = delete;

    //! Constructor with system resolutions
    PFFTEngine(Ccoord resolutions, Dim_t nb_components,
               Communicator comm=Communicator());

    //! Copy constructor
    PFFTEngine(const PFFTEngine &other) = delete;

    //! Move constructor
    PFFTEngine(PFFTEngine &&other) = default;

    //! Destructor
    virtual ~PFFTEngine() noexcept;

    //! Copy assignment operator
    PFFTEngine& operator=(const PFFTEngine &other) = delete;

    //! Move assignment operator
    PFFTEngine& operator=(PFFTEngine &&other) = default;

    // compute the plan, etc
    virtual void initialise(FFT_PlanFlags plan_flags) override;

    //! forward transform
    virtual Workspace_t & fft(Field_t & field) override;

    //! inverse transform
    virtual void ifft(Field_t & field) const override;

  protected:
    MPI_Comm mpi_comm; //! < MPI communicator
    static int nb_engines; //!< number of times this engine has been instatiated
    pfft_plan plan_fft{}; //!< holds the plan for forward fourier transform
    pfft_plan plan_ifft{}; //!< holds the plan for inverse fourier transform
    ptrdiff_t workspace_size{}; //!< size of workspace buffer returned by planner
    Real *real_workspace{}; //!< temporary real workspace that is correctly padded
    bool initialised{false}; //!< to prevent double initialisation
  private:
  };

}  // muSpectre

#endif /* PFFT_ENGINE_H */
