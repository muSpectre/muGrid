/**
 * @file   cell_factory.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Dec 2017
 *
 * @brief  Cell factories to help create cells with ease
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_CELL_CELL_FACTORY_HH_
#define SRC_CELL_CELL_FACTORY_HH_

#include "common/muSpectre_common.hh"
#include "cell/cell_base.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"
#include <libmugrid/ccoord_operations.hh>
#include <libmufft/fftw_engine.hh>

#ifdef WITH_MPI
#include <libmufft/communicator.hh>
#include <libmufft/fftwmpi_engine.hh>
#endif

#include <memory>

namespace muSpectre {

  /**
   * Create a unique ptr to a Projection operator (with appropriate
   * FFT_engine) to be used in a cell constructor
   */
  template <Dim_t DimS, Dim_t DimM,
            typename FFTEngine = muFFT::FFTWEngine<DimS>>
  inline std::unique_ptr<ProjectionBase<DimS, DimM>>
  cell_input(Ccoord_t<DimS> nb_grid_pts,
             Rcoord_t<DimS> lengths,
             Formulation form,
             Gradient_t<DimS> gradient = make_fourier_gradient<DimS>(),
             const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto fft_ptr{std::make_unique<FFTEngine>(
        nb_grid_pts, dof_for_formulation(form, DimS), comm)};
    switch (form) {
    case Formulation::finite_strain: {
      using Projection = ProjectionFiniteStrainFast<DimS, DimM>;
      return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                          gradient);
      break;
    }
    case Formulation::small_strain: {
      using Projection = ProjectionSmallStrain<DimS, DimM>;
      return std::make_unique<Projection>(std::move(fft_ptr), lengths,
                                          gradient);
      break;
    }
    default: {
      throw std::runtime_error("Unknown formulation.");
      break;
    }
    }
  }

  /**
   * convenience function to create a cell (avoids having to build
   * and move the chain of unique_ptrs
   */
  template <size_t DimS, size_t DimM = DimS,
            typename Cell = CellBase<DimS, DimM>,
            typename FFTEngine = muFFT::FFTWEngine<DimS>>
  inline Cell
  make_cell(Ccoord_t<DimS> nb_grid_pts,
            Rcoord_t<DimS> lengths,
            Formulation form,
            Gradient_t<DimS> gradient = make_fourier_gradient<DimS>(),
            const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto && input =
      cell_input<DimS, DimM, FFTEngine>(nb_grid_pts, lengths, form, gradient,
                                        comm);
    auto cell{Cell{std::move(input)}};
    return cell;
  }

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_FACTORY_HH_
