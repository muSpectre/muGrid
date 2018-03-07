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

#ifndef CELL_FACTORY_H
#define CELL_FACTORY_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_base.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/projection_small_strain.hh"
#include "fft/fftw_engine.hh"

#include <memory>

namespace muSpectre {


  /**
   * Create a unique ptr to a Projection operator (with appropriate
   * FFT_engine) to be used in a cell constructor
   */
  template <Dim_t DimS, Dim_t DimM,
            typename FFTEngine=FFTWEngine<DimS, DimM>>
  inline
  std::unique_ptr<ProjectionBase<DimS, DimM>>
  cell_input(Ccoord_t<DimS> resolutions,
               Rcoord_t<DimS> lengths,
               Formulation form) {
    auto fft_ptr{std::make_unique<FFTEngine>(resolutions, lengths)};
    switch (form)
      {
      case Formulation::finite_strain: {
        using Projection = ProjectionFiniteStrainFast<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr));
        break;
      }
      case Formulation::small_strain: {
        using Projection = ProjectionSmallStrain<DimS, DimM>;
        return std::make_unique<Projection>(std::move(fft_ptr));
        break;
      }
      default: {
        throw std::runtime_error("unknow formulation");
        break;
      }
    }
  }


  /**
   * convenience function to create a cell (avoids having to build
   * and move the chain of unique_ptrs
   */
  template <size_t DimS, size_t DimM=DimS,
            typename Cell=CellBase<DimS, DimM>,
            typename FFTEngine=FFTWEngine<DimS, DimM>>
  inline
  Cell make_cell(Ccoord_t<DimS> resolutions,
                     Rcoord_t<DimS> lengths,
                     Formulation form) {

    auto && input = cell_input<DimS, DimM, FFTEngine>(resolutions, lengths, form);
    auto cell{Cell{std::move(input)}};
    return cell;
  }

}  // muSpectre

#endif /* CELL_FACTORY_H */


