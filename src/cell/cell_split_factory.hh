/**
 * @file   cell_split_factory.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   01 Nov 2018
 *
 * @brief  Implementation for cell base class
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
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */
#ifndef SRC_CELL_CELL_SPLIT_FACTORY_HH_
#define SRC_CELL_CELL_SPLIT_FACTORY_HH_

#include "common/muSpectre_common.hh"
#include "libmugrid/ccoord_operations.hh"
#include "cell/cell_split.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"
#include "libmufft/fftw_engine.hh"
#include "cell/cell_factory.hh"

#ifdef WITH_MPI
#include <libmufft/communicator.hh>
#include <libmufft/fftwmpi_engine.hh>
#endif

#include <memory>
namespace muSpectre {
  template <typename Cell_t = CellSplit, class FFTEngine = muFFT::FFTWEngine>
  inline Cell_t
  make_cell_split(const DynCcoord_t & nb_grid_pts, const DynRcoord_t & lengths,
                  const Formulation & form, muFFT::Gradient_t gradient,
                  const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto && input{
        cell_input<FFTEngine>(nb_grid_pts, lengths, form, gradient, comm)};
    auto cell{Cell_t(std::move(input))};
    return cell;
  }
  /* ---------------------------------------------------------------------- */
  // this function returns a unique pointer to the CellSplit class of the cell
  // all members of cell and its descending cell class such as CellSplit are
  // available
  template <typename Cell_t = CellSplit, class FFTEngine = muFFT::FFTWEngine>
  std::unique_ptr<Cell_t>
  make_cell_ptr(const DynCcoord_t & nb_grid_pts, const DynRcoord_t & lengths,
                const Formulation & form, muFFT::Gradient_t gradient,
                const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto && input{
        cell_input<FFTEngine>(nb_grid_pts, lengths, form, gradient, comm)};
    return std::make_unique<Cell_t>(std::move(input));
  }
}  // namespace muSpectre

#endif  // SRC_CELL_CELL_SPLIT_FACTORY_HH_
