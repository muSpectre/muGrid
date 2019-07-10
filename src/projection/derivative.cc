/**
 * @file   derivative.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   05 June 2019
 *
 * @brief  Representation of finite-differences stencils
 *
 * Copyright © 2019 Lars Pastewka
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

#include "projection/derivative.hh"

using muGrid::pi;
using muGrid::CcoordOps::get_size;
using muGrid::CcoordOps::get_index;
using muGrid::CcoordOps::Pixels;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  FourierDerivative<DimS>::FourierDerivative(Dim_t direction) :
      direction{direction} {
    if (direction < 0 || direction >= DimS) {
      throw ProjectionError("Derivative direction is a Cartesian "
                            "direction. It must be larger than or "
                            "equal to zero and smaller than the spatial "
                            "dimension.");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  DiscreteDerivative<DimS>::DiscreteDerivative(
      Ccoord nb_pts, Ccoord lbounds, const std::vector<double> & stencil) :
      nb_pts{nb_pts}, lbounds{lbounds},
      stencil{Eigen::Map<const Eigen::ArrayXd>(stencil.data(),
                                               stencil.size())} {
    if (get_size(nb_pts) != stencil.size()) {
      throw ProjectionError("Number of provided data points different from "
                            "stencil size.");
    }
    if (std::abs(this->stencil.sum() / stencil.size())
        > std::numeric_limits<Real>::epsilon()) {
      throw ProjectionError("Stencil coefficients must sum to zero.");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  DiscreteDerivative<DimS>::DiscreteDerivative(
      Ccoord nb_pts, Ccoord lbounds, const Eigen::ArrayXd & stencil) :
      nb_pts{nb_pts}, lbounds{lbounds}, stencil{stencil} {
    if (get_size(nb_pts) != static_cast<size_t>(this->stencil.size())) {
      throw ProjectionError("Number of provided data points different from "
                            "stencil size.");
    }
    if (std::abs(this->stencil.sum() / stencil.size())
        > std::numeric_limits<Real>::epsilon()) {
      throw ProjectionError("Stencil coefficients must sum to zero.");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  DiscreteDerivative<DimS> DiscreteDerivative<DimS>::rollaxes(
      int distance) const {
    Ccoord nb_pts, lbounds;
    Eigen::ArrayXd stencil(this->stencil.size());

    for (Dim_t dim = 0; dim < DimS; ++dim) {
      nb_pts[(dim+distance)%DimS] = this->nb_pts[dim];
      lbounds[(dim+distance)%DimS] = this->lbounds[dim];
    }

    for (auto && pixel : Pixels<DimS>(this->nb_pts, this->lbounds)) {
      Ccoord rolled_pixel;
      for (Dim_t dim = 0; dim < DimS; ++dim) {
        Dim_t rolled_dim = (dim+distance)%DimS;
        rolled_pixel[rolled_dim] = pixel[dim]-this->lbounds[dim]+lbounds[dim];
      }
      stencil[get_index(nb_pts, lbounds, rolled_pixel)] = (*this)(pixel);
    }

    return DiscreteDerivative(nb_pts, lbounds, stencil);
  }

  template class FourierDerivative<oneD>;
  template class FourierDerivative<twoD>;
  template class FourierDerivative<threeD>;

  template class DiscreteDerivative<oneD>;
  template class DiscreteDerivative<twoD>;
  template class DiscreteDerivative<threeD>;
}  // namespace muSpectre
