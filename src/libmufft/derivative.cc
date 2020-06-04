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

#include <iostream>

#include "derivative.hh"

using muGrid::pi;
using muGrid::CcoordOps::DynamicPixels;
using muGrid::CcoordOps::get_index;
using muGrid::CcoordOps::get_size;
using muGrid::CcoordOps::modulo;

namespace muFFT {

  DerivativeBase::DerivativeBase(Index_t spatial_dimension)
      : spatial_dimension{spatial_dimension} {
    if ((spatial_dimension < 1) or (spatial_dimension > 3)) {
      throw DerivativeError("Only 1, 2, or 3-dimensional problems.");
    }
  }

  /* ---------------------------------------------------------------------- */
  FourierDerivative::FourierDerivative(Index_t spatial_dimension,
                                       Index_t direction)
      : Parent{spatial_dimension}, direction{direction},
        shift{Eigen::ArrayXd::Zero(spatial_dimension)} {
    if (direction < 0 || direction >= spatial_dimension) {
      throw DerivativeError("Derivative direction is a Cartesian "
                            "direction. It must be larger than or "
                            "equal to zero and smaller than the spatial "
                            "dimension.");
    }
  }

  /* ---------------------------------------------------------------------- */
  FourierDerivative::FourierDerivative(Index_t spatial_dimension,
                                       Index_t direction,
                                       const Eigen::ArrayXd & shift)
      : Parent{spatial_dimension}, direction{direction}, shift{shift} {
    if (direction < 0 || direction >= spatial_dimension) {
      throw DerivativeError("Derivative direction is a Cartesian "
                            "direction. It must be larger than or "
                            "equal to zero and smaller than the spatial "
                            "dimension.");
    }
  }

  /* ---------------------------------------------------------------------- */
  DiscreteDerivative::DiscreteDerivative(DynCcoord_t nb_pts,
                                         DynCcoord_t lbounds,
                                         const std::vector<Real> & stencil)
      : Parent{nb_pts.get_dim()}, pixels{nb_pts, lbounds},
        stencil{
            Eigen::Map<const Eigen::ArrayXd>(stencil.data(), stencil.size())} {
    if (get_size(nb_pts) != stencil.size()) {
      std::stringstream s;
      s << "Stencil is supposed to have " << nb_pts << " (=" << get_size(nb_pts)
        << " total) data points, but " << stencil.size() << " stencil "
        << "coefficients were provided.";
      throw DerivativeError(s.str());
    }
    if (std::abs(this->stencil.sum() / stencil.size()) >
        std::numeric_limits<Real>::epsilon()) {
      throw DerivativeError("Stencil coefficients must sum to zero.");
    }
  }

  /* ---------------------------------------------------------------------- */
  DiscreteDerivative::DiscreteDerivative(DynCcoord_t nb_pts,
                                         DynCcoord_t lbounds,
                                         const Eigen::ArrayXd & stencil)
      : Parent{nb_pts.get_dim()}, pixels{nb_pts, lbounds}, stencil{stencil} {
    if (get_size(nb_pts) != static_cast<size_t>(this->stencil.size())) {
      std::stringstream s;
      s << "Stencil is supposed to have " << nb_pts << " (=" << get_size(nb_pts)
        << " total) data points, but "
        << static_cast<size_t>(this->stencil.size()) << " stencil coefficients "
        << "were provided.";
      throw DerivativeError(s.str());
    }
    if (std::abs(this->stencil.sum() / stencil.size()) >
        std::numeric_limits<Real>::epsilon()) {
      throw DerivativeError("Stencil coefficients must sum to zero.");
    }
  }

  /* ---------------------------------------------------------------------- */
  DiscreteDerivative DiscreteDerivative::rollaxes(int distance) const {
    DynCcoord_t new_nb_pts(this->spatial_dimension),
        new_lbounds(this->spatial_dimension);
    Eigen::ArrayXd stencil(this->stencil.size());

    for (Index_t dim{0}; dim < this->spatial_dimension; ++dim) {
      Index_t rolled_dim{modulo(dim + distance, this->spatial_dimension)};
      new_nb_pts[rolled_dim] = this->pixels.get_nb_subdomain_grid_pts()[dim];
      new_lbounds[rolled_dim] = this->pixels.get_subdomain_locations()[dim];
    }

    for (auto && pixel : this->pixels) {
      DynCcoord_t rolled_pixel(this->spatial_dimension);
      for (Index_t dim{0}; dim < this->spatial_dimension; ++dim) {
        Index_t rolled_dim{modulo(dim + distance, this->spatial_dimension)};
        rolled_pixel[rolled_dim] = pixel[dim] -
                                   this->pixels.get_subdomain_locations()[dim] +
                                   new_lbounds[dim];
      }
      stencil[get_index(new_nb_pts, new_lbounds, rolled_pixel)] =
          this->operator()(pixel);
    }

    return DiscreteDerivative(new_nb_pts, new_lbounds, stencil);
  }

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative & derivative) {
    const auto & nb_pts{derivative.get_nb_pts()};
    const auto & lbounds{derivative.get_lbounds()};
    os << "{ ";
    muGrid::operator<<(os, nb_pts);
    os << " ";
    muGrid::operator<<(os, lbounds);
    os << " ";
    for (auto && pixel : muGrid::CcoordOps::DynamicPixels(nb_pts, lbounds)) {
      os << derivative(pixel) << " ";
    }
    os << "}";
    return os;
  }

  /* ---------------------------------------------------------------------- */
  Gradient_t make_fourier_gradient(const Index_t & spatial_dimension) {
    Gradient_t && g{};
    for (Index_t dim = 0; dim < spatial_dimension; ++dim) {
      g.push_back(std::make_shared<FourierDerivative>(spatial_dimension, dim));
    }
    return std::move(g);
  }

}  // namespace muFFT
