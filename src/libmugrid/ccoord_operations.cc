/**
 * @file   ccoord_operations.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Oct 2019
 *
 * @brief  pre-compilable pixel operations
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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
#include <iostream>
#include "ccoord_operations.hh"

namespace muGrid {

  namespace CcoordOps {

    //------------------------------------------------------------------------//
    Dim_t get_index(const DynCcoord_t & nb_grid_pts,
                    const DynCcoord_t & locations, const DynCcoord_t & ccoord) {
      const Dim_t dim{nb_grid_pts.get_dim()};
      if (locations.get_dim() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (dim = " << dim
              << ") and locations (dim = " << locations.get_dim() << ")";
        throw std::runtime_error(error.str());
      }
      if (ccoord.get_dim() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (dim = " << dim
              << ") and ccoord (dim = " << ccoord.get_dim() << ")";
        throw std::runtime_error(error.str());
      }
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim - 1; i >= 0; --i) {
        retval += (ccoord[i] - locations[i]) * factor;
        if (i != 0) {
          factor *= nb_grid_pts[i];
        }
      }
      return retval;
    }

    //-----------------------------------------------------------------------//
    Real compute_volume(const DynRcoord_t & lenghts) {
      Real vol{1.0};
      for (auto && length : lenghts) {
        vol *= length;
      }
      return vol;
    }

    //-----------------------------------------------------------------------//
    Real compute_pixel_volume(const DynCcoord_t & nb_grid_pts,
                              const DynRcoord_t & lenghts) {
      Real vol{1.0};
      for (auto && tup : akantu::zip(nb_grid_pts, lenghts)) {
        auto && nb_grid_pt{std::get<0>(tup)};
        auto && length{std::get<1>(tup)};
        vol *= (length / nb_grid_pt);
      }
      return vol;
    }
    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels()
        : dim{}, nb_grid_pts{}, locations{}, strides{} {}

    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels(const DynCcoord_t & nb_grid_pts,
                                 const DynCcoord_t & locations)
        : dim(nb_grid_pts.get_dim()), nb_grid_pts(nb_grid_pts),
          locations(locations), strides(get_default_strides(nb_grid_pts)) {
      if (this->dim != this->locations.get_dim()) {
        throw std::runtime_error(
            "dimension mismatch between locations and nb_grid_pts.");
      }
    }

    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels(const DynCcoord_t & nb_grid_pts,
                                 const DynCcoord_t & locations,
                                 const DynCcoord_t & strides)
        : dim(nb_grid_pts.get_dim()), nb_grid_pts(nb_grid_pts),
          locations(locations), strides{strides} {
      if (this->dim != this->locations.get_dim()) {
        throw std::runtime_error(
            "dimension mismatch between locations and nb_grid_pts.");
      }
      if (this->dim != this->strides.get_dim()) {
        throw std::runtime_error(
            "dimension mismatch between locations and strides.");
      }
    }

    /* ---------------------------------------------------------------------- */
    template <size_t Dim>
    DynamicPixels::DynamicPixels(const Ccoord_t<Dim> & nb_grid_pts,
                                 const Ccoord_t<Dim> & locations)
        : dim(Dim), nb_grid_pts(nb_grid_pts), locations(locations),
          strides(get_default_strides(nb_grid_pts)) {}

    /* ---------------------------------------------------------------------- */
    template <size_t Dim>
    DynamicPixels::DynamicPixels(const Ccoord_t<Dim> & nb_grid_pts,
                                 const Ccoord_t<Dim> & locations,
                                 const Ccoord_t<Dim> & strides)
        : dim(Dim), nb_grid_pts(nb_grid_pts),
          locations(locations), strides{strides} {}

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::begin() const -> iterator { return iterator(*this, 0); }

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::end() const -> iterator {
      return iterator(*this, this->size());
    }

    /* ---------------------------------------------------------------------- */
    DynamicPixels::Enumerator::Enumerator(const DynamicPixels & pixels)
        : pixels{pixels} {}

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::Enumerator::begin() const -> iterator {
      return iterator{this->pixels, 0};
    }

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::Enumerator::end() const -> iterator {
      return iterator{this->pixels, this->pixels.size()};
    }

    /* ---------------------------------------------------------------------- */
    size_t DynamicPixels::Enumerator::size() const {
      return this->pixels.size();
    }

    /* ---------------------------------------------------------------------- */
    size_t DynamicPixels::size() const { return get_size(this->nb_grid_pts); }

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::enumerate() const -> Enumerator {
      return Enumerator(*this);
    }

    /* ----------------------------------------------------------------------*/
    template <size_t Dim>
    const Pixels<Dim> & DynamicPixels::get_dimensioned_pixels() const {
      if (Dim != this->dim) {
        std::stringstream error{};
        error << "You are trying to get a " << Dim
              << "-dimensional statically dimensioned view on a " << this->dim
              << "-dimensional DynamicPixels object";
        throw std::runtime_error(error.str());
      }
      return static_cast<const Pixels<Dim> &>(*this);
    }

    template DynamicPixels::DynamicPixels(const Ccoord_t<oneD> &,
                                          const Ccoord_t<oneD> &);
    template DynamicPixels::DynamicPixels(const Ccoord_t<twoD> &,
                                          const Ccoord_t<twoD> &);
    template DynamicPixels::DynamicPixels(const Ccoord_t<threeD> &,
                                          const Ccoord_t<threeD> &);
    template DynamicPixels::DynamicPixels(const Ccoord_t<oneD> &,
                                          const Ccoord_t<oneD> &,
                                          const Ccoord_t<oneD> &);
    template DynamicPixels::DynamicPixels(const Ccoord_t<twoD> &,
                                          const Ccoord_t<twoD> &,
                                          const Ccoord_t<twoD> &);
    template DynamicPixels::DynamicPixels(const Ccoord_t<threeD> &,
                                          const Ccoord_t<threeD> &,
                                          const Ccoord_t<threeD> &);
    template const Pixels<oneD> &
    DynamicPixels::get_dimensioned_pixels<oneD>() const;
    template const Pixels<twoD> &
    DynamicPixels::get_dimensioned_pixels<twoD>() const;
    template const Pixels<threeD> &
    DynamicPixels::get_dimensioned_pixels<threeD>() const;
  }  // namespace CcoordOps

}  // namespace muGrid
