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

#include "exception.hh"
#include "ccoord_operations.hh"

namespace muGrid {

  namespace CcoordOps {

    //------------------------------------------------------------------------//
    Dim_t get_index(const DynCcoord_t & nb_grid_pts,
                    const DynCcoord_t & locations, const DynCcoord_t & ccoord) {
      const Dim_t dim{nb_grid_pts.get_dim()};
      if (locations.get_dim() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (= " << nb_grid_pts
              << ") and locations (= " << locations << ")";
        throw RuntimeError(error.str());
      }
      if (ccoord.get_dim() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (= " << nb_grid_pts
              << ") and locations (= " << locations << ")";
        throw RuntimeError(error.str());
      }
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = 0; i < dim; ++i) {
        retval += (ccoord[i] - locations[i]) * factor;
        if (i != dim - 1) {
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

    size_t get_buffer_size(const DynCcoord_t & nb_grid_pts,
                           const DynCcoord_t & strides) {
      const Dim_t & dim{nb_grid_pts.get_dim()};
      if (strides.get_dim() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (= " << nb_grid_pts
              << ") and strides (= " << strides << ")";
        throw RuntimeError(error.str());
      }
      size_t buffer_size{0};
      // We need to loop over the dimensions because the largest stride can
      // occur anywhere. (It depends on the storage order.)
      for (Dim_t i{0}; i < dim; ++i) {
        buffer_size = std::max(
            buffer_size, static_cast<size_t>(nb_grid_pts[i] * strides[i]));
      }
      return buffer_size;
    }

    size_t get_buffer_size(const Shape_t & nb_grid_pts,
                           const Shape_t & strides) {
      const size_t & dim{nb_grid_pts.size()};
      if (strides.size() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_grid_pts (= " << nb_grid_pts
              << ") and strides (= " << strides << ")";
        throw RuntimeError(error.str());
      }
      size_t buffer_size{0};
      // We need to loop over the dimensions because the largest stride can
      // occur anywhere. (It depends on the storage order.)
      for (size_t i{0}; i < dim; ++i) {
        buffer_size = std::max(
            buffer_size, static_cast<size_t>(nb_grid_pts[i] * strides[i]));
      }
      return buffer_size;
    }

    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels()
        : dim{}, nb_subdomain_grid_pts{}, subdomain_locations{}, strides{},
          axes_order{}, contiguous{false} {}

    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels(const DynCcoord_t & nb_subdomain_grid_pts,
                                 const DynCcoord_t & subdomain_locations)
        : dim(nb_subdomain_grid_pts.get_dim()),
          nb_subdomain_grid_pts(nb_subdomain_grid_pts),
          subdomain_locations{
              subdomain_locations.get_dim() == 0
                  ? DynCcoord_t(nb_subdomain_grid_pts.get_dim())
                  : subdomain_locations},
          strides(get_col_major_strides(nb_subdomain_grid_pts)),
          axes_order{compute_axes_order(nb_subdomain_grid_pts, this->strides)},
          contiguous{true} {
      if (this->dim != this->subdomain_locations.get_dim()) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_subdomain_grid_pts (= "
              << nb_subdomain_grid_pts << ") and subdomain_locations (= "
              << subdomain_locations << ")";
        throw RuntimeError(error.str());
      }
    }

    /* ---------------------------------------------------------------------- */
    DynamicPixels::DynamicPixels(const DynCcoord_t & nb_subdomain_grid_pts,
                                 const DynCcoord_t & subdomain_locations,
                                 const DynCcoord_t & strides)
        : dim(nb_subdomain_grid_pts.get_dim()),
          nb_subdomain_grid_pts(nb_subdomain_grid_pts),
          subdomain_locations{
              subdomain_locations.get_dim() == 0
              ? DynCcoord_t(nb_subdomain_grid_pts.get_dim())
              : subdomain_locations},
          strides{strides},
          axes_order{compute_axes_order(nb_subdomain_grid_pts, strides)},
          contiguous{is_buffer_contiguous(nb_subdomain_grid_pts,
                                          strides)} {
      if (this->dim != this->subdomain_locations.get_dim()) {
        std::stringstream error{};
        error << "Dimension mismatch between nb_subdomain_grid_pts (= "
              << nb_subdomain_grid_pts << ") and subdomain_locations (= "
              << subdomain_locations << ")";
        throw RuntimeError(error.str());
      }
      if (this->dim != this->strides.get_dim()) {
        std::stringstream error{};
        error << "Dimension mismatch between subdomain_locations (= "
              << subdomain_locations << ") and strides (= " << strides << ")";
        throw RuntimeError(error.str());
      }
    }

    /* ---------------------------------------------------------------------- */
    template <size_t Dim>
    DynamicPixels::DynamicPixels(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                                 const Ccoord_t<Dim> & subdomain_locations)
        : dim(Dim), nb_subdomain_grid_pts(nb_subdomain_grid_pts),
          subdomain_locations(subdomain_locations),
          strides(get_col_major_strides(nb_subdomain_grid_pts)),
          axes_order{compute_axes_order(DynCcoord_t{nb_subdomain_grid_pts},
                                        this->strides)},
          contiguous{true} {}

    /* ---------------------------------------------------------------------- */
    template <size_t Dim>
    DynamicPixels::DynamicPixels(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                                 const Ccoord_t<Dim> & subdomain_locations,
                                 const Ccoord_t<Dim> & strides)
        : dim(Dim), nb_subdomain_grid_pts(nb_subdomain_grid_pts),
          subdomain_locations(subdomain_locations), strides{strides},
          axes_order{compute_axes_order(DynCcoord_t{nb_subdomain_grid_pts},
                                        DynCcoord_t{strides})},
          contiguous{is_buffer_contiguous(DynCcoord_t{nb_subdomain_grid_pts},
                                          DynCcoord_t{strides})} {}

    /* ---------------------------------------------------------------------- */
    auto DynamicPixels::begin() const -> iterator {
      return iterator(*this, 0);
    }

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
    size_t DynamicPixels::size() const {
      return get_size(this->nb_subdomain_grid_pts);
    }

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
        throw RuntimeError(error.str());
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
