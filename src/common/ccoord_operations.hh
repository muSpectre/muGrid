/**
 * file   ccoord_operations.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  common operations on pixel addressing 
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <functional>
#include <numeric>
#include <vector>

#include "common/common.hh"

#ifndef CCOORD_OPERATIONS_H
#define CCOORD_OPERATIONS_H

namespace muSpectre {

  namespace CcoordOps {

    //----------------------------------------------------------------------------//
    template <size_t dim>
    Ccoord_t<dim> get_cube(Dim_t size) {
      Ccoord_t<dim> retval{0};
      for(auto & val: retval) val=size;
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr Ccoord_t<dim> get_ccoord(const Ccoord_t<dim> & sizes, Dim_t index) {
      Ccoord_t<dim> retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval[i] = index/factor%sizes[i];
        if (i != 0 ) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr Dim_t get_index(const Ccoord_t<dim> & sizes,
                              const Ccoord_t<dim> & ccoord) {
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval += ccoord[i]*factor;
        if (i != 0) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr size_t get_size(const Ccoord_t<dim>& sizes) {
      return std::accumulate(sizes.begin(), sizes.end(), 1,
                             std::multiplies<size_t>());
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    class Pixels
    {
    public:
      Pixels(const Ccoord_t<dim> & sizes):sizes(sizes){};
      virtual ~Pixels() = default;
      class iterator
      {
      public:
        using value_type = Ccoord_t<dim>;
        using iterator_category = std::forward_iterator_tag;
        iterator(const Pixels & pixels, bool begin=true);
        virtual ~iterator() = default;
        inline value_type operator*() const;
        inline iterator & operator++();
        inline bool operator!=(const iterator & other) const;

      protected:
        const Pixels& pixels;
        size_t index;
      };
      inline iterator begin() const {return iterator(*this);}
      inline iterator end() const {return iterator(*this, false);}
    protected:
      const Ccoord_t<dim> & sizes;
    };

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    Pixels<dim>::iterator::iterator(const Pixels & pixels, bool begin)
      :pixels{pixels}, index{begin? 0: get_size(pixels.sizes)}
    {}

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    typename Pixels<dim>::iterator::value_type
    Pixels<dim>::iterator::operator*() const {
      return get_ccoord(pixels.sizes, this->index);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    bool
    Pixels<dim>::iterator::operator!=(const iterator &other) const {
      return (this->index != other.index) && (&this->pixels != &other.pixels);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    typename Pixels<dim>::iterator&
    Pixels<dim>::iterator::operator++() {
      ++this->index;
      return *this;
    }

  }  // CcoordOps

}  // muSpectre

#endif /* CCOORD_OPERATIONS_H */
