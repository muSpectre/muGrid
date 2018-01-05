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
#include <utility>

#include <Eigen/Dense>

#include "common/common.hh"
#include "common/iterators.hh"

#ifndef CCOORD_OPERATIONS_H
#define CCOORD_OPERATIONS_H

namespace muSpectre {

  namespace CcoordOps {

    namespace internal {
      template <typename T>
      constexpr T ret(T val, size_t /*dummy*/) {return val;}

      template <Dim_t Dim, typename T, size_t... I>
      constexpr std::array<T, Dim> cube_fun(T val, std::index_sequence<I...>) {
        return std::array<T, Dim>{ret(val, I)...};
      }

      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & full_sizes,
                                   std::index_sequence<I...>) {
        return Ccoord_t<Dim>{full_sizes[I]..., (full_sizes.back()+1)/2};
      }

      template <Dim_t Dim>
      constexpr Dim_t stride(const Ccoord_t<Dim> & sizes,
                             const size_t index) {
        static_assert(Dim > 0, "only for positive numbers of dimensions");

        auto const diff{Dim - 1 - Dim_t(index)};
        Dim_t ret_val{1};
        for (Dim_t i{0}; i < diff; ++i) {
          ret_val *= sizes[Dim-1-i];
        }
        return ret_val;
      }

      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> compute_strides(const Ccoord_t<Dim> & sizes,
                                              std::index_sequence<I...>) {
        return Ccoord_t<Dim>{stride<Dim>(sizes, I)...};
      }
    }  // internal

    //----------------------------------------------------------------------------//
    template <size_t dim, typename T>
    constexpr std::array<T, dim> get_cube(T size) {
      return internal::cube_fun<dim>(size, std::make_index_sequence<dim>{});
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    constexpr Ccoord_t<dim> get_hermitian_sizes(Ccoord_t<dim> full_sizes) {
      return internal::herm<dim>(full_sizes, std::make_index_sequence<dim-1>{});
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    Eigen::Matrix<Real, dim, 1> get_vector(const Ccoord_t<dim> & ccoord, Real pix_size = 1.) {
      Eigen::Matrix<Real, dim, 1> retval;
      for (size_t i = 0; i < dim; ++i) {
        retval[i] = pix_size * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim, typename T>
    Eigen::Matrix<T, dim, 1> get_vector(const Ccoord_t<dim> & ccoord,
                                        Eigen::Matrix<T, Dim_t(dim), 1> pix_size) {
      Eigen::Matrix<T, dim, 1> retval = pix_size;
      for (size_t i = 0; i < dim; ++i) {
        retval[i] *= ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim, typename T>
    Eigen::Matrix<T, dim, 1> get_vector(const Ccoord_t<dim> & ccoord,
                                        const std::array<T, dim>& pix_size) {
      Eigen::Matrix<T, dim, 1> retval{};
      for (size_t i = 0; i < dim; ++i) {
        retval[i] = pix_size[i]*ccoord[i];
      }
      return retval;
    }


    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    constexpr Ccoord_t<dim> get_default_strides(const Ccoord_t<dim> & sizes) {
      return internal::compute_strides<dim>(sizes,
                                            std::make_index_sequence<dim>{});
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr Ccoord_t<dim> get_ccoord(const Ccoord_t<dim> & sizes, Dim_t index) {
      Ccoord_t<dim> retval{{0}};
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
    constexpr Dim_t get_index_from_strides(const Ccoord_t<dim> & strides,
                                           const Ccoord_t<dim> & ccoord) {
      Dim_t retval{0};
      for (const auto & tup: akantu::zip(strides, ccoord)) {
        const auto & stride = std::get<0>(tup);
        const auto & ccord_ = std::get<1>(tup);
        retval += stride * ccord_;
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr size_t get_size(const Ccoord_t<dim>& sizes) {
      Dim_t retval{1};
      for (size_t i = 0; i < dim; ++i) {
        retval *= sizes[i];
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    template <size_t dim>
    constexpr size_t get_size_from_strides(const Ccoord_t<dim>& sizes,
                                           const Ccoord_t<dim>& strides) {
      return sizes[0]*strides[0];
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    class Pixels {
    public:
      Pixels(const Ccoord_t<dim> & sizes=Ccoord_t<dim>{}):sizes(sizes){};
      Pixels(const Pixels & other) = default;
      Pixels & operator=(const Pixels & other) = default;
      virtual ~Pixels() = default;
      class iterator
      {
      public:
        using value_type = Ccoord_t<dim>;
        using const_value_type = const value_type;
        using pointer = value_type*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using reference = value_type;
        iterator(const Pixels & pixels, bool begin=true);
        virtual ~iterator() = default;
        inline value_type operator*() const;
        inline iterator & operator++();
        inline bool operator!=(const iterator & other) const;
        inline bool operator==(const iterator & other) const;

      protected:
        const Pixels& pixels;
        size_t index;
      };
      inline iterator begin() const {return iterator(*this);}
      inline iterator end() const {return iterator(*this, false);}
      inline size_t size() const {return get_size(this->sizes);}
    protected:
      Ccoord_t<dim>  sizes;
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
      return (this->index != other.index) || (&this->pixels != &other.pixels);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim>
    bool
    Pixels<dim>::iterator::operator==(const iterator &other) const {
      return !(*this!= other);
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
