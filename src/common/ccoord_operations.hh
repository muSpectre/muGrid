/**
 * @file   ccoord_operations.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   29 Sep 2017
 *
 * @brief  common operations on pixel addressing 
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
      //! simple helper returning the first argument and ignoring the second
      template <typename T>
      constexpr T ret(T val, size_t /*dummy*/) {return val;}

      //! helper to build cubes
      template <Dim_t Dim, typename T, size_t... I>
      constexpr std::array<T, Dim> cube_fun(T val, std::index_sequence<I...>) {
        return std::array<T, Dim>{ret(val, I)...};
      }

      //! computes hermitian size according to FFTW
      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & full_sizes,
                                   std::index_sequence<I...>) {
        return Ccoord_t<Dim>{full_sizes[I]..., full_sizes.back()/2+1};
      }

      //! compute the stride in a direction of a row-major grid
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

      //! get all strides from a row-major grid (helper function)
      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> compute_strides(const Ccoord_t<Dim> & sizes,
                                              std::index_sequence<I...>) {
        return Ccoord_t<Dim>{stride<Dim>(sizes, I)...};
      }
    }  // internal

    //----------------------------------------------------------------------------//
    //! returns a grid of equal resolutions in each direction
    template <size_t dim, typename T>
    constexpr std::array<T, dim> get_cube(T size) {
      return internal::cube_fun<dim>(size, std::make_index_sequence<dim>{});
    }

    /* ---------------------------------------------------------------------- */
    //! returns the hermition grid to correcsponding to a full grid
    template <size_t dim>
    constexpr Ccoord_t<dim> get_hermitian_sizes(Ccoord_t<dim> full_sizes) {
      return internal::herm<dim>(full_sizes, std::make_index_sequence<dim-1>{});
    }

    //! return physical vector of a cell of cubic pixels
    template <size_t dim>
    Eigen::Matrix<Real, dim, 1> get_vector(const Ccoord_t<dim> & ccoord, Real pix_size = 1.) {
      Eigen::Matrix<Real, dim, 1> retval;
      for (size_t i = 0; i < dim; ++i) {
        retval[i] = pix_size * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! return physical vector of a cell of general pixels
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
    //! return physical vector of a cell of general pixels
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
    //! get all strides from a row-major grid
    template <size_t dim>
    constexpr Ccoord_t<dim> get_default_strides(const Ccoord_t<dim> & sizes) {
      return internal::compute_strides<dim>(sizes,
                                            std::make_index_sequence<dim>{});
    }

    //----------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size sizes
    template <size_t dim>
    constexpr Ccoord_t<dim> get_ccoord(const Ccoord_t<dim> & resolutions,
                                       const Ccoord_t<dim> & locations,
                                       Dim_t index) {
      Ccoord_t<dim> retval{{0}};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval[i] = index/factor%resolutions[i] + locations[i];
        if (i != 0 ) {
          factor *= resolutions[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size sizes
    template <size_t dim, size_t... I>
    constexpr Ccoord_t<dim> get_ccoord(const Ccoord_t<dim> & resolutions,
                                       const Ccoord_t<dim> & locations,
                                       Dim_t index,
                                       std::index_sequence<I...>) {
      Ccoord_t<dim> ccoord{get_ccoord<dim>(resolutions, locations, index)};
      return Ccoord_t<dim>({ccoord[I]...});
    }

    //----------------------------------------------------------------------------//
    //! get the linear index of a pixel in a given grid
    template <size_t dim>
    constexpr Dim_t get_index(const Ccoord_t<dim> & sizes,
                              const Ccoord_t<dim> & locations,
                              const Ccoord_t<dim> & ccoord) {
      Dim_t retval{0};
      Dim_t factor{1};
      for (Dim_t i = dim-1; i >=0; --i) {
        retval += (ccoord[i]-locations[i])*factor;
        if (i != 0) {
          factor *= sizes[i];
        }
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    //! get the linear index of a pixel given a set of strides
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
    //! get the number of pixels in a grid
    template <size_t dim>
    constexpr size_t get_size(const Ccoord_t<dim>& sizes) {
      Dim_t retval{1};
      for (size_t i = 0; i < dim; ++i) {
        retval *= sizes[i];
      }
      return retval;
    }

    //----------------------------------------------------------------------------//
    //! get the number of pixels in a grid given its strides
    template <size_t dim>
    constexpr size_t get_size_from_strides(const Ccoord_t<dim>& sizes,
                                           const Ccoord_t<dim>& strides) {
      return sizes[0]*strides[0];
    }

    /* ---------------------------------------------------------------------- */
    /**
     * centralises iterating over square (or cubic) discretisation
     * grids.  The optional parameter pack `dmap` can be used to
     * specify the order of the axes in which to iterate over the
     * dimensions (i.e., dmap = 0, 1, 2 is rowmajor, and 0, 2, 1 would
     * be a custom order in which the second and third dimension are
     * transposed
     */
    template <size_t dim, int ...dmap>
    class Pixels {
    public:
      //! constructor
      Pixels(const Ccoord_t<dim> & resolutions=Ccoord_t<dim>{},
             const Ccoord_t<dim> & locations=Ccoord_t<dim>{})
        :resolutions{resolutions}, locations{locations}{};
      //! copy constructor
      Pixels(const Pixels & other) = default;
      //! assignment operator
      Pixels & operator=(const Pixels & other) = default;
      virtual ~Pixels() = default;

      /**
       * iterators over `Pixels` dereferences to cell coordinates
       */
      class iterator
      {
      public:
        using value_type = Ccoord_t<dim>; //!< stl conformance
        using const_value_type = const value_type; //!< stl conformance
        using pointer = value_type*; //!< stl conformance
        using difference_type = std::ptrdiff_t; //!< stl conformance
        using iterator_category = std::forward_iterator_tag;//!<stl conformance
        using reference = value_type; //!< stl conformance

        //! constructor
        iterator(const Pixels & pixels, bool begin=true);
        virtual ~iterator() = default;
        //! dereferencing
        inline value_type operator*() const;
        //! pre-increment
        inline iterator & operator++();
        //! inequality
        inline bool operator!=(const iterator & other) const;
        //! equality
        inline bool operator==(const iterator & other) const;

      protected:
        const Pixels& pixels; //!< ref to pixels in cell
        size_t index; //!< index of currect pointed-to pixel
      };
      //! stl conformance
      inline iterator begin() const {return iterator(*this);}
      //! stl conformance
      inline iterator end() const {return iterator(*this, false);}
      //! stl conformance
      inline size_t size() const {return get_size(this->resolutions);}
    protected:
      Ccoord_t<dim> resolutions; //!< resolutions of this domain
      Ccoord_t<dim> locations; //!< locations of this domain
    };

    /* ---------------------------------------------------------------------- */
    template <size_t dim, int ...dmap>
    Pixels<dim, dmap...>::iterator::iterator(const Pixels & pixels, bool begin)
      :pixels{pixels}, index{begin? 0: get_size(pixels.resolutions)}
    {}

    /* ---------------------------------------------------------------------- */
    template <size_t dim, int ...dmap>
    typename Pixels<dim, dmap...>::iterator::value_type
    Pixels<dim, dmap...>::iterator::operator*() const {
      return get_ccoord(pixels.resolutions, pixels.locations, this->index,
                        std::conditional_t<sizeof...(dmap) == 0,
                                           std::make_index_sequence<dim>,
                                           std::index_sequence<dmap...>>{});
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim, int ...dmap>
    bool
    Pixels<dim, dmap...>::iterator::operator!=(const iterator &other) const {
      return (this->index != other.index) || (&this->pixels != &other.pixels);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim, int ...dmap>
    bool
    Pixels<dim, dmap...>::iterator::operator==(const iterator &other) const {
      return !(*this!= other);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t dim, int ...dmap>
    typename Pixels<dim, dmap...>::iterator&
    Pixels<dim, dmap...>::iterator::operator++() {
      ++this->index;
      return *this;
    }

  }  // CcoordOps

}  // muSpectre

#endif /* CCOORD_OPERATIONS_H */
