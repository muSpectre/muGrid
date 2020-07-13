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

#include <functional>
#include <numeric>
#include <utility>

#include <Eigen/Dense>

#include "exception.hh"
#include "grid_common.hh"
#include "iterators.hh"

#include <utility>

#ifndef SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_
#define SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_

namespace muGrid {

  namespace CcoordOps {
    namespace internal {
      //! simple helper returning the first argument and ignoring the second
      template <typename T>
      constexpr T ret(T val, size_t /*dummy*/) {
        return val;
      }

      //! helper to build cubes
      template <Dim_t Dim, typename T, size_t... I>
      constexpr std::array<T, Dim> cube_fun(T val, std::index_sequence<I...>) {
        return std::array<T, Dim>{ret(val, I)...};
      }

      //! computes hermitian size according to FFTW
      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & nb_grid_pts,
                                   std::index_sequence<I...>) {
        return Ccoord_t<Dim>{nb_grid_pts.front() / 2 + 1,
                             nb_grid_pts[I + 1]...};
      }

      //! compute the stride in a direction of a column-major grid
      template <Dim_t Dim>
      constexpr Index_t col_major_stride(const Ccoord_t<Dim> & nb_grid_pts,
                               const size_t index) {
        static_assert(Dim > 0, "only for positive numbers of dimensions");

        Index_t ret_val{1};
        for (size_t i{0}; i < index; ++i) {
          ret_val *= nb_grid_pts[i];
        }
        return ret_val;
      }

      //! compute the stride in a direction of a row-major grid
      template <Dim_t Dim>
      constexpr Index_t row_major_stride(const Ccoord_t<Dim> & nb_grid_pts,
                               const size_t index) {
        static_assert(Dim > 0, "only for positive numbers of dimensions");

        Index_t ret_val{1};
        for (size_t i{Dim-1}; i > index; --i) {
          ret_val *= nb_grid_pts[i];
        }
        return ret_val;
      }

      //! get all strides from a column-major grid (helper function)
      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> compute_col_major_strides(
          const Ccoord_t<Dim> & nb_grid_pts, std::index_sequence<I...>) {
        return Ccoord_t<Dim>{col_major_stride<Dim>(nb_grid_pts, I)...};
      }

      //! get all strides from a row-major grid (helper function)
      template <Dim_t Dim, size_t... I>
      constexpr Ccoord_t<Dim> compute_row_major_strides(
          const Ccoord_t<Dim> & nb_grid_pts, std::index_sequence<I...>) {
        return Ccoord_t<Dim>{row_major_stride<Dim>(nb_grid_pts, I)...};
      }
    }  // namespace internal

    //! modulo operator that can handle negative values
    template <typename T>
    inline T modulo(const T & a, const T & b) {
      return (b + (a % b)) % b;
    }

    //! returns a grid of equal number of grid points in each direction
    template <size_t Dim, typename T>
    constexpr std::array<T, Dim> get_cube(T nb_grid_pts) {
      return internal::cube_fun<Dim>(nb_grid_pts,
                                     std::make_index_sequence<Dim>{});
    }

    //! returns a grid of equal number of grid points in each direction
    template <size_t MaxDim = threeD>
    DynCcoord<MaxDim> get_cube(const Dim_t & dim, const Index_t & nb_grid_pts) {
      switch (dim) {
      case oneD: {
        return DynCcoord<MaxDim>{get_cube<oneD>(nb_grid_pts)};
        break;
      }
      case twoD: {
        return DynCcoord<MaxDim>{get_cube<twoD>(nb_grid_pts)};
        break;
      }
      case threeD: {
        return DynCcoord<MaxDim>{get_cube<threeD>(nb_grid_pts)};
        break;
      }
      default:
        throw RuntimeError("Unknown dimension");
        break;
      }
    }

    //! return physical vector of a cell of cubic pixels
    template <size_t Dim>
    Eigen::Matrix<Real, Dim, 1> get_vector(const Ccoord_t<Dim> & ccoord,
                                           Real pix_size = 1.) {
      Eigen::Matrix<Real, Dim, 1> retval;
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size * ccoord[i];
      }
      return retval;
    }

    //! return physical vector of a cell of general pixels
    template <size_t Dim, typename T>
    Eigen::Matrix<T, Dim, 1>
    get_vector(const Ccoord_t<Dim> & ccoord,
               Eigen::Matrix<T, Dim_t(Dim), 1> pix_size) {
      Eigen::Matrix<T, Dim, 1> retval{pix_size};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] *= ccoord[i];
      }
      return retval;
    }

    //! return physical vector of a cell of general pixels
    template <size_t Dim, typename T>
    Eigen::Matrix<T, Dim, 1> get_vector(const Ccoord_t<Dim> & ccoord,
                                        const std::array<T, Dim> & pix_size) {
      Eigen::Matrix<T, Dim, 1> retval{};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size[i] * ccoord[i];
      }
      return retval;
    }

    //! return physical vector of a cell of general pixels
    template <size_t Dim, size_t MaxDim, typename T>
    Eigen::Matrix<T, Dim, 1> get_vector(const Ccoord_t<Dim> & ccoord,
                                        const DynCcoord<MaxDim, T> & pix_size) {
      assert(Dim == pix_size.get_dim());
      Eigen::Matrix<T, Dim, 1> retval{};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size[i] * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! return physical vector of a cell of cubic pixels
    template <size_t Dim>
    Eigen::Matrix<Real, Dim, 1> get_vector(const DynCcoord_t & ccoord,
                                           Real pix_size = 1.) {
      assert(Dim == ccoord.get_dim());
      Eigen::Matrix<Real, Dim, 1> retval;
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! return physical vector of a cell of general pixels
    template <size_t Dim, typename T>
    Eigen::Matrix<T, Dim, 1>
    get_vector(const DynCcoord_t ccoord,
               Eigen::Matrix<T, Dim_t(Dim), 1> pix_size) {
      assert(Dim == ccoord.get_dim());
      Eigen::Matrix<T, Dim, 1> retval = pix_size;
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] *= ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! return physical vector of a cell of general pixels
    template <size_t Dim, typename T>
    Eigen::Matrix<T, Dim, 1> get_vector(const DynCcoord_t ccoord,
                                        const std::array<T, Dim> & pix_size) {
      assert(Dim == ccoord.get_dim());
      Eigen::Matrix<T, Dim, 1> retval{};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size[i] * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! return physical vector of a cell of general pixels
    template <size_t Dim, size_t MaxDim, typename T>
    Eigen::Matrix<T, Dim, 1> get_vector(const DynCcoord_t ccoord,
                                        const DynCcoord<MaxDim, T> & pix_size) {
      assert(Dim == ccoord.get_dim());
      assert(Dim == pix_size.get_dim());
      Eigen::Matrix<T, Dim, 1> retval{};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = pix_size[i] * ccoord[i];
      }
      return retval;
    }

    /* ---------------------------------------------------------------------- */
    //! get all strides from a column-major grid
    template <size_t Dim>
    constexpr Ccoord_t<Dim>
    get_col_major_strides(const Ccoord_t<Dim> & nb_grid_pts) {
      return internal::compute_col_major_strides<Dim>(
          nb_grid_pts, std::make_index_sequence<Dim>{});
    }

    /* ---------------------------------------------------------------------- */
    //! get all strides from a column-major grid
    template <size_t MaxDim>
    constexpr DynCcoord<MaxDim>
    get_col_major_strides(const DynCcoord<MaxDim> & nb_grid_pts) {
      switch (nb_grid_pts.get_dim()) {
      case oneD: {
        return DynCcoord<MaxDim>{
            internal::compute_col_major_strides<oneD>(
                nb_grid_pts.template get<oneD>(),
                std::make_index_sequence<oneD>{})};
        break;
      }
      case twoD: {
        return DynCcoord<MaxDim>{
            internal::compute_col_major_strides<twoD>(
                nb_grid_pts.template get<twoD>(),
                std::make_index_sequence<twoD>{})};
        break;
      }
      case threeD: {
        return DynCcoord<MaxDim>{
            internal::compute_col_major_strides<threeD>(
                nb_grid_pts.template get<threeD>(),
                std::make_index_sequence<threeD>{})};
        break;
      }
      default:
        throw RuntimeError(
            "unforeseen dimensionality, is it really necessary to have other "
            "dimensions than 1, 2, and 3?");
        break;
      }
    }

    /* ---------------------------------------------------------------------- */
    //! get all strides from a row-major grid
    template <size_t Dim>
    constexpr Ccoord_t<Dim>
    get_row_major_strides(const Ccoord_t<Dim> & nb_grid_pts) {
      return internal::compute_row_major_strides<Dim>(
          nb_grid_pts, std::make_index_sequence<Dim>{});
    }

    /* ---------------------------------------------------------------------- */
    //! get all strides from a row-major grid
    template <size_t MaxDim>
    constexpr DynCcoord<MaxDim>
    get_row_major_strides(const DynCcoord<MaxDim> & nb_grid_pts) {
      switch (nb_grid_pts.get_dim()) {
      case oneD: {
        return DynCcoord<MaxDim>{
            internal::compute_row_major_strides<oneD>(
                nb_grid_pts.template get<oneD>(),
                std::make_index_sequence<oneD>{})};
        break;
      }
      case twoD: {
        return DynCcoord<MaxDim>{
            internal::compute_row_major_strides<twoD>(
                nb_grid_pts.template get<twoD>(),
                std::make_index_sequence<twoD>{})};
        break;
      }
      case threeD: {
        return DynCcoord<MaxDim>{
            internal::compute_row_major_strides<threeD>(
                nb_grid_pts.template get<threeD>(),
                std::make_index_sequence<threeD>{})};
        break;
      }
      default:
        throw RuntimeError(
            "unforeseen dimensionality, is it really necessary to have other "
            "dimensions than 1, 2, and 3?");
        break;
      }
    }

    //------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size nb_grid_pts
    template <size_t Dim>
    constexpr Ccoord_t<Dim> get_ccoord(const Ccoord_t<Dim> & nb_grid_pts,
                                       const Ccoord_t<Dim> & locations,
                                       Index_t index) {
      Ccoord_t<Dim> retval{{0}};
      Index_t factor{1};
      for (size_t i{0}; i < Dim; ++i) {
        retval[i] = index / factor % nb_grid_pts[i] + locations[i];
        if (i != Dim - 1) {
          factor *= nb_grid_pts[i];
        }
      }
      return retval;
    }

    //------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size nb_grid_pts
    template <size_t Dim, size_t... I>
    constexpr Ccoord_t<Dim> get_ccoord(const Ccoord_t<Dim> & nb_grid_pts,
                                       const Ccoord_t<Dim> & locations,
                                       Index_t index,
                                       std::index_sequence<I...>) {
      Ccoord_t<Dim> ccoord{get_ccoord<Dim>(nb_grid_pts, locations, index)};
      return Ccoord_t<Dim>({ccoord[I]...});
    }

    //------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size nb_grid_pts - specialization for
    //! one dimension
    template <size_t... I>
    constexpr Ccoord_t<1> get_ccoord(const Ccoord_t<1> & nb_grid_pts,
                                     const Ccoord_t<1> & locations,
                                     Index_t index, std::index_sequence<I...>) {
      return Ccoord_t<1>({get_ccoord<1>(nb_grid_pts, locations, index)});
    }

    //! compute the order of the axes given strides, fastest first
    template <class T>
    T compute_axes_order(const T & shape, const T & strides) {
      T axes_order(shape.size());
      std::iota(axes_order.begin(), axes_order.end(), 0);
      std::sort(axes_order.begin(), axes_order.end(),
                [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                  return (strides[a] == 1 and strides[b] == 1 and
                          shape[a] < shape[b]) or
                         strides[a] < strides[b];
                });
      return axes_order;
    }

    //! compute the order of the axes given strides, fastest first
    template <size_t dim>
    Ccoord_t<dim>
    compute_axes_order(const Ccoord_t<dim> & shape,
                       const Ccoord_t<dim> & strides) {
      Ccoord_t<dim> axes_order;
      std::iota(axes_order.begin(), axes_order.end(), 0);
      std::sort(axes_order.begin(), axes_order.end(),
                [&shape, &strides](const Dim_t & a, const Dim_t & b) {
                  return (strides[a] == 1 and strides[b] == 1 and
                          shape[a] < shape[b]) or
                         strides[a] < strides[b];
                });
      return axes_order;
    }

    //------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
    template <size_t dim>
    Ccoord_t<dim>
    get_ccoord_from_axes_order(const Ccoord_t<dim> & nb_grid_pts,
                               const Ccoord_t<dim> & locations,
                               const Ccoord_t<dim> & strides,
                               const Ccoord_t<dim> & axes_order,
                               Index_t index) {
      Ccoord_t<dim> retval{{nb_grid_pts[0]}};
      for (Index_t i{dim-1}; i >= 0; --i) {
        Index_t cur_coord{index / strides[axes_order[i]]};
        retval[axes_order[i]] = cur_coord;
        index -= cur_coord * strides[axes_order[i]];
      }
      for (size_t i{0}; i < dim; ++i) {
        retval[i] += locations[i];
      }      return retval;
    }

    //! get the i-th pixel in a grid of size nb_grid_pts, with strides
    template <size_t dim>
    Ccoord_t<dim>
    get_ccoord_from_strides(const Ccoord_t<dim> & nb_grid_pts,
                            const Ccoord_t<dim> & locations,
                            const Ccoord_t<dim> & strides,
                            Index_t index) {
      return get_ccoord_from_axes_order(
          nb_grid_pts, locations, strides,
          compute_axes_order(nb_grid_pts, strides), index);
    }

    //------------------------------------------------------------------------//
    //! get the i-th pixel in a grid of size nb_grid_pts, with axes order
    template <class T>
    T get_ccoord_from_axes_order(const T & nb_grid_pts, const T & locations,
                                 const T & strides, const T & axes_order,
                                 Index_t index) {
      auto & dim{nb_grid_pts.get_dim()};
      T retval(dim);
      for (Index_t i{dim-1}; i >= 0; --i) {
        Index_t cur_coord{index / strides[axes_order[i]]};
        retval[axes_order[i]] = cur_coord;
        index -= cur_coord * strides[axes_order[i]];
      }
      for (Dim_t i{0}; i < dim; ++i) {
        retval[i] += locations[i];
      }
      return retval;
    }

    //! get the i-th pixel in a grid of size nb_grid_pts, with strides
    template <class T>
    T get_ccoord_from_strides(const T & nb_grid_pts, const T & locations,
                              const T & strides, Index_t index) {
      return get_ccoord_from_axes_order(
          nb_grid_pts, locations, strides,
          compute_axes_order(nb_grid_pts, strides), index);
    }

    //------------------------------------------------------------------------//
    //! get the linear index of a pixel in a column-major grid
    template <size_t Dim>
    constexpr Dim_t get_index(const Ccoord_t<Dim> & nb_grid_pts,
                              const Ccoord_t<Dim> & locations,
                              const Ccoord_t<Dim> & ccoord) {
      Dim_t retval{0};
      Dim_t factor{1};
      for (size_t i{0}; i < Dim; ++i) {
        retval += (ccoord[i] - locations[i]) * factor;
        if (i != Dim - 1) {
          factor *= nb_grid_pts[i];
        }
      }
      return retval;
    }

    //! get the linear index of a pixel in a column-major grid
    Dim_t get_index(const DynCcoord_t & nb_grid_pts,
                    const DynCcoord_t & locations, const DynCcoord_t & ccoord);

    //-----------------------------------------------------------------------//
    //! these functions can be used whenever it is necessary to calcluate the
    //! volume of a cell or each pixle of the cell
    Real compute_volume(const DynRcoord_t & lenghts);

    Real compute_pixel_volume(const DynCcoord_t & nb_grid_pts,
                              const DynRcoord_t & lenghts);

    //! check whether strides represent a contiguous buffer
    template <class T>
    bool is_buffer_contiguous(const T & nb_grid_pts, const T & strides) {
      Index_t dim{static_cast<Index_t>(nb_grid_pts.size())};
      if (dim == 0) {
        return true;
      }
      if (static_cast<Index_t>(strides.size()) != dim) {
        throw RuntimeError("Mismatch between dimensions of nb_grid_pts and "
                           "strides");
      }
      auto axes{compute_axes_order(nb_grid_pts, strides)};
      Dim_t stride{1};
      bool is_contiguous{true};
      for (Index_t i{0}; i < dim; ++i) {
        is_contiguous &= strides[axes[i]] == stride;
        stride *= nb_grid_pts[axes[i]];
      }
      return is_contiguous;
    }

    //! get the linear index of a pixel given a set of strides
    template <size_t Dim>
    constexpr Index_t get_index_from_strides(const Ccoord_t<Dim> & strides,
                                             const Ccoord_t<Dim> & locations,
                                             const Ccoord_t<Dim> & ccoord) {
      Index_t retval{0};
      for (const auto & tup : akantu::zip(strides, locations, ccoord)) {
        const auto & stride{std::get<0>(tup)};
        const auto & location{std::get<1>(tup)};
        const auto & coord{std::get<2>(tup)};
        retval += stride * (coord - location);
      }
      return retval;
    }

    //! get the linear index of a pixel given a set of strides
    template <class T>
    Index_t get_index_from_strides(const T & strides, const T & locations,
                                   const T & ccoord) {
      const auto dim{strides.size()};
      if (locations.size() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between strides (dim = " << dim
              << ") and locations (dim = " << locations.get_dim() << ")";
        throw RuntimeError(error.str());
      }
      if (ccoord.size() != dim) {
        std::stringstream error{};
        error << "Dimension mismatch between strides (dim = " << dim
              << ") and ccoord (dim = " << ccoord.get_dim() << ")";
        throw RuntimeError(error.str());
      }
      Index_t retval{0};
      for (const auto & tup : akantu::zip(strides, locations, ccoord)) {
        const auto & stride{std::get<0>(tup)};
        const auto & location{std::get<1>(tup)};
        const auto & coord{std::get<2>(tup)};
        retval += stride * (coord - location);
      }
      return retval;
    }

    //-----------------------------------------------------------------------//
    //! get the number of pixels in a grid
    template <size_t Dim>
    constexpr size_t get_size(const Ccoord_t<Dim> & nb_grid_pts) {
      size_t retval{1};
      for (size_t i{0}; i < Dim; ++i) {
        retval *= nb_grid_pts[i];
      }
      return retval;
    }

    //-----------------------------------------------------------------------//
    //! get the number of pixels in a grid
    template <size_t MaxDim>
    size_t get_size(const DynCcoord<MaxDim> & nb_grid_pts) {
      size_t retval{1};
      Dim_t dim{nb_grid_pts.get_dim()};
      for (Dim_t i{0}; i < dim; ++i) {
        retval *= nb_grid_pts[i];
      }
      return retval;
    }

    //-----------------------------------------------------------------------//
    //! get the buffer size required to store a grid given its strides
    template <size_t dim>
    constexpr size_t get_buffer_size(const Ccoord_t<dim> & nb_grid_pts,
                                     const Ccoord_t<dim> & strides) {
      size_t buffer_size{0};
      // We need to loop over the dimensions because the largest stride can
      // occur anywhere. (It depends on the storage order.)
      for (size_t i{0}; i < dim; ++i) {
        buffer_size = std::max(
            buffer_size, static_cast<size_t>(nb_grid_pts[i] * strides[i]));
      }
      return buffer_size;
    }

    //-----------------------------------------------------------------------//
    //! get the buffer size required to store a grid given its strides
    size_t get_buffer_size(const DynCcoord_t & nb_grid_pts,
                           const DynCcoord_t & strides);

    //-----------------------------------------------------------------------//
    //! get the buffer size required to store a grid given its strides
    size_t get_buffer_size(const Shape_t & nb_grid_pts,
                           const Shape_t & strides);

    //! forward declaration
    template <size_t Dim>
    class Pixels;

    /**
     * Iteration over square (or cubic) discretisation grids. Duplicates
     * capabilities of `muGrid::CcoordOps::Pixels` without needing to be
     * templated with the spatial dimension. Iteration is slower, though.
     */
    class DynamicPixels {
     public:
      DynamicPixels();

      //! Constructor with default strides (column-major pixel storage order)
      explicit DynamicPixels(
          const DynCcoord_t & nb_subdomain_grid_pts,
          const DynCcoord_t & subdomain_locations = DynCcoord_t{});

      /**
       * Constructor with custom strides (any, including partially transposed
       * pixel storage order)
       */
      DynamicPixels(const DynCcoord_t & nb_subdomain_grid_pts,
                    const DynCcoord_t & subdomain_locations,
                    const DynCcoord_t & strides);

      //! Constructor with default strides from statically sized coords
      template <size_t Dim>
      explicit DynamicPixels(
          const Ccoord_t<Dim> & nb_subdomain_grid_pts,
          const Ccoord_t<Dim> & subdomain_locations = Ccoord_t<Dim>{});

      //! Constructor with custom strides from statically sized coords
      template <size_t Dim>
      DynamicPixels(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                    const Ccoord_t<Dim> & subdomain_locations,
                    const Ccoord_t<Dim> & strides);

      //! Copy constructor
      DynamicPixels(const DynamicPixels & other) = default;

      //! Move constructor
      DynamicPixels(DynamicPixels && other) = default;

      //! Destructor
      virtual ~DynamicPixels() = default;

      //! Copy assignment operator
      DynamicPixels & operator=(const DynamicPixels & other) = default;

      //! Move assignment operator
      DynamicPixels & operator=(DynamicPixels && other) = default;

      //! evaluate and return the linear index corresponding to dynamic `ccoord`
      Index_t get_index(const DynCcoord_t & ccoord) const {
        return get_index_from_strides(this->strides, this->subdomain_locations,
                                      ccoord);
      }

      //! evaluate and return the linear index corresponding to `ccoord`
      template <size_t Dim>
      Index_t get_index(const Ccoord_t<Dim> & ccoord) const {
        if (this->dim != Dim) {
          throw RuntimeError("dimension mismatch");
        }
        return get_index_from_strides(
            this->strides.template get<Dim>(),
            this->subdomain_locations.template get<Dim>(), ccoord);
      }

      //! return coordinates of the i-th pixel
      DynCcoord_t get_ccoord(const Index_t & index) const {
        return get_ccoord_from_axes_order(this->nb_subdomain_grid_pts,
                                          this->subdomain_locations,
                                          this->strides,
                                          this->axes_order,
                                          index);
      }

      /**
       * return a reference to the Pixels object cast into a statically
       * dimensioned grid. the statically dimensioned version duplicates
       * `muGrid::Ccoordops::DynamicPixels`'s capabilities, but iterates much
       * more efficiently.
       */
      template <size_t Dim>
      const Pixels<Dim> & get_dimensioned_pixels() const;

      class iterator;
      //! stl conformance
      iterator begin() const;
      //! stl conformance
      iterator end() const;
      //! stl conformance
      size_t size() const;

      //! return spatial dimension
      const Dim_t & get_dim() const { return this->dim; }

      //! return the resolution of the discretisation grid in each spatial dim
      const DynCcoord_t & get_nb_subdomain_grid_pts() const {
        return this->nb_subdomain_grid_pts;
      }

      /**
       * return the ccoordinates of the bottom, left, (front) pixel/voxel of
       * this processors partition of the discretisation grid. For sequential
       * calculations, this is alvays the origin
       */
      const DynCcoord_t & get_subdomain_locations() const {
        return this->subdomain_locations;
      }

      //! return the strides used for iterating over the pixels
      const DynCcoord_t & get_strides() const { return this->strides; }

      class Enumerator;
      /**
       * iterates in tuples of pixel index ond coordinate. Useful in parallel
       * problems, where simple enumeration of the pixels would be incorrect
       */
      Enumerator enumerate() const;

     protected:
      Dim_t dim;                          //!< spatial dimension
      DynCcoord_t nb_subdomain_grid_pts;  //!< nb_grid_pts of this domain
      DynCcoord_t subdomain_locations;    //!< locations of this domain
      DynCcoord_t strides;                //!< strides of memory layout
      DynCcoord_t axes_order;             //!< order of axes
      bool contiguous;                    //!< is this a contiguous buffer?
    };

    /**
     * Iterator class for `muSpectre::DynamicPixels`
     */
    class DynamicPixels::iterator {
     public:
      //! stl
      using value_type = DynCcoord<threeD>;
      using const_value_type = const value_type;            //!< stl conformance
      using pointer = value_type *;                         //!< stl conformance
      using difference_type = std::ptrdiff_t;               //!< stl conformance
      using iterator_category = std::forward_iterator_tag;  //!< stl
                                                            //!< conformance

      //! constructor
      iterator(const DynamicPixels & pixels, size_t index)
          : pixels{pixels}, index{index} {
        if (!pixels.contiguous) {
          std::stringstream message{};
          message << "Iterating over a DynamicPixels object is only supported "
                     "for contiguous buffers. You specified a grid of shape "
                  << pixels.nb_subdomain_grid_pts << " with non-contiguous "
                  << "strides " << pixels.strides << ".";
          throw RuntimeError{message.str()};
        }
      }
      //! Default constructor
      iterator() = delete;

      //! Copy constructor
      iterator(const iterator & other) = default;

      //! Move constructor
      iterator(iterator && other) = default;

      //! Destructor
      ~iterator() = default;

      //! Copy assignment operator
      iterator & operator=(const iterator & other) = delete;

      //! Move assignment operator
      iterator & operator=(iterator && other) = delete;

      //! dereferencing
      inline value_type operator*() const {
        return this->pixels.get_ccoord(this->index);
      }

      //! pre-increment
      inline iterator & operator++() {
        ++this->index;
        return *this;
      }

      //! inequality
      bool operator!=(const iterator & other) const {
        return this->index != other.index;
      }

      //! equality
      bool operator==(const iterator & other) const {
        return not(*this != other);
      }

     protected:
      const DynamicPixels & pixels;  //!< ref to pixels in cell
      size_t index;                  //!< index of currently pointed-to pixel
    };

    /**
     * enumerator class for `muSpectre::DynamicPixels`
     */
    class DynamicPixels::Enumerator final {
     public:
      //! Default constructor
      Enumerator() = delete;

      //! Constructor
      explicit Enumerator(const DynamicPixels & pixels);

      //! Copy constructor
      Enumerator(const Enumerator & other) = default;

      //! Move constructor
      Enumerator(Enumerator && other) = default;

      //! Destructor
      virtual ~Enumerator() = default;

      //! Copy assignment operator
      Enumerator & operator=(const Enumerator & other) = delete;

      //! Move assignment operator
      Enumerator & operator=(Enumerator && other) = delete;

      class iterator final : public DynamicPixels::iterator {
       public:
        using Parent = DynamicPixels::iterator;
        using Parent::Parent;
        std::tuple<Index_t, Parent::value_type> operator*() const {
          auto && pixel{this->Parent::operator*()};
          return std::tuple<Index_t, Parent::value_type>{
              this->pixels.get_index(pixel), pixel};
        }
      };

      //! stl conformance
      iterator begin() const;
      //! stl conformance
      iterator end() const;
      //! stl conformance
      size_t size() const;

     protected:
      const DynamicPixels & pixels;
    };

    /**
     * Centralised iteration over square (or cubic) discretisation grids.
     */
    template <size_t Dim>
    class Pixels : public DynamicPixels {
     public:
      //! base class
      using Parent = DynamicPixels;

      //! cell coordinates
      using Ccoord = Ccoord_t<Dim>;

      //! constructor
      Pixels(const Ccoord & nb_subdomain_grid_pts = Ccoord{},
             const Ccoord & subdomain_locations = Ccoord{})
          : Parent{nb_subdomain_grid_pts, subdomain_locations} {}
      //! constructor with strides
      Pixels(const Ccoord & nb_subdomain_grid_pts,
             const Ccoord & subdomain_locations, const Ccoord & strides)
          : Parent{nb_subdomain_grid_pts, subdomain_locations, strides} {}
      //! copy constructor
      Pixels(const Pixels & other) = default;
      //! assignment operator
      Pixels & operator=(const Pixels & other) = default;
      virtual ~Pixels() = default;

      //! return index for a ccoord
      Index_t get_index(const Ccoord & ccoord) const {
        return muGrid::CcoordOps::get_index(this->get_nb_grid_pts(),
                                            this->get_location(), ccoord);
      }

      //! return coordinates of the i-th pixel
      Ccoord get_ccoord(const Index_t & index) const {
        return get_ccoord_from_axes_order(
            this->nb_subdomain_grid_pts.template get<Dim>(),
            this->subdomain_locations.template get<Dim>(),
            this->strides.template get<Dim>(),
            this->axes_order.template get<Dim>(), index);
      }

      /**
       * iterators over `Pixels` dereferences to cell coordinates
       */
      class iterator {
       public:
        using value_type = Ccoord;                  //!< stl conformance
        using const_value_type = const value_type;  //!< stl conformance
        using pointer = value_type *;               //!< stl conformance
        using difference_type = std::ptrdiff_t;     //!< stl conformance
        using iterator_category = std::forward_iterator_tag;  //!< stl
                                                              //!< conformance
        using reference = value_type;  //!< stl conformance

        //! constructor
        explicit iterator(const Pixels & pixels, bool begin = true);
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
        const Pixels & pixels;  //!< ref to pixels in cell
        size_t index;           //!< index of currently pointed-to pixel
      };
      //! stl conformance
      inline iterator begin() const { return iterator(*this); }
      //! stl conformance
      inline iterator end() const { return iterator(*this, false); }
      //! stl conformance
      inline size_t size() const { return get_size(this->get_nb_grid_pts()); }

     protected:
      const Ccoord & get_nb_grid_pts() const {
        return this->nb_subdomain_grid_pts.template get<Dim>();
      }
      const Ccoord & get_subdomain_locations() const {
        return this->subdomain_locations.template get<Dim>();
      }
      const Ccoord & get_strides() const {
        return this->strides.template get<Dim>();
      }
    };

    /* ----------------------------------------------------------------------
     */
    template <size_t Dim>
    Pixels<Dim>::iterator::iterator(const Pixels & pixels, bool begin)
        : pixels{pixels}, index{begin ? 0
                                      : get_size(pixels.get_nb_grid_pts())} {}

    /* ----------------------------------------------------------------------
     */
    template <size_t Dim>
    typename Pixels<Dim>::iterator::value_type
        Pixels<Dim>::iterator::operator*() const {
      return this->pixels.get_ccoord(this->index);
    }

    /* ----------------------------------------------------------------------
     */
    template <size_t Dim>
    bool Pixels<Dim>::iterator::operator!=(const iterator & other) const {
      return this->index != other.index;
    }

    /* ----------------------------------------------------------------------
     */
    template <size_t Dim>
    bool Pixels<Dim>::iterator::operator==(const iterator & other) const {
      return not(*this != other);
    }

    /* ----------------------------------------------------------------------
     */
    template <size_t Dim>
    typename Pixels<Dim>::iterator & Pixels<Dim>::iterator::operator++() {
      ++this->index;
      return *this;
    }

  }  // namespace CcoordOps

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CCOORD_OPERATIONS_HH_
