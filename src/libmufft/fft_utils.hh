/**
 * @file   fft_utils.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Dec 2017
 *
 * @brief  collection of functions used in the context of spectral operations
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_LIBMUFFT_FFT_UTILS_HH_
#define SRC_LIBMUFFT_FFT_UTILS_HH_

#include <libmugrid/exception.hh>

#include "mufft_common.hh"

#include <Eigen/Dense>

#include <array>
#include <valarray>

namespace muFFT {

  namespace internal {
    //! computes hermitian size according to FFTW
    template <Dim_t Dim, size_t... I>
    constexpr Ccoord_t<Dim> herm(const Ccoord_t<Dim> & nb_grid_pts,
                                 std::index_sequence<I...>) {
      return Ccoord_t<Dim>{nb_grid_pts.front() / 2 + 1, nb_grid_pts[I+1]... };
    }
  }

  /**
   * returns the hermition grid to correcsponding to a full grid, assuming
   * that the last dimension is not fully represented in reciprocal space
   */
  template <size_t dim>
  constexpr Ccoord_t<dim>
  get_nb_hermitian_grid_pts(Ccoord_t<dim> full_nb_grid_pts) {
    return internal::herm<dim>(full_nb_grid_pts,
                               std::make_index_sequence<dim - 1>{});
  }

  /**
   * returns the hermition grid to correcsponding to a full grid, assuming
   * that the last dimension is not fully represented in reciprocal space
   */
  template <size_t MaxDim>
  inline muGrid::DynCcoord<MaxDim>
  get_nb_hermitian_grid_pts(muGrid::DynCcoord<MaxDim> full_nb_grid_pts) {
    switch (full_nb_grid_pts.get_dim()) {
    case oneD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<oneD>(full_nb_grid_pts))};
      break;
    }
    case twoD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<twoD>(full_nb_grid_pts))};
      break;
    }
    case threeD: {
      return muGrid::DynCcoord<MaxDim>{
          get_nb_hermitian_grid_pts(Ccoord_t<threeD>(full_nb_grid_pts))};
      break;
    }
    default:
      throw RuntimeError(
          "One 1, 2, and 3-dimensional cases are allowed");
      break;
    }
  }

  /**
   * compute fft frequency (in time (or length) units of of sampling
   * periods), see numpy's fftfreq function for reference
   */
  inline Int fft_freq(Int i, size_t nb_samples) {
    Int N = (nb_samples - 1) / 2 + 1;  // needs to be signed int for neg freqs
    if (i < N) {
      return i;
    } else {
      return -Int(nb_samples) / 2 + i - N;
    }
  }

  /**
   * compute fft frequency in correct length or time units. Here,
   * length refers to the total size of the domain over which the fft
   * is taken (for instance the length of an edge of an RVE)
   */
  inline Real fft_freq(Int i, size_t nb_samples, Real length) {
    return static_cast<Real>(fft_freq(i, nb_samples)) / length;
  }

  /**
   * compute fft frequencies (in time (or length) units of of sampling
   * periods), see numpy's fftfreq function for reference
   */
  std::valarray<Real> fft_freqs(size_t nb_samples);

  /**
   * compute fft frequencies in correct length or time units. Here,
   * length refers to the total size of the domain over which the fft
   * is taken (for instance the length of an edge of an RVE)
   */
  std::valarray<Real> fft_freqs(size_t nb_samples, Real length);

  /**
   * Get fft_freqs for a grid
   */
  template <size_t dim>
  inline std::array<std::valarray<Real>, dim>
  fft_freqs(Ccoord_t<dim> nb_grid_pts) {
    std::array<std::valarray<Real>, dim> retval{};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] = std::move(fft_freqs(nb_grid_pts[i]));
    }
    return retval;
  }

  /**
   * Get fft_freqs for a grid in correct length or time units.
   */
  template <size_t dim>
  inline std::array<std::valarray<Real>, dim>
  fft_freqs(Ccoord_t<dim> nb_grid_pts, std::array<Real, dim> lengths) {
    std::array<std::valarray<Real>, dim> retval{};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] = std::move(fft_freqs(nb_grid_pts[i], lengths[i]));
    }
    return retval;
  }

  /**
   * simple class encapsulating the creation, and retrieval of
   * wave vectors
   */
  template <Dim_t dim>
  class FFT_freqs {
   public:
    //! Eigen variant equivalent to Ccoord_t
    using CcoordVector = Eigen::Matrix<Dim_t, dim, 1>;
    //! return type for wave vectors
    using Vector = Eigen::Matrix<Real, dim, 1>;
    //! return type for complex wave vectors
    using VectorComplex = Eigen::Matrix<Complex, dim, 1>;
    //! Default constructor
    FFT_freqs() = delete;

    //! constructor with just number of grid points
    explicit FFT_freqs(Ccoord_t<dim> nb_grid_pts)
        : freqs{fft_freqs(nb_grid_pts)} {}

    //! constructor with domain length
    FFT_freqs(Ccoord_t<dim> nb_grid_pts, std::array<Real, dim> lengths)
        : freqs{fft_freqs(nb_grid_pts, lengths)} {}

    //! Copy constructor
    FFT_freqs(const FFT_freqs & other) = delete;

    //! Move constructor
    FFT_freqs(FFT_freqs && other) = default;

    //! Destructor
    virtual ~FFT_freqs() = default;

    //! Copy assignment operator
    FFT_freqs & operator=(const FFT_freqs & other) = delete;

    //! Move assignment operator
    FFT_freqs & operator=(FFT_freqs && other) = default;

    //! get unnormalised wave vector (in sampling units)
    inline Vector get_xi(const Ccoord_t<dim> ccoord) const;

    //! get unnormalised complex wave vector (in sampling units)
    inline VectorComplex get_complex_xi(const Ccoord_t<dim> ccoord) const;

    //! get normalised wave vector
    inline Vector get_unit_xi(const Ccoord_t<dim> ccoord) const {
      auto && xi = this->get_xi(std::move(ccoord));
      return xi / xi.norm();
    }

    inline Index_t get_nb_grid_pts(Index_t i) const {
      return this->freqs[i].size();
    }

   protected:
    //! container for frequencies ordered by spatial dimension
    const std::array<std::valarray<Real>, dim> freqs;
  };

  template <Dim_t dim>
  typename FFT_freqs<dim>::Vector
  FFT_freqs<dim>::get_xi(const Ccoord_t<dim> ccoord) const {
    Vector retval{};
    for (Index_t i = 0; i < dim; ++i) {
      retval(i) = this->freqs[i][ccoord[i]];
    }
    return retval;
  }

}  // namespace muFFT

#endif  // SRC_LIBMUFFT_FFT_UTILS_HH_
