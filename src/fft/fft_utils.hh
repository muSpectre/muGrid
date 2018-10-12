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

#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include "common/common.hh"

#include <Eigen/Dense>

#include <valarray>
#include <array>

namespace muSpectre {

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
  inline std::array<std::valarray<Real>, dim> fft_freqs(Ccoord_t<dim> sizes,
                                                 std::array<Real, dim> lengths) {
    std::array<std::valarray<Real>, dim> retval{};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] = std::move(fft_freqs(sizes[i], lengths[i]));
    }
    return retval;
  }

  /**
   * simple class encapsulating the creation, and retrieval of
   * wave vectors
   */
  template <Dim_t dim>
  class FFT_freqs
  {
  public:
    //! return type for wave vectors
    using Vector = Eigen::Matrix<Real, dim, 1>;
    //! Default constructor
    FFT_freqs() = delete;

    //! constructor with problem sizes
    FFT_freqs(Ccoord_t<dim> sizes, std::array<Real, dim> lengths)
      : freqs{fft_freqs(sizes, lengths)} {}

    //! Copy constructor
    FFT_freqs(const FFT_freqs &other) = delete;

    //! Move constructor
    FFT_freqs(FFT_freqs &&other)  = default;

    //! Destructor
    virtual ~FFT_freqs() = default;

    //! Copy assignment operator
    FFT_freqs& operator=(const FFT_freqs &other) = delete;

    //! Move assignment operator
    FFT_freqs& operator=(FFT_freqs &&other) = default;

    //! get unnormalised wave vector (in sampling units)
    inline Vector get_xi(const Ccoord_t<dim> ccoord) const;

    //! get normalised wave vector
    inline Vector get_unit_xi(const Ccoord_t<dim> ccoord) const{
      auto && xi = this->get_xi(std::move(ccoord));
      return xi/xi.norm();
    }

  protected:
    //! container for frequencies ordered by spatial dimension
    const std::array<std::valarray<Real>, dim> freqs;
  private:
  };


  template <Dim_t dim>
  typename FFT_freqs<dim>::Vector
  FFT_freqs<dim>::get_xi(const Ccoord_t<dim> ccoord) const {
    Vector retval{};
    for (Dim_t i = 0; i < dim; ++i) {
      retval(i) = this->freqs[i][ccoord[i]];
    }
    return retval;
  }
}  // muSpectre

#endif /* FFT_UTILS_H */
