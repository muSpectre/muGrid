/**
 * @file   derivative.hh
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

#ifndef SRC_LIBMUFFT_DERIVATIVE_HH_
#define SRC_LIBMUFFT_DERIVATIVE_HH_

#include <memory>

#include "common/muSpectre_common.hh"
#include <libmugrid/ccoord_operations.hh>

namespace muFFT {
  /**
   * base class for projection related exceptions
   */
  class DerivativeError : public std::runtime_error {
   public:
    //! constructor
    explicit DerivativeError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit DerivativeError(const char * what) : std::runtime_error(what) {}
  };

  /**
   * Representation of a derivative
   */
  class DerivativeBase {
   public:
    //! convenience alias
    using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    //! Deleted default constructor
    DerivativeBase() = delete;

    //! constructor with spatial dimension
    explicit DerivativeBase(Dim_t spatial_dimension);

    //! Copy constructor
    DerivativeBase(const DerivativeBase & other) = default;

    //! Move constructor
    DerivativeBase(DerivativeBase && other) = default;

    //! Destructor
    virtual ~DerivativeBase() = default;

    //! Copy assignment operator
    DerivativeBase & operator=(const DerivativeBase & other) = delete;

    //! Move assignment operator
    DerivativeBase & operator=(DerivativeBase && other) = delete;

    /**
     * Return Fourier representation of the derivative as a function of the
     * phase. The phase is the wavevector times cell dimension, but lacking a
     * factor of 2 π.
     */
    virtual Complex fourier(const Vector & phase) const = 0;

   protected:
    //! spatial dimension of the problem
    Dim_t spatial_dimension;
  };

  /**
   * Representation of a derivative computed by Fourier interpolation
   */
  class FourierDerivative : public DerivativeBase {
   public:
    using Parent = DerivativeBase;  //!< base class
    //! convenience alias
    using Vector = typename Parent::Vector;

    //! Default constructor
    FourierDerivative() = delete;

    //! Constructor with raw FourierDerivative information
    explicit FourierDerivative(Dim_t spatial_dimension, Dim_t direction);

    //! Copy constructor
    FourierDerivative(const FourierDerivative & other) = default;

    //! Move constructor
    FourierDerivative(FourierDerivative && other) = default;

    //! Destructor
    virtual ~FourierDerivative() = default;

    //! Copy assignment operator
    FourierDerivative & operator=(const FourierDerivative & other) = delete;

    //! Move assignment operator
    FourierDerivative & operator=(FourierDerivative && other) = delete;

    /**
     * Return Fourier representation of the Fourier interpolated derivative.
     * This here simply returns I*2*pi*phase. (I*2*pi*wavevector is the
     * Fourier representation of the derivative.)
     */
    virtual Complex fourier(const Vector & phase) const {
      return Complex(0, 2 * muGrid::pi * phase[this->direction]);
    }

   protected:
    //! spatial direction in which to perform differentiation
    Dim_t direction;
  };

  /**
   * Representation of a finite-differences stencil
   */
  class DiscreteDerivative : public DerivativeBase {
   public:
    using Parent = DerivativeBase;  //!< base class
    //! convenience alias
    using Vector = typename Parent::Vector;

    //! Default constructor
    DiscreteDerivative() = delete;

    /**
     * Constructor with raw stencil information
     * @param nb_pts: stencil size
     * @param lbounds: relative starting point of stencil
     * @param stencil: stencil coefficients
     */
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const std::vector<Real> & stencil);

    //! Constructor with raw stencil information
    DiscreteDerivative(DynCcoord_t nb_pts, DynCcoord_t lbounds,
                       const Eigen::ArrayXd & stencil);

    //! Copy constructor
    DiscreteDerivative(const DiscreteDerivative & other) = default;

    //! Move constructor
    DiscreteDerivative(DiscreteDerivative && other) = default;

    //! Destructor
    virtual ~DiscreteDerivative() = default;

    //! Copy assignment operator
    DiscreteDerivative & operator=(const DiscreteDerivative & other) = delete;

    //! Move assignment operator
    DiscreteDerivative & operator=(DiscreteDerivative && other) = delete;

    //! Return stencil value
    Real operator()(const DynCcoord_t & dcoord) const {
      return this->stencil[muGrid::CcoordOps::get_index(this->nb_pts,
                                                        this->lbounds, dcoord)];
    }

    //! Return number of grid points in stencil
    const DynCcoord_t & get_nb_pts() const { return this->nb_pts; }

    //! Return lower stencil bound
    const DynCcoord_t & get_lbounds() const { return this->lbounds; }

    /**
     * Any translationally invariant linear combination of grid values (as
     * expressed through a "stencil") becomes a multiplication with a number
     * in Fourier space. This method returns the Fourier representation of
     * this stencil.
     */
    virtual Complex fourier(const Vector & phase) const {
      Complex s{0, 0};
      for (auto && dcoord :
           muGrid::CcoordOps::DynamicPixels(this->nb_pts, this->lbounds)) {
        const Real arg{phase.matrix().dot(eigen(dcoord).template cast<Real>())};
        s += this->operator()(dcoord) *
             std::exp(Complex(0, 2 * muGrid::pi * arg));
      }
      return s;
    }

    /**
     * Return a new stencil rolled axes. Given a stencil on a
     * three-dimensional grid with axes (x, y, z), the stencil
     * that has been "rolled" by distance one has axes (z, x, y).
     * This is a simple implementation of a rotation operation.
     * For example, given a stencil that described the derivative in
     * the x-direction, rollaxes(1) gives the derivative in the
     * y-direction and rollaxes(2) gives the derivative in the
     * z-direction.
     */
    DiscreteDerivative rollaxes(int distance = 1) const;

   protected:
    const DynCcoord_t nb_pts;  //!< Number of stencil points
    const DynCcoord_t
        lbounds;  //!< Lower bound of the finite-differences stencil
    const Eigen::ArrayXd stencil;  //!< Finite-differences stencil
  };

  /**
   * Allows inserting `muFFT::DiscreteDerivative`s into `std::ostream`s
   */
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative & derivative);

  //! convenience alias
  using Derivative_ptr = std::shared_ptr<DerivativeBase>;

  //! convenience alias
  using Gradient_t = std::vector<Derivative_ptr>;

  /**
   * convenience function to build a spatial_dimension-al gradient operator
   * using exact Fourier differentiation
   *
   * @param spatial_dimension number of spatial dimensions
   */
  Gradient_t make_fourier_gradient(const Dim_t & spatial_dimension);
}  // namespace muFFT

#endif  // SRC_LIBMUFFT_DERIVATIVE_HH_
