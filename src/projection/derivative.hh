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
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_PROJECTION_DERIVATIVE_HH_
#define SRC_PROJECTION_DERIVATIVE_HH_

#include <memory>

#include "common/muSpectre_common.hh"
#include "libmugrid/ccoord_operations.hh"

namespace muSpectre {

  /**
   * base class for projection related exceptions
   */
  class ProjectionError : public std::runtime_error {
   public:
    //! constructor
    explicit ProjectionError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit ProjectionError(const char * what) : std::runtime_error(what) {}
  };

  /**
   * Representation of a derivative
   */
  template <Dim_t DimS>
  class DerivativeBase {
   public:
    constexpr static Dim_t sdim{DimS};  //!< spatial dimension of the cell
    //! cell coordinates type
    using Ccoord = Ccoord_t<DimS>;
    using Vector = Eigen::Matrix<Real, DimS, 1>;

    //! Default constructor
    DerivativeBase() = default;

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
     * factor of 2 pi.
     */
    virtual Complex fourier(const Vector & phase) const = 0;
  };

  /**
   * Representation of a derivative computed by Fourier interpolation
   */
  template <Dim_t DimS>
  class FourierDerivative : public DerivativeBase<DimS> {
   public:
    using Parent = DerivativeBase<DimS>;  //!< base class
    //! cell coordinates type
    using Ccoord = typename Parent::Ccoord;
    using Vector = typename Parent::Vector;

    //! Default constructor
    FourierDerivative() = delete;

    //! Constructor with raw FourierDerivative information
    explicit FourierDerivative(Dim_t direction);

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
      return Complex(0, 2*muGrid::pi*phase[this->direction]);
    }

   protected:
    Dim_t direction;
  };

  /**
   * Representation of a finite-differences stencil
   */
  template <Dim_t DimS>
  class DiscreteDerivative : public DerivativeBase<DimS> {
   public:
    using Parent = DerivativeBase<DimS>;  //!< base class
    //! cell coordinates type
    using Ccoord = typename Parent::Ccoord;
    using Vector = typename Parent::Vector;

    //! Default constructor
    DiscreteDerivative() = delete;

    /**
     * Constructor with raw stencil information
     * @param nb_pts: stencil size
     * @param lbounds: relative starting point of stencil
     * @param stencil: stencil coefficients
     */
    DiscreteDerivative(Ccoord nb_pts, Ccoord lbounds,
                       const std::vector<Real> & stencil);

    //! Constructor with raw stencil information
    DiscreteDerivative(Ccoord nb_pts, Ccoord lbounds,
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
    Real operator()(const Ccoord & dcoord) const {
      return this->stencil[
          muGrid::CcoordOps::get_index(this->nb_pts, this->lbounds, dcoord)];
    }

    //! Return number of grid points in stencil
    const Ccoord & get_nb_pts() const {
      return this->nb_pts;
    }

    //! Return lower stencil bound
    const Ccoord & get_lbounds() const {
      return this->lbounds;
    }

    /**
     * Any translationally invariant linear combination of grid values (as
     * expressed through a "stencil") becomes a multiplication with a number
     * in Fourier space. This method returns the Fourier representation of
     * this stencil.
     */
    virtual Complex fourier(const Vector & phase) const {
      Complex s{0, 0};
      for (auto && dcoord :
           muGrid::CcoordOps::Pixels<DimS>(this->nb_pts, this->lbounds)) {
        const Real arg{phase.matrix().dot(eigen(dcoord).template cast<Real>())};
        s += this->operator()(dcoord) *
            std::exp(Complex(0, 2*muGrid::pi * arg));
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
    const Ccoord nb_pts;   //!< Number of stencil points
    const Ccoord lbounds;  //!< Lower bound of the finite-differences stencil
    const Eigen::ArrayXd stencil;  //!< Finite-differences stencil
  };

  /**
   * Allows inserting `muSpectre::DiscreteDerivative`s into `std::ostream`s
   */
  template <Dim_t DimS>
  std::ostream & operator<<(std::ostream & os,
                            const DiscreteDerivative<DimS> & derivative) {
    const typename DiscreteDerivative<DimS>::Ccoord &
        nb_pts{derivative.get_nb_pts()};
    const typename DiscreteDerivative<DimS>::Ccoord &
        lbounds{derivative.get_lbounds()};
    os << "{ ";
    muGrid::operator<<(os, nb_pts);
    os << " ";
    muGrid::operator<<(os, lbounds);
    os << " ";
    for (auto && pixel : muGrid::CcoordOps::Pixels<DimS>(nb_pts, lbounds)) {
      os << derivative(pixel) << " ";
    }
    os << "}";
    return os;
  }

  template<Dim_t DimS>
  using Derivative_ptr = std::shared_ptr<DerivativeBase<DimS>>;

  template<size_t DimS>
  using Gradient_t = std::array<Derivative_ptr<static_cast<Dim_t>(DimS)>, DimS>;

  template<Dim_t DimS>
  Gradient_t<DimS> make_fourier_gradient() {
    Gradient_t<DimS> && g{};
    for (Dim_t dim = 0; dim < DimS; ++dim) {
      g[dim] = std::make_shared<FourierDerivative<DimS>>(dim);
    }
    return std::move(g);
  }
}  // namespace muSpectre

#endif  // SRC_PROJECTION_DERIVATIVE_HH_
