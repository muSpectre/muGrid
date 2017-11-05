/**
 * file   materials_toolbox.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Nov 2017
 *
 * @brief  collection of common continuum mechanics tools
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

#ifndef MATERIALS_TOOLBOX_H
#define MATERIALS_TOOLBOX_H

#include <exception>
#include <sstream>
#include <iostream>
#include "common/common.hh"
#include "common/tensor_algebra.hh"

namespace muSpectre {

  namespace MatTB {
    class MaterialsToolboxError:public std::runtime_error{
    public:
      explicit MaterialsToolboxError(const std::string& what)
        :std::runtime_error(what){}
      explicit MaterialsToolboxError(const char * what)
        :std::runtime_error(what){}
    };

    /* ---------------------------------------------------------------------- */
    //! Material laws can declare which type of stress measure they provide,
    //! and µSpectre will handle conversions
    enum class StressMeasure {
      Cauchy, PK1, PK2, Kirchhoff, Biot, Mandel};
    std::ostream & operator<<(std::ostream & os, StressMeasure s) {
      switch (s) {
      case StressMeasure::Cauchy:    {os << "Cauchy";    break;}
      case StressMeasure::PK1:       {os << "PK1";       break;}
      case StressMeasure::PK2:       {os << "PK2";       break;}
      case StressMeasure::Kirchhoff: {os << "Kirchhoff"; break;}
      case StressMeasure::Biot:      {os << "Biot";      break;}
      case StressMeasure::Mandel:    {os << "Mandel";    break;}
      default:
        throw MaterialsToolboxError
          ("a stress measure must be missing");
        break;
      }
      return os;
    }

    /* ---------------------------------------------------------------------- */
    //! Material laws can declare which type of strain measure they require and
    //! µSpectre will provide it
    enum class StrainMeasure {
      GreenLagrange, Biot, Log, Almansi, RCauchyGreen, LCauchyGreen};
    std::ostream & operator<<(std::ostream & os, StrainMeasure s) {
      switch (s) {
      case StrainMeasure::GreenLagrange: {os << "Green-Lagrange"; break;}
      case StrainMeasure::Biot:          {os << "Biot"; break;}
      case StrainMeasure::Log:           {os << "Logarithmic"; break;}
      case StrainMeasure::Almansi:       {os << "Almansi"; break;}
      case StrainMeasure::RCauchyGreen:  {os << "Right Cauchy-Green"; break;}
      case StrainMeasure::LCauchyGreen:  {os << "Left Cauchy-Green"; break;}
      default:
        throw MaterialsToolboxError
          ("a strain measure must be missing");
        break;
      }
      return os;
    }

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM>
    auto PK2_stress = [](auto && stress, auto && Strain)  -> decltype(auto) {
      // the following test always fails to generate a compile-time error
      static_assert((StressM == StressMeasure::Cauchy) &&
                    (StressM == StressMeasure::PK1),
                    "The requested Stress conversion is not implemented. "
                    "You either made a programming mistake or need to "
                    "implement it as a specialisation of this function. "
                    "See PK2stress<PK1,T1, T2> for an example.");
    };

    /* ---------------------------------------------------------------------- */
    template <>
    auto PK2_stress<StressMeasure::PK1> =
      [](auto && S, auto && F) -> decltype(auto) {
      return F*S;
    };

    /* ---------------------------------------------------------------------- */
    //! set of functions returning ∂strain/∂F (where F is the transformation
    //! gradient. (e.g. for a material law expressed as PK2 stress S as
    //! function of Green-Lagrange strain E, this would return F^T ⊗̲ I) (⊗_
    //! refers to Curnier's notation) WARNING: this function assumes that your
    //! stress measure has the two minor symmetries due to stress equilibrium
    //! and the major symmetry due to the existance of an elastic potential
    //! W(E). Do not rely on this function if you are using exotic stress
    //! measures
    template<StrainMeasure StrainM>
    auto StrainDerivative = [](auto && F) decltype(auto) {
      // the following test always fails to generate a compile-time error
      static_assert((StrainM == StrainMeasure::Almansi) &&
                    (StrainM == StrainMeasure::Biot),
                    "The requested StrainDerivative calculation is not "
                    "implemented. You either made a programming mistake or "
                    "need to implement it as a specialisation of this "
                    "function. See "
                    "StrainDerivative<StrainMeasure::GreenLagrange> for an "
                    "example.");
    };

    /* ---------------------------------------------------------------------- */
    template<>
    auto StrainDerivative<StrainMeasure::GreenLagrange> =
      [] (auto && F) -> decltype(auto) {
      constexpr size_t order{2};
      constexpr Dim_t dim{F.Dimensions[0]};
      return Tensors::outer_under<dim>
        (F.shuffle(std::array<Dim_t, order>{1,0}), Tensors::I2<dim>());
    };

    /* ---------------------------------------------------------------------- */
    //! function returning expressions for PK2 stress and stiffness
    template <class Tens1, class Tens2,
              StressMeasure StressM, StrainMeasure StrainM>
    decltype(auto) PK2_stress_stiffness(Tens1 && stress, Tens2 && F) {
    }

  }  // MatTB

}  // muSpectre

#endif /* MATERIALS_TOOLBOX_H */
