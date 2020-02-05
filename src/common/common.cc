/**
 * @file   common.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Nov 2017
 *
 * @brief  Implementation for common functions
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
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

#include <libmugrid/exception.hh>

#include "common/muSpectre_common.hh"

#include <stdexcept>
#include <iostream>

using muGrid::RuntimeError;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, Formulation f) {
    switch (f) {
    case Formulation::small_strain: {
      os << "small_strain";
      break;
    }
    case Formulation::finite_strain: {
      os << "finite_strain";
      break;
    }
    default:
      throw RuntimeError("unknown formulation.");
      break;
    }
    return os;
  }

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, StressMeasure s) {
    switch (s) {
    case StressMeasure::Cauchy: {
      os << "Cauchy";
      break;
    }
    case StressMeasure::PK1: {
      os << "PK1";
      break;
    }
    case StressMeasure::PK2: {
      os << "PK2";
      break;
    }
    case StressMeasure::Kirchhoff: {
      os << "Kirchhoff";
      break;
    }
    case StressMeasure::Biot: {
      os << "Biot";
      break;
    }
    case StressMeasure::Mandel: {
      os << "Mandel";
      break;
    }
    default:
      throw RuntimeError("a stress measure must be missing");
      break;
    }
    return os;
  }

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, StrainMeasure s) {
    switch (s) {
    case StrainMeasure::Gradient: {
      os << "Gradient";
      break;
    }
    case StrainMeasure::Infinitesimal: {
      os << "Infinitesimal";
      break;
    }
    case StrainMeasure::GreenLagrange: {
      os << "Green-Lagrange";
      break;
    }
    case StrainMeasure::Biot: {
      os << "Biot";
      break;
    }
    case StrainMeasure::Log: {
      os << "Logarithmic";
      break;
    }
    case StrainMeasure::Almansi: {
      os << "Almansi";
      break;
    }
    case StrainMeasure::RCauchyGreen: {
      os << "Right Cauchy-Green";
      break;
    }
    case StrainMeasure::LCauchyGreen: {
      os << "Left Cauchy-Green";
      break;
    }
    default:
      throw RuntimeError("a strain measure must be missing");
    }
    return os;
  }

  /* ---------------------------------------------------------------------- */
  void banner(std::string name, Uint year, std::string cpy_holder) {
    std::cout << std::endl
              << "µSpectre " << name << std::endl
              << "Copyright © " << year << "  " << cpy_holder << std::endl
              << "This program comes with ABSOLUTELY NO WARRANTY." << std::endl
              << "This is free software, and you are welcome to redistribute it"
              << std::endl
              << "under certain conditions, see the license file." << std::endl
              << std::endl;
  }

}  // namespace muSpectre
