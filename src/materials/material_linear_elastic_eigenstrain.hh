/**
 * @file   material_linear_elastic_eigenstrain.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Feb 2018
 *
 * @brief linear elastic material with imposed eigenstrain and its
 *        type traits. Uses the MaterialMuSpectre facilities to keep it
 *        simple
 *
 * @section LICENSE
 *
 * Copyright © 2018 Till Junge
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

#ifndef MATERIAL_LINEAR_ELASTIC_EIGENSTRAIN_H
#define MATERIAL_LINEAR_ELASTIC_EIGENSTRAIN_H

#include "material_muSpectre_base.hh"

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic2;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic2<DimS, DimM>> {
    using InternalVariables = 
  };
}  // muSpectre

#endif /* MATERIAL_LINEAR_ELASTIC_EIGENSTRAIN_H */
