/**
 * @file   fem_library.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   18 Jan 2021
 *
 * @brief  Factory functions returning FEM stencils for different element types
 *
 * Copyright © 2021 Till Junge, Martin Ladecký
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

#ifndef SRC_PROJECTION_FEM_LIBRARY_HH_
#define SRC_PROJECTION_FEM_LIBRARY_HH_

#include "fem_stencil.hh"

namespace muSpectre {
  namespace FEMLibrary {

    //! Finite-element discretisation for one-dimensional linear elements
    std::shared_ptr<FEMStencilBase> linear_1d(std::shared_ptr<CellData> cell);

    /**
     * Finite-element discretisation for two-dimensional triangular elements
     * with a right angle at the pixel origin and a north-west to south-east
     * diagonal
     */
    std::shared_ptr<FEMStencilBase>
    linear_triangle_straight(std::shared_ptr<CellData> cell);

    /**
     * Finite-element discretisation in bilinear quadrangles
     */
    std::shared_ptr<FEMStencilBase>
    bilinear_quadrangle(std::shared_ptr<CellData> cell);

    /**
     * Finite-element discretisation in trilinear hexahedra
     */
    std::shared_ptr<FEMStencilBase>
    trilinear_hexahedron(std::shared_ptr<CellData> cell);

  }  // namespace FEMLibrary
}  // namespace muSpectre

#endif  // SRC_PROJECTION_FEM_LIBRARY_HH_
