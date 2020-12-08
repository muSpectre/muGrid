/**
 * @file   discretisation.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   02 Aug 2020
 *
 * @brief A discretisation is an instance of a FEM stencil applied to a Cell
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "common/muSpectre_common.hh"
#include "libmugrid/grid_common.hh"
#include "fem_stencil.hh"
#include "stiffness_operator.hh"

#include <Eigen/Dense>
#include <cell/cell_data.hh>

#ifndef SRC_PROJECTION_DISCRETISATION_HH_
#define SRC_PROJECTION_DISCRETISATION_HH_

namespace muSpectre {
  class Discretisation {
   public:
    //! Default constructor
    Discretisation() = delete;

    /**
     * constructor with FEM stencil
     */
    explicit Discretisation(const std::shared_ptr<FEMStencilBase> stencil);

    //! Copy constructor
    Discretisation(const Discretisation & other) = delete;

    //! Move constructor
    Discretisation(Discretisation && other) = default;

    //! Destructor
    virtual ~Discretisation() = default;

    //! Copy assignment operator
    Discretisation & operator=(const Discretisation & other) = delete;

    //! Move assignment operator
    Discretisation & operator=(Discretisation && other) = delete;

    StiffnessOperator
    get_stiffness_operator(const Index_t displacement_rank) const;

    std::unique_ptr<muGrid::RealField, muGrid::FieldDestructor<muGrid::Field>>
    compute_impulse_response(
        const Index_t & displacement_rank,
        Eigen::Ref<const Eigen::MatrixXd> ref_material_properties) const;

    Index_t get_nb_quad_pts() const;
    Index_t get_nb_nodal_pts() const;
    std::shared_ptr<CellData> get_cell();

   protected:
    std::shared_ptr<CellData> cell_ptr{nullptr};
    std::shared_ptr<FEMStencilBase> stencil_ptr{nullptr};
  };

}  // namespace muSpectre
#endif  // SRC_PROJECTION_DISCRETISATION_HH_
