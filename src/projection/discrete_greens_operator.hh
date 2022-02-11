/**
 * @file   discrete_greens_operator.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   09 Jul 2020
 *
 * @brief  Class implements the action of the inverse of a  block-circulant
 *         operator
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

#include "common/muSpectre_common.hh"
#include "solver/matrix_adaptor.hh"

#include <libmufft/fft_engine_base.hh>

#include <libmugrid/field_typed.hh>

#include <vector>

#ifndef SRC_PROJECTION_DISCRETE_GREENS_OPERATOR_HH_
#define SRC_PROJECTION_DISCRETE_GREENS_OPERATOR_HH_

namespace muSpectre {

  class DiscreteGreensOperator : public MatrixAdaptable {
   public:
    using Parent = MatrixAdaptable;
    using EigenCVecRef = Parent::EigenCVecRef;
    using EigenVecRef = Parent::EigenVecRef;
    using RealSpaceField_t = muGrid::TypedFieldBase<Real>;
    using FourierSpaceField_t = muGrid::ComplexField;
    //! Default constructor
    DiscreteGreensOperator() = delete;

    /**
     * Constructor
     *
     * @param engine FFT engine, used to derive domain geometry and perform
     * transforms
     *
     * @param impulse_response action of unit impulse of one degree of freedom:
     * On the origin pixel, each component represents a degree of freedom. In
     * analogy to a 1-d case, where we have a circulant matrix, this corresponds
     * to the first column. The field stores a matrix of size N_dof × N_dof per
     * pixel, where N_dof is the number of types of degrees of freedom we have
     * on a pixel (e.g., a 1-node pixel with 2-d vectorial unknows has 2×2, same
     * as a 2-node pixel in 3-d scalar unknows.)
     */
    DiscreteGreensOperator(muFFT::FFTEngine_ptr engine,
                           const RealSpaceField_t & impulse_response,
                           const Index_t & displacement_rank);

    //! Copy constructor
    DiscreteGreensOperator(const DiscreteGreensOperator & other) = delete;

    //! Move constructor
    DiscreteGreensOperator(DiscreteGreensOperator && other) = default;

    //! Destructor
    virtual ~DiscreteGreensOperator() = default;

    //! Copy assignment operator
    DiscreteGreensOperator &
    operator=(const DiscreteGreensOperator & other) = delete;

    //! Move assignment operator
    DiscreteGreensOperator &
    operator=(DiscreteGreensOperator && other) = delete;

    //! apply inverse operator
    void apply(RealSpaceField_t & field);

    //! apply inverse operator
    void apply(const RealSpaceField_t & input_field,
               RealSpaceField_t & output_field);

    //! apply inverse incementall
    void apply_increment(const RealSpaceField_t & input_field,
                         const Real & alpha, RealSpaceField_t & output_field);

    Index_t get_nb_dof() const final;

    void action_increment(EigenCVecRef delta_grad, const Real & alpha,
                          EigenVecRef del_flux) final;

    const muFFT::Communicator & get_communicator() const final;

   protected:
    muFFT::FFTEngine_ptr engine;
    Index_t nb_dof_per_pixel;
    Index_t displacement_rank;
    FourierSpaceField_t & diagonals;
    FourierSpaceField_t & field_values;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_DISCRETE_GREENS_OPERATOR_HH_
