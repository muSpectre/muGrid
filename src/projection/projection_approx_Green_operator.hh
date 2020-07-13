/**
 * @file   projection_approx_Green_operator.hh
 *
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   01 Feb 2020
 *
 * @brief  Discrete Green's function for constant material properties
 *
 * Copyright © 2020 Martin Ladecký
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

#ifndef SRC_PROJECTION_PROJECTION_APPROX_GREEN_OPERATOR_HH_
#define SRC_PROJECTION_PROJECTION_APPROX_GREEN_OPERATOR_HH_

#include "projection/projection_default.hh"

namespace muSpectre {

  template <Index_t DimS>
  class ProjectionApproxGreenOperator : public ProjectionDefault<DimS> {
   public:
    using Parent = ProjectionDefault<DimS>;  //!< base class
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = typename Parent::Gradient_t;
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord;  //!< spatial coordinates type
    //! Fourier-space field containing the projection operator itself
    using Proj_t = muGrid::RealField;
    //! iterable operator
    using Proj_map =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimS, IterUnit::SubPt>;
    //! iterable vectorised version of the Fourier-space tensor field
    using Vector_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * DimS, 1,
                               IterUnit::SubPt>;

    //! Default constructor
    ProjectionApproxGreenOperator() = delete;

    //! Constructor with fft_engine
    ProjectionApproxGreenOperator(
        muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
        const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
            C_ref,
        Gradient_t gradient);

    //! Constructor with fft_engine and default (Fourier) gradient
    ProjectionApproxGreenOperator(
        muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
        const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
            C_ref);

    //! Copy constructor
    ProjectionApproxGreenOperator(const ProjectionApproxGreenOperator & other) =
        delete;

    //! Move constructor
    ProjectionApproxGreenOperator(ProjectionApproxGreenOperator && other) =
        default;

    //! Destructor
    virtual ~ProjectionApproxGreenOperator() = default;

    //! Copy assignment operator
    ProjectionApproxGreenOperator &
    operator=(const ProjectionApproxGreenOperator & other) = delete;

    //! Move assignment operator
    ProjectionApproxGreenOperator &
    operator=(ProjectionApproxGreenOperator && other) = delete;

    //! initialises the fft engine (plan the transform)
    void initialise() final;
    //! initialises the fft engine (plan the transform)
    void reinitialise(
        const Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
            C_ref_new);

    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    std::unique_ptr<ProjectionBase> clone() const final;

   protected:
    //! Elastic tensor of reference material
    using C_t = Eigen::Matrix<Real, DimS * DimS, DimS * DimS>;
    std::unique_ptr<C_t> C_ref_holder;
    C_t & C_ref;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_APPROX_GREEN_OPERATOR_HH_
