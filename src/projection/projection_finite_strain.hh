/**
 * @file   projection_finite_strain.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Apr 2019
 *
 * @brief  Class for discrete finite-strain gradient projections
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

#ifndef SRC_PROJECTION_PROJECTION_FINITE_STRAIN_HH_
#define SRC_PROJECTION_PROJECTION_FINITE_STRAIN_HH_

#include "projection/projection_default.hh"

namespace muSpectre {

  /**
   * Implements the discrete finite strain gradient projection operator
   */
  template <Index_t DimS>
  class ProjectionFiniteStrain : public ProjectionDefault<DimS> {
   public:
    using Parent = ProjectionDefault<DimS>;  //!< base class
    //! polymorphic pointer to FFT engines
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord;  //!< spatial coordinates type
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = typename Parent::Gradient_t;
    //! Field type on which to apply the projection
    using Proj_map =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimS, IterUnit::SubPt>;
    //! iterable vectorised version of the Fourier-space tensor field
    using Vector_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * DimS, 1,
                               IterUnit::SubPt>;

    //! Default constructor
    ProjectionFiniteStrain() = delete;

    //! Constructor with fft_engine and stencil
    ProjectionFiniteStrain(muFFT::FFTEngine_ptr engine,
                           const DynRcoord_t & lengths, Gradient_t gradient);

    //! Constructor with fft_engine and default (Fourier) gradient
    ProjectionFiniteStrain(muFFT::FFTEngine_ptr engine,
                           const DynRcoord_t & lengths);

    //! Copy constructor
    ProjectionFiniteStrain(const ProjectionFiniteStrain & other) = delete;

    //! Move constructor
    ProjectionFiniteStrain(ProjectionFiniteStrain && other) = default;

    //! Destructor
    virtual ~ProjectionFiniteStrain() = default;

    //! Copy assignment operator
    ProjectionFiniteStrain &
    operator=(const ProjectionFiniteStrain & other) = delete;

    //! Move assignment operator
    ProjectionFiniteStrain &
    operator=(ProjectionFiniteStrain && other) = default;

    //! initialises the fft engine (plan the transform)
    void initialise() final;
    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    std::unique_ptr<ProjectionBase> clone() const final;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_FINITE_STRAIN_HH_
