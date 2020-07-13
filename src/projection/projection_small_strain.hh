/**
 * @file   projection_small_strain.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jan 2018
 *
 * @brief  Small strain projection operator as defined in Appendix A1 of
 *         DOI: 10.1002/nme.5481 ("A finite element perspective on nonlinear
 *         FFT-based micromechanical simulations", Int. J. Numer. Meth. Engng
 *         2017; 111 :903–926)
 *
 * Copyright © 2018 Till Junge
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

#ifndef SRC_PROJECTION_PROJECTION_SMALL_STRAIN_HH_
#define SRC_PROJECTION_PROJECTION_SMALL_STRAIN_HH_

#include "projection/projection_default.hh"

namespace muSpectre {

  /**
   * Implements the small strain projection operator as defined in
   * Appendix A1 of DOI: 10.1002/nme.5481 ("A finite element
   * perspective on nonlinear FFT-based micromechanical
   * simulations", Int. J. Numer. Meth. Engng 2017; 111
   * :903–926)
   */
  template <Index_t DimS>
  class ProjectionSmallStrain : public ProjectionDefault<DimS> {
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
    ProjectionSmallStrain() = delete;

    //! Constructor with fft_engine
    ProjectionSmallStrain(muFFT::FFTEngine_ptr engine,
                          const DynRcoord_t & lengths, Gradient_t gradient);

    //! Constructor with fft_engine and default (Fourier) gradient
    ProjectionSmallStrain(muFFT::FFTEngine_ptr engine,
                          const DynRcoord_t & lengths);

    //! Copy constructor
    ProjectionSmallStrain(const ProjectionSmallStrain & other) = delete;

    //! Move constructor
    ProjectionSmallStrain(ProjectionSmallStrain && other) = default;

    //! Destructor
    virtual ~ProjectionSmallStrain() = default;

    //! Copy assignment operator
    ProjectionSmallStrain &
    operator=(const ProjectionSmallStrain & other) = delete;

    //! Move assignment operator
    ProjectionSmallStrain & operator=(ProjectionSmallStrain && other) = delete;

    //! initialises the fft engine (plan the transform)
    void initialise() final;
    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    std::unique_ptr<ProjectionBase> clone() const final;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_SMALL_STRAIN_HH_
