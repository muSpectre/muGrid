/**
 * @file   projection_small_strain.cc
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

#ifndef SRC_FFT_PROJECTION_SMALL_STRAIN_HH_
#define SRC_FFT_PROJECTION_SMALL_STRAIN_HH_

#include "fft/projection_default.hh"

namespace muSpectre {

  /**
   * Implements the small strain projection operator as defined in
   * Appendix A1 of DOI: 10.1002/nme.5481 ("A finite element
   * perspective on nonlinear FFT-based micromechanical
   * simulations", Int. J. Numer. Meth. Engng 2017; 111
   * :903–926)
   */
  template <Dim_t DimS, Dim_t DimM>
  class ProjectionSmallStrain : public ProjectionDefault<DimS, DimM> {
   public:
    using Parent = ProjectionDefault<DimS, DimM>;  //!< base class
    //! polymorphic pointer to FFT engines
    using FFTEngine_ptr = typename Parent::FFTEngine_ptr;
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord;  //!< spatial coordinates type
    //! local field collection (for Fourier-space representations)
    using LFieldCollection_t = LocalFieldCollection<DimS>;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = TensorField<LFieldCollection_t, Real, fourthOrder, DimM>;
    //! iterable operator
    using Proj_map = T4MatrixFieldMap<LFieldCollection_t, Real, DimM>;
    //! iterable vectorised version of the Fourier-space tensor field
    using Vector_map =
        MatrixFieldMap<LFieldCollection_t, Complex, DimM * DimM, 1>;

    //! Default constructor
    ProjectionSmallStrain() = delete;

    //! Constructor with fft_engine
    ProjectionSmallStrain(FFTEngine_ptr engine, Rcoord lengths);

    //! Copy constructor
    ProjectionSmallStrain(const ProjectionSmallStrain &other) = delete;

    //! Move constructor
    ProjectionSmallStrain(ProjectionSmallStrain &&other) = default;

    //! Destructor
    virtual ~ProjectionSmallStrain() = default;

    //! Copy assignment operator
    ProjectionSmallStrain &
    operator=(const ProjectionSmallStrain &other) = delete;

    //! Move assignment operator
    ProjectionSmallStrain &operator=(ProjectionSmallStrain &&other) = delete;

    //! initialises the fft engine (plan the transform)
    void
    initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate) final;
  };

}  // namespace muSpectre

#endif  // SRC_FFT_PROJECTION_SMALL_STRAIN_HH_
