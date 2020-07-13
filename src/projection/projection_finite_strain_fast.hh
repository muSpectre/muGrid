/**
 * @file   projection_finite_strain_fast.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Dec 2017
 *
 * @brief  Faster alternative to ProjectionFinitestrain
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

#ifndef SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_
#define SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_

#include <libmugrid/field_collection.hh>
#include <libmugrid/mapped_field.hh>

#include <libmufft/derivative.hh>

#include "common/muSpectre_common.hh"
#include "projection/projection_base.hh"

namespace muSpectre {

  /**
   * replaces `muSpectre::ProjectionFiniteStrain` with a faster and
   * less memory-hungry alternative formulation. Use this if you don't
   * have a very good reason not to (and tell me (author) about it,
   * I'd be interested to hear it).
   */
  template <Index_t DimS, Index_t NbQuadPts = OneQuadPt>
  class ProjectionFiniteStrainFast : public ProjectionBase {
   public:
    using Parent = ProjectionBase;  //!< base class
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = muFFT::Gradient_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>;  //!< spatial coordinates type
    //! Real space second order tensor fields (to be projected)
    using Field_t = muGrid::TypedFieldBase<Real>;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = muGrid::ComplexField;
    //! iterable form of the operator
    using Proj_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * NbQuadPts, 1,
                               muGrid::IterUnit::Pixel>;
    //! iterable Fourier-space second-order tensor field
    using Grad_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS, DimS * NbQuadPts,
                               muGrid::IterUnit::Pixel>;

    //! Default constructor
    ProjectionFiniteStrainFast() = delete;

    //! Constructor with FFT engine
    ProjectionFiniteStrainFast(muFFT::FFTEngine_ptr engine,
                               const DynRcoord_t & lengths,
                               const Gradient_t & gradient);

    //! Constructor with FFT engine and default (Fourier) gradient
    ProjectionFiniteStrainFast(muFFT::FFTEngine_ptr engine,
                               const DynRcoord_t & lengths);

    //! Copy constructor
    ProjectionFiniteStrainFast(const ProjectionFiniteStrainFast & other) =
        delete;

    //! Move constructor
    ProjectionFiniteStrainFast(ProjectionFiniteStrainFast && other) = default;

    //! Destructor
    virtual ~ProjectionFiniteStrainFast() = default;

    //! Copy assignment operator
    ProjectionFiniteStrainFast &
    operator=(const ProjectionFiniteStrainFast & other) = delete;

    //! Move assignment operator
    ProjectionFiniteStrainFast &
    operator=(ProjectionFiniteStrainFast && other) = default;

    //! initialises the fft engine (plan the transform)
    void initialise() final;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) final;

    Eigen::Map<MatrixXXc> get_operator();

    //! return the gradient operator
    const Gradient_t & get_gradient() const;

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Index_t, 2> get_strain_shape() const final;

    //! get number of components to project per pixel
    constexpr static Index_t NbComponents() { return DimS * DimS * NbQuadPts; }

    //! get number of components to project per pixel
    virtual Index_t get_nb_dof_per_pixel() const { return NbComponents(); }

    //! perform a deep copy of the projector (this should never be necessary in
    //! c++)
    std::unique_ptr<ProjectionBase> clone() const final;

   protected:
    //! field of normalised wave vectors
    muGrid::MappedT1Field<Complex, Mapping::Mut, DimS * NbQuadPts,
                          IterUnit::SubPt>
        xi_field;

    /**
     * gradient (nabla) operator, can be computed using Fourier interpolation
     * or through a weighted residual
     */
    Gradient_t gradient;
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_
