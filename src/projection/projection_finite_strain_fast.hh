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

#include <libmugrid/nfield_collection.hh>
#include <libmugrid/nfield_map_static.hh>

#include "common/muSpectre_common.hh"
#include "projection/projection_base.hh"

namespace muSpectre {

  /**
   * replaces `muSpectre::ProjectionFiniteStrain` with a faster and
   * less memory-hungry alternative formulation. Use this if you don't
   * have a very good reason not to (and tell me (author) about it,
   * I'd be interested to hear it).
   */
  template <Dim_t DimS, Dim_t NbQuadPts = 1>
  class ProjectionFiniteStrainFast : public ProjectionBase<DimS> {
   public:
    using Parent = ProjectionBase<DimS>;  //!< base class
    //! polymorphic pointer to FFT engines
    using FFTEngine_ptr = typename Parent::FFTEngine_ptr;
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = typename Parent::Gradient_t;
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    using Rcoord = typename Parent::Rcoord;  //!< spatial coordinates type
    //! Real space second order tensor fields (to be projected)
    using Field_t = muGrid::RealNField;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = muGrid::ComplexNField;
    //! iterable form of the operator
    using Proj_map = muGrid::MatrixNFieldMap<Complex, false, DimS, 1,
                                             muGrid::Iteration::Pixel>;
    //! iterable Fourier-space second-order tensor field
    using Grad_map = muGrid::MatrixNFieldMap<Complex, false, DimS, DimS,
                                             muGrid::Iteration::Pixel>;

    //! Default constructor
    ProjectionFiniteStrainFast() = delete;

    //! Constructor with fft_engine
    ProjectionFiniteStrainFast(
        FFTEngine_ptr engine, Rcoord lengths,
        Gradient_t gradient = make_fourier_gradient<DimS>());

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
    void initialise(
        muFFT::FFT_PlanFlags flags = muFFT::FFT_PlanFlags::estimate) final;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) final;

    Eigen::Map<MatrixXXc> get_operator() final;

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Dim_t, 2> get_strain_shape() const final;

    //! get number of components to project per pixel
    constexpr static Dim_t NbComponents() { return DimS * DimS * NbQuadPts; }

    //! get number of components to project per pixel
    virtual Dim_t get_nb_components() const { return NbComponents(); }

   protected:
    Proj_t & xi_field;  //!< field of normalised wave vectors
    Proj_map xis;       //!< iterable normalised wave vectors
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_
