/**
 * @file   projection_default.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   14 Jan 2018
 *
 * @brief  virtual base class for default projection implementation, where the
 *         projection operator is stored as a full fourth-order tensor per
 *         k-space point (as opposed to 'smart' faster implementations, such as
 *         ProjectionFiniteStrainFast
 *
 * Copyright (C) 2018 Till Junge
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

#ifndef SRC_PROJECTION_PROJECTION_DEFAULT_HH_
#define SRC_PROJECTION_PROJECTION_DEFAULT_HH_

#include <libmugrid/field_map_static.hh>

#include <libmufft/derivative.hh>

#include "projection/projection_base.hh"

namespace muSpectre {

  /**
   * base class to inherit from if one implements a projection
   * operator that is stored in form of a fourth-order tensor of real
   * values per k-grid point
   */
  template <Index_t DimS, Index_t NbQuadPts = OneQuadPt>
  class ProjectionDefault : public ProjectionBase {
   public:
    using Parent = ProjectionBase;               //!< base class
    using Vector_t = typename Parent::Vector_t;  //!< to represent fields
    //! gradient, i.e. derivatives in each Cartesian direction
    using Gradient_t = Parent::Gradient_t;
    //! weight for each quadrature point
    using Weights_t = typename Parent::Weights_t;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>;  //!< spatial coordinates type
    //! global field collection
    using GFieldCollection_t = muGrid::GlobalFieldCollection;
    //! Real-space second order tensor fields (to be projected)
    using Field_t = typename Parent::Field_t;
    //! Fourier-space field containing the projection operator itself
    using Proj_t = muGrid::ComplexField;
    //! Fourier-space field containing the integrator
    using Integrator_t = muGrid::ComplexField;
    //! Field type on which to apply the projection
    using Proj_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS * DimS * NbQuadPts,
                               DimS * DimS * NbQuadPts, IterUnit::Pixel>;
    //! iterable form of the integrator
    using Integrator_map =
        muGrid::MatrixFieldMap<Complex, Mapping::Mut, DimS,
                               DimS * DimS * NbQuadPts, IterUnit::Pixel>;
    //! vectorized version of the Fourier-space second-order tensor field
    using Vector_map =
        muGrid::T1FieldMap<Complex, Mapping::Mut, DimS * DimS * NbQuadPts,
                           IterUnit::Pixel>;
    //! Default constructor
    ProjectionDefault() = delete;

    //! Constructor with cell sizes and formulation
    ProjectionDefault(
        muFFT::FFTEngine_ptr engine, const DynRcoord_t & lengths,
        const Gradient_t & gradient, const Weights_t & weights,
        const Formulation & form,
        const MeanControl & mean_control = MeanControl::StrainControl);

    //! Copy constructor
    ProjectionDefault(const ProjectionDefault & other) = delete;

    //! Move constructor
    ProjectionDefault(ProjectionDefault && other) = default;

    //! Destructor
    virtual ~ProjectionDefault() = default;

    //! Copy assignment operator
    ProjectionDefault & operator=(const ProjectionDefault & other) = delete;

    //! Move assignment operator
    ProjectionDefault & operator=(ProjectionDefault && other) = delete;

    //! apply the projection operator to a field
    void apply_projection(Field_t & field) final;

    //! compute the positions of the nodes of the pixels.
    //! This function is only applicable in serial.
    Field_t & integrate(Field_t & strain) final;

    //! compute the nonaffine displacements of the nodes of the pixels.
    //! This function is applicable in serial and parallel.
    Field_t & integrate_nonaffine_displacements(Field_t & strain) final;

    Eigen::Map<MatrixXXc> get_operator();

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

   protected:
    Proj_t & Gfield;        //!< field holding the operator
    Proj_map Ghat;          //!< iterable version of operator
    Integrator_t & Ifield;  //!< field holding the integrator
    Integrator_map Ihat;    //!< iterable version of integrator
  };

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_DEFAULT_HH_
