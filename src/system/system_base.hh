/**
 * file   system_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief Base class representing a unit cell system with single
 *        projection operator
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <vector>
#include <memory>
#include <tuple>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field_collection.hh"
#include "materials/material_base.hh"
#include "fft/projection_base.hh"

#ifndef SYSTEM_BASE_H
#define SYSTEM_BASE_H

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class SystemBase
  {
  public:
    constexpr static Formulation form{Formulation::finite_strain};
    using Ccoord = Ccoord_t<DimS>;
    using Rcoord = Rcoord_t<DimS>;
    using FieldCollection_t = FieldCollection<DimS, DimM>;
    using Material_ptr = std::unique_ptr<MaterialBase<DimS, DimM>>;
    using Projection_ptr = std::unique_ptr<ProjectionBase<DimS, DimM>>;
    using StrainField_t = TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    using StressField_t = TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    using TangentField_t = TensorField<FieldCollection_t, Real, fourthOrder, DimM>;
    using FullResponse_t = std::tuple<const StressField_t&, const TangentField_t&>;
    using iterator = typename CcoordOps::Pixels<DimS>::iterator;

    //! Default constructor
    SystemBase() = delete;

    //! constructor using sizes and resolution
    SystemBase(Projection_ptr projection);

    //! Copy constructor
    SystemBase(const SystemBase &other) = delete;

    //! Move constructor
    SystemBase(SystemBase &&other) noexcept = default;

    //! Destructor
    virtual ~SystemBase() noexcept = default;

    //! Copy assignment operator
    SystemBase& operator=(const SystemBase &other) = delete;

    //! Move assignment operator
    SystemBase& operator=(SystemBase &&other) = default;

    /**
     * Materials can only be moved. This is to assure exclusive
     * ownership of any material by this system
     */
    void add_material(Material_ptr mat);

    /**
     * evaluates all materials
     */
    FullResponse_t evaluate_stress_tangent(StrainField_t & F);

    /**
     * evaluate directional stiffness (i.e. G:K:δF)
     */

    void directional_stiffness(const TangentField_t & K,
                               const StrainField_t & delF,
                               StressField_t & delP);

    /**
     * Convenience function circumventing the neeed to use the
     * underlying projection
     */
    void convolve(StressField_t & field);

    StrainField_t & get_strain();

    const StressField_t & get_stress() const;

    /**
     * general initialisation; initialises the projection and
     * fft_engine (i.e. infrastructure) but not the materials. These
     * need to be initialised separately
     */
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);
    /**
     * initialise materials (including resetting any history variables)
     */
    void initialise_materials(bool stiffness=false);

    iterator begin();
    iterator end();
    size_t size() const {return pixels.size();}

    const Ccoord & get_resolutions() const {return this->resolutions;}
    const Rcoord & get_lengths() const {return this->lengths;}
  protected:
    //! make sure that every pixel is assigned to one and only one material
    void check_material_coverage();

    const Ccoord & resolutions;
    CcoordOps::Pixels<DimS> pixels;
    const Rcoord & lengths;
    FieldCollection_t fields;
    StrainField_t & F;
    StressField_t & P;
    //! Tangent field migth not even be required; so this is a
    //! pointer instead of a ref
    TangentField_t * K_ptr;
    std::vector<Material_ptr> materials{};
    Projection_ptr projection;
    bool is_initialised{false};
  private:
  };

}  // muSpectre

#endif /* SYSTEM_BASE_H */
