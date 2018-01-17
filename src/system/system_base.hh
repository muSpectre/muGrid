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
 * @section LICENSE
 *
 * Copyright © 2017 Till Junge
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

#ifndef SYSTEM_BASE_H
#define SYSTEM_BASE_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field.hh"
#include "common/utilities.hh"
#include "materials/material_base.hh"
#include "fft/projection_base.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template <Dim_t DimS, Dim_t DimM>
  class SystemBase
  {
  public:
    using Ccoord = Ccoord_t<DimS>;
    using Rcoord = Rcoord_t<DimS>;
    using FieldCollection_t = FieldCollection<DimS, DimM>;
    using Collection_ptr = std::unique_ptr<FieldCollection_t>;
    using Material_t = MaterialBase<DimS, DimM>;
    using Material_ptr = std::unique_ptr<Material_t>;
    using Projection_t = ProjectionBase<DimS, DimM>;
    using Projection_ptr = std::unique_ptr<Projection_t>;
    using StrainField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    using StressField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    using TangentField_t =
      TensorField<FieldCollection_t, Real, fourthOrder, DimM>;
    using FullResponse_t =
      std::tuple<const StressField_t&, const TangentField_t&>;
    using iterator = typename CcoordOps::Pixels<DimS>::iterator;

    //! Default constructor
    SystemBase() = delete;

    //! constructor using sizes and resolution
    SystemBase(Projection_ptr projection);

    //! Copy constructor
    SystemBase(const SystemBase &other) = delete;

    //! Move constructor
    SystemBase(SystemBase &&other) = default;

    //! Destructor
    virtual ~SystemBase() = default;

    //! Copy assignment operator
    SystemBase& operator=(const SystemBase &other) = delete;

    //! Move assignment operator
    SystemBase& operator=(SystemBase &&other) = default;

    /**
     * Materials can only be moved. This is to assure exclusive
     * ownership of any material by this system
     */
    Material_t & add_material(Material_ptr mat);

    /**
     * evaluates all materials
     */
    FullResponse_t evaluate_stress_tangent(StrainField_t & F);

    /**
     * evaluate directional stiffness (i.e. G:K:δF or G:K:δε)
     */

    StressField_t & directional_stiffness(const TangentField_t & K,
                                          const StrainField_t & delF,
                                          StressField_t & delP);
    /**
     * Evaluate directional stiffness into a temporary array and
     * return a copy. This is a costly and wasteful interface to
     * directional_stiffness and should only be used for debugging or
     * in the python interface
     */
    Eigen::ArrayXXd directional_stiffness_with_copy
      (Eigen::Ref<Eigen::ArrayXXd> delF);

    /**
     * Convenience function circumventing the neeed to use the
     * underlying projection
     */
    StressField_t & project(StressField_t & field);

    StrainField_t & get_strain();

    const StressField_t & get_stress() const;

    const TangentField_t & get_tangent(bool create = false);

    StrainField_t & get_managed_field(std::string unique_name);

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

    /**
     * formulation is hard set by the choice of the projection class
     */
    const Formulation & get_formulation() const {
      return this->projection->get_formulation();}

    bool is_initialised() const {return this->initialised;}

    Eigen::Map<Eigen::ArrayXXd> get_projection() {
      return this->projection->get_operator();}
  protected:
    //! make sure that every pixel is assigned to one and only one material
    void check_material_coverage();

    const Ccoord & resolutions;
    CcoordOps::Pixels<DimS> pixels;
    const Rcoord & lengths;
    Collection_ptr fields;
    StrainField_t & F;
    StressField_t & P;
    //! Tangent field might not even be required; so this is an
    //! optional ref_wrapper instead of a ref
    optional<std::reference_wrapper<TangentField_t>> K{};
    std::vector<Material_ptr> materials{};
    Projection_ptr projection;
    bool initialised{false};
    const Formulation form;
  private:
  };

}  // muSpectre

#endif /* SYSTEM_BASE_H */
