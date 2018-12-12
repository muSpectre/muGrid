/**
 * @file   material_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials (constitutive models)
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

#ifndef SRC_MATERIALS_MATERIAL_BASE_HH_
#define SRC_MATERIALS_MATERIAL_BASE_HH_

#include "common/common.hh"
#include "common/field.hh"
#include "common/field_collection.hh"

#include <string>

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law) and
  /**
   * @a DimM is the material dimension (i.e., the dimension of constitutive
   * law; even for e.g. two-dimensional problems the constitutive law could
   * live in three-dimensional space for e.g. plane strain or stress problems)
   */
  template <Dim_t DimS, Dim_t DimM> class MaterialBase {
   public:
    //! typedefs for data handled by this interface
    //! global field collection for cell-wide fields, like stress, strain, etc
    using GFieldCollection_t = GlobalFieldCollection<DimS>;
    //! field collection for internal variables, such as eigen-strains,
    //! plastic strains, damage variables, etc, but also for managing which
    //! pixels the material is responsible for
    using MFieldCollection_t = LocalFieldCollection<DimS>;

    using iterator = typename MFieldCollection_t::iterator;  //!< pixel iterator
    //! polymorphic base class for fields only to be used for debugging
    using Field_t = internal::FieldBase<GFieldCollection_t>;
    //! Full type for stress fields
    using StressField_t =
        TensorField<GFieldCollection_t, Real, secondOrder, DimM>;
    //! Full type for strain fields
    using StrainField_t = StressField_t;
    //! Full type for tangent stiffness fields fields
    using TangentField_t =
        TensorField<GFieldCollection_t, Real, fourthOrder, DimM>;
    using Ccoord = Ccoord_t<DimS>;  //!< cell coordinates type
    //! Default constructor
    MaterialBase() = delete;

    //! Construct by name
    explicit MaterialBase(std::string name);

    //! Copy constructor
    MaterialBase(const MaterialBase &other) = delete;

    //! Move constructor
    MaterialBase(MaterialBase &&other) = delete;

    //! Destructor
    virtual ~MaterialBase() = default;

    //! Copy assignment operator
    MaterialBase &operator=(const MaterialBase &other) = delete;

    //! Move assignment operator
    MaterialBase &operator=(MaterialBase &&other) = delete;

    /**
     *  take responsibility for a pixel identified by its cell coordinates
     *  WARNING: this won't work for materials with additional info per pixel
     *  (as, e.g. for eigenstrain), we need to pass more parameters. Materials
     *  of this tye need to overload add_pixel
     */
    virtual void add_pixel(const Ccoord &ccooord);

    //! allocate memory, etc, but also: wipe history variables!
    virtual void initialise() = 0;

    /**
     * for materials with state variables, these typically need to be
     * saved/updated an the end of each load increment, the virtual
     * base implementation does nothing, but materials with history
     * variables need to implement this
     */
    virtual void save_history_variables() {}

    //! return the material's name
    const std::string &get_name() const;

    //! spatial dimension for static inheritance
    constexpr static Dim_t sdim() { return DimS; }
    //! material dimension for static inheritance
    constexpr static Dim_t mdim() { return DimM; }
    //! computes stress
    virtual void compute_stresses(const StrainField_t &F, StressField_t &P,
                                  Formulation form) = 0;
    /**
     * Convenience function to compute stresses, mostly for debugging and
     * testing. Has runtime-cost associated with compatibility-checking and
     * conversion of the Field_t arguments that can be avoided by using the
     * version with strongly typed field references
     */
    void compute_stresses(const Field_t &F, Field_t &P, Formulation form);
    //! computes stress and tangent moduli
    virtual void compute_stresses_tangent(const StrainField_t &F,
                                          StressField_t &P, TangentField_t &K,
                                          Formulation form) = 0;
    /**
     * Convenience function to compute stresses and tangent moduli, mostly for
     * debugging and testing. Has runtime-cost associated with
     * compatibility-checking and conversion of the Field_t arguments that can
     * be avoided by using the version with strongly typed field references
     */
    void compute_stresses_tangent(const Field_t &F, Field_t &P, Field_t &K,
                                  Formulation form);

    //! iterator to first pixel handled by this material
    inline iterator begin() { return this->internal_fields.begin(); }
    //! iterator past the last pixel handled by this material
    inline iterator end() { return this->internal_fields.end(); }
    //! number of pixels assigned to this material
    inline size_t size() const { return this->internal_fields.size(); }

    //! gives access to internal fields
    inline MFieldCollection_t &get_collection() {
      return this->internal_fields;
    }

   protected:
    const std::string name;  //!< material's name (for output and debugging)
    MFieldCollection_t internal_fields{};  //!< storage for internal variables

   private:
  };
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_BASE_HH_
