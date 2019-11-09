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
 * Lesser General Public License for more details.
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
 *
 */

#ifndef SRC_MATERIALS_MATERIAL_BASE_HH_
#define SRC_MATERIALS_MATERIAL_BASE_HH_

#include "common/muSpectre_common.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/nfield_collection_local.hh>
#include <libmugrid/nfield_typed.hh>
#include <libmugrid/mapped_nfield.hh>

#include <string>
#include <tuple>
namespace muSpectre {
  /**
   * base class for material-related exceptions
   */
  class MaterialError : public std::runtime_error {
   public:
    //! constructor
    explicit MaterialError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit MaterialError(const char * what) : std::runtime_error(what) {}
  };

  /**
   * base class for materials
   */
  class MaterialBase {
   public:
    //! Default constructor
    MaterialBase() = delete;

    /**
     * Construct by name
     * @param name of the material
     * @param spatial_dimension is the number of spatial dimension, i.e. the
     * grid
     * @param material_dimension is the material dimension (i.e., the
     * dimension of constitutive law; even for e.g. two-dimensional problems the
     * constitutive law could live in three-dimensional space for e.g. plane
     * strain or stress problems)
     * @param nb_quad_pts is the number of quadrature points per grid cell
     */
    MaterialBase(const std::string & name, const Dim_t & spatial_dimension,
                 const Dim_t & material_dimension, const Dim_t & nb_quad_pts);

    //! Copy constructor
    MaterialBase(const MaterialBase & other) = delete;

    //! Move constructor
    MaterialBase(MaterialBase && other) = delete;

    //! Destructor
    virtual ~MaterialBase() = default;

    //! Copy assignment operator
    MaterialBase & operator=(const MaterialBase & other) = delete;

    //! Move assignment operator
    MaterialBase & operator=(MaterialBase && other) = delete;

    /**
     *  take responsibility for a pixel identified by its cell coordinates
     *  WARNING: this won't work for materials with additional info per pixel
     *  (as, e.g. for eigenstrain), we need to pass more parameters. Materials
     *  of this type need to overload add_pixel
     */
    virtual void add_pixel(const size_t & pixel_index);

    virtual void add_pixel_split(const size_t & pixel_index,
                                 const Real & ratio);

    // this function is responsible for allocating fields in case cells are
    // split or laminate
    void allocate_optional_fields(SplitCell is_cell_split = SplitCell::no);

    //! allocate memory, etc, but also: wipe history variables!
    virtual void initialise();

    /**
     * for materials with state variables, these typically need to be
     * saved/updated an the end of each load increment, the virtual
     * base implementation does nothing, but materials with history
     * variables need to implement this
     */
    virtual void save_history_variables() {}

    //! return the material's name
    const std::string & get_name() const;

    //! material dimension for  inheritance
    Dim_t get_material_dimension() { return this->material_dimension; }
    //! computes stress
    virtual void compute_stresses(const muGrid::RealNField & F,
                                  muGrid::RealNField & P, Formulation form,
                                  SplitCell is_cell_split = SplitCell::no) = 0;
    /**
     * Convenience function to compute stresses, mostly for debugging and
     * testing. Has runtime-cost associated with compatibility-checking and
     * conversion of the Field_t arguments that can be avoided by using the
     * version with strongly typed field references
     */
    void compute_stresses(const muGrid::NField & F, muGrid::NField & P,
                          Formulation form,
                          SplitCell is_cell_split = SplitCell::no);
    //! computes stress and tangent moduli
    virtual void
    compute_stresses_tangent(const muGrid::RealNField & F,
                             muGrid::RealNField & P, muGrid::RealNField & K,
                             Formulation form,
                             SplitCell is_cell_split = SplitCell::no) = 0;
    /**
     * Convenience function to compute stresses and tangent moduli, mostly for
     * debugging and testing. Has runtime-cost associated with
     * compatibility-checking and conversion of the Field_t arguments that can
     * be avoided by using the version with strongly typed field references
     */

    void compute_stresses_tangent(const muGrid::NField & F, muGrid::NField & P,
                                  muGrid::NField & K, Formulation form,
                                  SplitCell is_cell_split = SplitCell::no);

    // this function return the ratio of which the
    // input pixel is consisted of this material
    Real get_assigned_ratio(const size_t & pixel_id);

    void get_assigned_ratios(std::vector<Real> & pixel_assigned_ratios);

    // This function returns the local field containing assigned ratios of this
    // material
    auto get_assigned_ratio_field() -> muGrid::RealNField &;

    //! return and iterable proxy over the indices of this material's pixels
    typename muGrid::LocalNFieldCollection::PixelIndexIterable
    get_pixel_indices() const;

    /**
     * return and iterable proxy over the indices of this material's quadrature
     * points
     */
    typename muGrid::LocalNFieldCollection::IndexIterable
    get_quad_pt_indices() const;

    //! number of quadrature points assigned to this material
    inline Dim_t size() const { return this->internal_fields.get_nb_entries(); }

    /**
     * list the names of all internal fields
     */
    std::vector<std::string> list_fields() const;

    //! gives access to internal fields
    // TODO(junge): rename get_collection to get_fields
    inline muGrid::LocalNFieldCollection & get_collection() {
      return this->internal_fields;
    }

    using DynMatrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    virtual std::tuple<DynMatrix_t, DynMatrix_t>
    constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                             const size_t & quad_pt_index,
                             const Formulation & form) = 0;

   protected:
    const std::string name;  //!< material's name (for output and debugging)
    muGrid::LocalNFieldCollection
        internal_fields;  //!< storage for internal variables
    //! spatial dimension of the material
    Dim_t material_dimension;

    //!< field holding the assigned ratios of the material
    std::unique_ptr<muGrid::MappedScalarNField<Real, muGrid::Mapping::Mut>>
        assigned_ratio{nullptr};

    bool is_initialised{false};
  };
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_BASE_HH_
