/**
 * @file   cell.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Sep 2019
 *
 * @brief  definition of the cell class, the fundamental representation of a
 *         homogenisation problem in µSpectre
 *
 * Copyright © 2019 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_CELL_CELL_HH_
#define SRC_CELL_CELL_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"
#include "cell/cell_traits.hh"

namespace muSpectre {
  /**
   * Cell adaptors implement the matrix-vector multiplication and
   * allow the system to be used like a sparse matrix in
   * conjugate-gradient-type solvers
   */
  template <class Cell>
  class CellAdaptor;

  /**
   * Base class for cells that is not templated and therefore can be
   * in solvers that see cells as runtime-polymorphic objects. This
   * allows the use of standard
   * (i.e. spectral-method-implementation-agnostic) solvers, as for
   * instance the scipy solvers
   */

  class Cell {
   public:
    //! sparse matrix emulation
    using Adaptor = CellAdaptor<Cell>;

    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::unique_ptr<MaterialBase>;

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

    //! dynamic matrix type for setting strains
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //! dynamic generic array type for interaction with numpy, i/o, etc
    template <typename T>
    using Array_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

    //! ref to dynamic generic array
    template <typename T>
    using Array_ref = Eigen::Map<Array_t<T>>;

    //! ref to constant vector
    using ConstVector_ref = Eigen::Map<const Vector_t>;

    //! output vector reference for solvers
    using Vector_ref = Eigen::Map<Vector_t>;

    //! Default constructor
    Cell() = default;

    //! Copy constructor
    Cell(const Cell & other) = default;

    //! Move constructor
    Cell(Cell && other) = default;

    //! Destructor
    virtual ~Cell() = default;

    //! Copy assignment operator
    Cell & operator=(const Cell & other) = default;

    //! Move assignment operator
    Cell & operator=(Cell && other) = default;

    //! for handling double initialisations right
    bool is_initialised() const { return this->initialised; }

    //! returns the number of degrees of freedom in the cell
    virtual Dim_t get_nb_dof() const = 0;

    //! number of pixels in the cell
    virtual size_t size() const = 0;

    //! return the communicator object
    virtual const muFFT::Communicator & get_communicator() const = 0;

    /**
     * formulation is hard set by the choice of the projection class
     */
    virtual const Formulation & get_formulation() const = 0;

    /**
     * returns the material dimension of the problem
     */
    virtual Dim_t get_material_dim() const = 0;

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    virtual std::array<Dim_t, 2> get_strain_shape() const = 0;

    /**
     * returns a writable map onto the strain field of this cell. This
     * corresponds to the unknowns in a typical solve cycle.
     */
    virtual Vector_ref get_strain_vector() = 0;

    /**
     * returns a read-only map onto the stress field of this
     * cell. This corresponds to the intermediate (and finally, total)
     * solution in a typical solve cycle
     */
    virtual ConstVector_ref get_stress_vector() const = 0;

    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual ConstVector_ref evaluate_stress() = 0;

    /**
     * evaluates and returns the stress and stiffness for the currently set
     * strain
     */
    virtual std::array<ConstVector_ref, 2> evaluate_stress_tangent() = 0;

    /**
     * applies the projection operator in-place on the input vector
     */
    virtual void apply_projection(Eigen::Ref<Vector_t> vec) = 0;

    /**
     * evaluates the projection of the input field (this corresponds
     * to G:P in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). The first time,
     * this allocates the memory for the return value, and reuses it
     * on subsequent calls
     */
    virtual Vector_ref evaluate_projection(Eigen::Ref<const Vector_t> P) = 0;

    /**
     * freezes all the history variables of the materials
     */
    virtual void save_history_variables() = 0;

    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). It seems that
     * this operation needs to be implemented with a copy in oder to
     * be compatible with scipy and EigenCG etc (At the very least,
     * the copy is only made once)
     */
    virtual Vector_ref evaluate_projected_directional_stiffness(
        Eigen::Ref<const Vector_t> delF) = 0;

    /**
     * returns a ref to a field named 'unique_name" of real values
     * managed by the cell. If the field does not yet exist, it is
     * created.
     *
     * @param unique_name name of the field. If the field already
     * exists, an array ref mapped onto it is returned. Else, a new
     * field with that name is created and returned-
     *
     * @param nb_components number of components to be stored *per
     * pixel*. For new fields any positive number can be chosen. When
     * accessing an existing field, this must correspond to the
     * existing field size, and a `std::runtime_error` is thrown if
     * this is not satisfied
     */
    virtual Array_ref<Real> get_managed_real_array(std::string unique_name,
                                                   size_t nb_components) = 0;

    /**
     * Convenience function to copy local (internal) fields of
     * materials into a global field. At least one of the materials in
     * the cell needs to contain an internal field named
     * `unique_name`. If multiple materials contain such a field, they
     * all need to be of same scalar type and same number of
     * components. This does not work for split pixel cells or
     * laminate pixel cells, as they can have multiple entries for the
     * same pixel. Pixels for which no field named `unique_name`
     * exists get an array of zeros.
     *
     * @param unique_name fieldname to fill the global field with. At
     * least one material must have such a field, or a
     * `std::runtime_error` is thrown
     */
    virtual Array_ref<Real>
    get_globalised_internal_real_array(const std::string & unique_name) = 0;

    /**
     * Convenience function to copy local (internal) state fields
     * (current state) of materials into a global field. At least one
     * of the materials in the cell needs to contain an internal field
     * named `unique_name`. If multiple materials contain such a
     * field, they all need to be of same scalar type and same number
     * of components. This does not work for split pixel cells or
     * laminate pixel cells, as they can have multiple entries for the
     * same pixel. Pixels for which no field named `unique_name`
     * exists get an array of zeros.
     *
     * @param unique_name fieldname to fill the global field with. At
     * least one material must have such a field, or a
     * `std::runtime_error` is thrown
     */
    virtual Array_ref<Real>
    get_globalised_current_real_array(const std::string & unique_name) = 0;

    /**
     * Convenience function to copy local (internal) state fields
     * (old state) of materials into a global field. At least one
     * of the materials in the cell needs to contain an internal field
     * named `unique_name`. If multiple materials contain such a
     * field, they all need to be of same scalar type and same number
     * of components. This does not work for split pixel cells or
     * laminate pixel cells, as they can have multiple entries for the
     * same pixel. Pixels for which no field named `unique_name`
     * exists get an array of zeros.
     *
     * @param unique_name fieldname to fill the global field with. At least one
     * material must have such a field, or a `std::runtime_error` is thrown
     * @param nb_steps_ago for history variables which remember more than a
     * single previous value, `nb_steps_ago` can be used to specify which old
     * value to access.
     */
    virtual Array_ref<Real>
    get_globalised_old_real_array(const std::string & unique_name,
                                  int nb_steps_ago = 1) = 0;

    /**
     * set uniform strain (typically used to initialise problems
     */
    virtual void set_uniform_strain(const Eigen::Ref<const Matrix_t> &) = 0;

    //! get a sparse matrix view on the cell
    virtual Adaptor get_adaptor() = 0;

    MaterialBase& add_material(Material_ptr mat);

   protected:
    bool initialised{false};  //!< to handle double initialisation right
    //! container of the materials present in the cell
    std::vector<Material_ptr> materials{};
  };

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_HH_
