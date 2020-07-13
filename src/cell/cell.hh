/**
 * @file   cell.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Sep 2019
 *
 * @brief  Class for the representation of a homogenisation problem in µSpectre
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

#ifndef SRC_CELL_CELL_HH_
#define SRC_CELL_CELL_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"

#include <libmugrid/ccoord_operations.hh>

#include <memory>
namespace muSpectre {
  /**
   * Cell adaptors implement the matrix-vector multiplication and
   * allow the system to be used like a sparse matrix in
   * conjugate-gradient-type solvers
   */
  template <class Cell>
  class CellAdaptor;

  /**
   * Base class for the representation of a homogenisatonion problem in
   * µSpectre. The `muSpectre::Cell` holds the global strain, stress and
   * (optionally) tangent moduli fields of the problem, maintains the list of
   * materials present, as well as the projection operator.
   */
  class Cell {
   public:
    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::unique_ptr<MaterialBase>;
    using Material_sptr = std::shared_ptr<MaterialBase>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<ProjectionBase>;

    //! short-hand for matrices
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

    //! ref to constant vector
    using Eigen_cmap = muGrid::RealField::Eigen_cmap;
    //! ref to  vector
    using Eigen_map = muGrid::RealField::Eigen_map;

    //! Ref to input/output vector
    using EigenVec_t = Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! Ref to input vector
    using EigenCVec_t =
        Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>>;

    //! adaptor to represent the cell as an Eigen sparse matrix
    using Adaptor = CellAdaptor<Cell>;

    //! Deleted default constructor
    Cell() = delete;

    //! Constructor from a projection operator
    explicit Cell(Projection_ptr projection,
                  SplitCell is_cell_split = SplitCell::no);

    //! Copy constructor
    Cell(const Cell & other) = delete;

    //! Move constructor
    Cell(Cell && other) = default;

    //! Destructor
    virtual ~Cell() = default;

    //! Copy assignment operator
    Cell & operator=(const Cell & other) = delete;

    //! Move assignment operator
    Cell & operator=(Cell && other) = delete;

    //! for handling double initialisations right
    bool is_initialised() const;

    //! returns the number of degrees of freedom in the cell
    Index_t get_nb_dof() const;

    //! number of pixels on this processor
    size_t get_nb_pixels() const;

    //! return the communicator object
    const muFFT::Communicator & get_communicator() const;

    /**
     * formulation is hard set by the choice of the projection class
     */
    const Formulation & get_formulation() const;

    /**
     * returns the material dimension of the problem
     */
    Index_t get_material_dim() const;

    /**
     * set uniform strain (typically used to initialise problems
     */
    void set_uniform_strain(const Eigen::Ref<const Matrix_t> &);

    /**
     * add a new material to the cell
     */
    virtual MaterialBase & add_material(Material_ptr mat);

    /**
     * By taking a material as input this function assigns all the
     * untouched(not-assigned) pixels to that material
     */
    void complete_material_assignment_simple(MaterialBase & material);

    /**
     * Given the vertices of polygonal/Polyhedral precipitate, this function
     * assign pixels 1. inside precipitate->mat_precipitate_cell, material at
     * the interface of precipitae-> to mat_precipitate & mat_matrix according
     * to the intersection of pixels with the precipitate
     */

    void make_pixels_precipitate_for_laminate_material(
        const std::vector<DynRcoord_t> & precipitate_vertices,
        MaterialBase & mat_laminate, MaterialBase & mat_precipitate_cell,
        Material_sptr mat_precipitate, Material_sptr mat_matrix);

    template <Index_t Dim, Formulation From>
    void make_pixels_precipitate_for_laminate_material_helper(
        const std::vector<DynRcoord_t> & precipitate_vertices,
        MaterialBase & mat_laminate, MaterialBase & mat_precipitate_cell,
        Material_sptr mat_precipitate, Material_sptr mat_matrix);

    //! get a sparse matrix view on the cell
    Adaptor get_adaptor();

    /**
     * freezes all the history variables of the materials
     */
    void save_history_variables();

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetric storage, it is a column
     * vector)
     */
    Shape_t get_strain_shape() const;

    /**
     * returns the number of components for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetric storage, it is a column
     * vector)
     */
    Index_t get_strain_size() const;

    //! return the spatial dimension of the discretisation grid
    const Index_t & get_spatial_dim() const;

    //! return the number of quadrature points stored per pixel
    const Index_t & get_nb_quad_pts() const;

    //! return the number of nodal points stored per pixel
    const Index_t & get_nb_nodal_pts() const;

    //! makes sure every pixel has been assigned to exactly one material
    virtual void check_material_coverage() const;

    //! initialise the projection, the materials and the global fields
    void initialise();

    //! return a const reference to the grids pixels iterator
    const muGrid::CcoordOps::DynamicPixels & get_pixels() const;

    /**
     * return an iterable proxy to this cell's field collection, iterable by
     * quadrature point
     */
    muGrid::FieldCollection::IndexIterable get_quad_pt_indices() const;

    /**
     * return an iterable proxy to this cell's field collection, iterable by
     * pixel
     */
    muGrid::FieldCollection::PixelIndexIterable get_pixel_indices() const;

    //! return a reference to the cell's strain field
    muGrid::RealField & get_strain();

    //! return a const reference to the cell's stress field
    const muGrid::RealField & get_stress() const;

    //! return a const reference to the cell's field of tangent moduli
    const muGrid::RealField & get_tangent(bool do_create = false);

    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual const muGrid::RealField & evaluate_stress();

    /**
     * evaluates and returns the stress for the currently set strain
     */
    Eigen_cmap evaluate_stress_eigen();

    /**
     * evaluates and returns the stress and tangent moduli for the currently set
     * strain
     */
    virtual std::tuple<const muGrid::RealField &, const muGrid::RealField &>
    evaluate_stress_tangent();

    /**
     * evaluates and returns the stress and tangent moduli for the currently set
     * strain
     */
    std::tuple<const Eigen_cmap, const Eigen_cmap>
    evaluate_stress_tangent_eigen();

    /**
     * collect the real-valued fields of name `unique_name` of each material in
     * the cell and write their values into a global field of same type and name
     */
    muGrid::RealField &
    globalise_real_internal_field(const std::string & unique_name);

    /**
     * collect the integer-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::IntField &
    globalise_int_internal_field(const std::string & unique_name);

    /**
     * collect the unsigned integer-valued fields of name `unique_name` of each
     * material in the cell and write their values into a global field of same
     * type and name
     */
    muGrid::UintField &
    globalise_uint_internal_field(const std::string & unique_name);

    /**
     * collect the complex-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::ComplexField &
    globalise_complex_internal_field(const std::string & unique_name);

    /**
     * collect the real-valued fields of name `unique_name` of each material in
     * the cell and write their values into a global field of same type and name
     */
    muGrid::RealField &
    globalise_real_current_field(const std::string & unique_name);

    /**
     * collect the integer-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::IntField &
    globalise_int_current_field(const std::string & unique_name);

    /**
     * collect the unsigned integer-valued fields of name `unique_name` of each
     * material in the cell and write their values into a global field of same
     * type and name
     */
    muGrid::UintField &
    globalise_uint_current_field(const std::string & unique_name);

    /**
     * collect the complex-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::ComplexField &
    globalise_complex_current_field(const std::string & unique_name);

    /**
     * collect the real-valued fields of name `unique_name` of each material in
     * the cell and write their values into a global field of same type and name
     */
    muGrid::RealField &
    globalise_real_old_field(const std::string & unique_name,
                             const size_t & nb_stpes_ago);

    /**
     * collect the integer-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::IntField & globalise_int_old_field(const std::string & unique_name,
                                               const size_t & nb_stpes_ago);

    /**
     * collect the unsigned integer-valued fields of name `unique_name` of each
     * material in the cell and write their values into a global field of same
     * type and name
     */
    muGrid::UintField &
    globalise_uint_old_field(const std::string & unique_name,
                             const size_t & nb_stpes_ago);

    /**
     * collect the complex-valued fields of name `unique_name` of each material
     * in the cell and write their values into a global field of same type and
     * name
     */
    muGrid::ComplexField &
    globalise_complex_old_field(const std::string & unique_name,
                                const size_t & nb_stpes_ago);

    //! return a reference to the cell's global fields
    muGrid::GlobalFieldCollection & get_fields();

    //! apply the cell's projection operator to field `field` (i.e., return G:f)
    void apply_projection(muGrid::TypedFieldBase<Real> & field);
    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF (note the negative sign in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032).
     */
    void evaluate_projected_directional_stiffness(
        const muGrid::TypedFieldBase<Real> & delta_strain,
        muGrid::TypedFieldBase<Real> & del_stress);

    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF (note the negative sign in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). and then adds it do the
     * values already in del_stress, scaled by alpha (i.e., del_stress +=
     * alpha*Q:K:δStrain. This function should not be used directly, as it does
     * absolutely no input checking. Rather, it is meant to be called by the
     * scaleAndAddTo function in the CellAdaptor
     */
    void add_projected_directional_stiffness(EigenCVec_t delta_strain,
                                             const Real & alpha,
                                             EigenVec_t del_stress);

    //! transitional function, use discouraged
    SplitCell get_splitness() const { return this->is_cell_split; }

    //! return a const ref to the projection implementation
    const ProjectionBase & get_projection() const;

    //! check if the pixel is inside of the cell
    bool is_point_inside(const DynRcoord_t & point) const;
    //! check if the point is inside of the cell
    bool is_pixel_inside(const DynCcoord_t & pixel) const;

    //! Check if either the material or the strain formulation introduces
    // nonlinearities into the problem
    bool was_last_eval_non_linear() const;

   protected:
    //! statically dimensioned worker for evaluating the tangent operator
    template <Index_t DimM>
    static void apply_directional_stiffness(
        const muGrid::TypedFieldBase<Real> & delta_strain,
        const muGrid::TypedFieldBase<Real> & tangent,
        muGrid::TypedFieldBase<Real> & delta_stress);

    /**
     * statically dimensioned worker for evaluating the incremental tangent
     * operator
     */
    template <Index_t DimM>
    static void add_projected_directional_stiffness_helper(
        const muGrid::TypedFieldBase<Real> & delta_strain,
        const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
        muGrid::TypedFieldBase<Real> & delta_stress);

    //! helper function for the globalise_<T>_internal_field() functions
    template <typename T>
    muGrid::TypedField<T> &
    globalise_internal_field(const std::string & unique_name);

    //! helper function for the globalise_<T>_current_field() functions
    template <typename T>
    muGrid::TypedField<T> &
    globalise_current_field(const std::string & unique_name);

    //! helper function for the globalise_<T>_old_field() functions
    template <typename T>
    muGrid::TypedField<T> & globalise_old_field(const std::string & unique_name,
                                                const size_t & nb_steps_ago);

    bool initialised{false};  //!< to handle double initialisations right

    //! container of the materials present in the cell
    std::vector<Material_ptr> materials{};

    Projection_ptr projection;  //!< handle for the projection operator

    //! handle for the global fields associated with this cell
    std::unique_ptr<muGrid::GlobalFieldCollection> fields;
    muGrid::RealField & strain;  //!< ref to strain field (compatible)
    muGrid::RealField & stress;  //!< ref to stress field

    //! Tangent field might not even be required; so this is an
    //! optional ref_wrapper instead of a ref
    optional<std::reference_wrapper<muGrid::RealField>> tangent{};

    SplitCell is_cell_split{SplitCell::no};
  };

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_HH_
