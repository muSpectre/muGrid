/**
 * @file   cell_data.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   30 Apr 2020
 *
 * @brief  Manager of all data structures necessary to discretise a PDE problem.
 *         These data structures include the global fields of input and output
 *         unknowns (e.g., nodal displacements and forces), as well the
 *         constitutive laws and the pixels/elements they are responsible for,
 *         including their internal variables.
 *
 * Copyright © 2020 Till Junge
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

#ifndef SRC_CELL_CELL_DATA_HH_
#define SRC_CELL_CELL_DATA_HH_

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"

#include <libmugrid/state_field.hh>
#include <libmugrid/physics_domain.hh>
#include <libmugrid/units.hh>
#include <libmugrid/exception.hh>
#include <libmufft/fft_engine_base.hh>

#include <memory>
#include <map>

namespace muSpectre {

  /**
   * base class for cell data-related exceptions
   */
  class CellDataError : public muGrid::RuntimeError {
   public:
    //! constructor
    explicit CellDataError(const std::string & what)
        : muGrid::RuntimeError(what) {}
    //! constructor
    explicit CellDataError(const char * what) : muGrid::RuntimeError(what) {}
  };

  class CellData : public std::enable_shared_from_this<CellData> {
   private:
    //! Deleted default constructor
    CellData() = delete;

    /**
     * constructor with domain size and domain decomposition for distributed
     * memory calculations
     */
    CellData(std::shared_ptr<muFFT::FFTEngineBase> engine,
             const DynRcoord_t & domain_lengths);

   public:
    /**
     * factory function from a FFT engine
     */
    static std::shared_ptr<CellData>
    make(std::shared_ptr<muFFT::FFTEngineBase> engine,
         const DynRcoord_t & domain_lengths);
    /**
     * factory function with domain size (sequential calculations), creates
     * default sequential FFT engine
     */
    static std::shared_ptr<CellData>
    make(const DynCcoord_t & nb_domain_grid_pts,
         const DynRcoord_t & domain_lengths);

#ifdef WITH_MPI
    /**
     * factory_function with domain size and domain decomposition for
     * distributed memory calculations, creates default MPI FFT engine
     */
    static std::shared_ptr<CellData> make_parallel(
        const DynCcoord_t & nb_domain_grid_pts,
        const DynRcoord_t & domain_lengths,
        const muFFT::Communicator & communicator = muFFT::Communicator());
#endif

    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::shared_ptr<MaterialBase>;

    using DomainMaterialsMap_t =
        std::map<muGrid::PhysicsDomain, std::vector<Material_ptr>>;

    //! Copy constructor
    CellData(const CellData & other) = delete;

    //! Move constructor
    CellData(CellData && other) = default;

    //! Destructor
    virtual ~CellData() = default;

    //! Copy assignment operator
    CellData & operator=(const CellData & other) = delete;

    //! Move assignment operator
    CellData & operator=(CellData && other) = default;

    //! returns a reference to the field collection
    muGrid::GlobalFieldCollection & get_fields();

    //! returns a reference to the field collection
    const muGrid::GlobalFieldCollection & get_fields() const;

    /**
     * adds a new material to the cell and returns a reference to the same
     * material
     */
    virtual MaterialBase & add_material(Material_ptr mat);

    /**
     * check whether every pixel has been assigned exacty one material of each
     * physics domain
     */
    void check_material_coverage() const;

    //! return the spatial dimension of the discretisation grid
    const Dim_t & get_spatial_dim() const;

    //! return the dimension of the material
    const Dim_t & get_material_dim() const;

    //! return the number of quadrature points stored per pixel
    const Index_t & get_nb_quad_pts() const;

    //! return the number of nodal points stored per pixel
    const Index_t & get_nb_nodal_pts() const;

    //! return the number of quadrature points stored per pixel
    void set_nb_quad_pts(const Index_t & nb_quad_pts);

    //! return the number of nodal points stored per pixel
    void set_nb_nodal_pts(const Index_t & nb_nodal_pts);

    //! return whether the number of quadrature points stored per pixel is set
    bool has_nb_quad_pts() const;

    //! return whether the number of nodal points stored per pixel is set
    bool has_nb_nodal_pts() const;

    //! return a const reference to the grid's pixels iterable
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

    //! check if the pixel is inside of the cell
    bool is_point_inside(const DynRcoord_t & point) const;
    //! check if the point is inside of the cell
    bool is_pixel_inside(const DynCcoord_t & pixel) const;

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

    //! return the communicator object
    const muFFT::Communicator & get_communicator() const;

    //! give access to the stored materials
    DomainMaterialsMap_t & get_domain_materials();

    //! Check if one of the materials introduces nonlinearities into the problem
    bool was_last_eval_non_linear() const;

    /**
     * returns the global number of grid points in each direction of the cell
     */
    const DynCcoord_t & get_nb_domain_grid_pts() const;

    /**
     * returns the process-local number of grid points in each direction of the
     * cell
     */
    const DynCcoord_t & get_nb_subdomain_grid_pts() const;

    //! returns the process-local locations of the cell
    const DynCcoord_t & get_subdomain_locations() const;

    //! return the physical size of the computational domain
    const DynRcoord_t & get_domain_lengths() const;

    //! return the fft engine
    std::shared_ptr<muFFT::FFTEngineBase> get_FFT_engine();

    /**
     * freezes all the history variables of the materials
     */
    void save_history_variables();

   protected:
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

    // Cell's (possibly shared) FFT engine
    std::shared_ptr<muFFT::FFTEngineBase> fft_engine;

    // different constitutive model categories are identified by their rank
    // and the units of the input and output fields. Input units multiplied
    // by output units must equal energy units for the model to be plausible
    //! map for constitutive models
    DomainMaterialsMap_t domain_materials{};

    //! geometric size of domain
    DynRcoord_t domain_lengths;

    //! handle for the global fields associated with this cell
    std::unique_ptr<muGrid::GlobalFieldCollection> fields;

    //! communicator for distributed calculations
    muFFT::Communicator communicator;

    /**
     * dimension of the materinot necessarily same as the problem's spatial
     * dimension
     */
    Dim_t material_dim{muGrid::Unknown};
  };

  //! convenience alias for CellData
  using CellData_ptr = std::shared_ptr<CellData>;

}  // namespace muSpectre

#endif  // SRC_CELL_CELL_DATA_HH_
