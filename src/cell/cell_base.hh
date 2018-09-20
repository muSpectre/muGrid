/**
 * @file   cell_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief Base class representing a unit cell cell with single
 *        projection operator
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

#ifndef CELL_BASE_H
#define CELL_BASE_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field.hh"
#include "common/utilities.hh"
#include "materials/material_base.hh"
#include "fft/projection_base.hh"
#include "cell/cell_traits.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <functional>
#include <array>

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

  class Cell
  {
  public:
    //! sparse matrix emulation
    using Adaptor = CellAdaptor<Cell>;

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
    Cell(const Cell &other) = default;

    //! Move constructor
    Cell(Cell &&other) = default;

    //! Destructor
    virtual ~Cell()  = default;

    //! Copy assignment operator
    Cell& operator=(const Cell &other) = default;

    //! Move assignment operator
    Cell& operator=(Cell &&other) = default;

    //! for handling double initialisations right
    bool is_initialised() const {return this->initialised;}

    //! returns the number of degrees of freedom in the cell
    virtual Dim_t get_nb_dof() const = 0;

    //! number of pixels in the cell
    virtual size_t size() const = 0;

    //! return the communicator object
    virtual const Communicator & get_communicator() const = 0;

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
     * evaluates and returns the stress and stiffness for the currently set strain
     */
    virtual std::array<ConstVector_ref, 2> evaluate_stress_tangent() = 0;


    /**
     * applies the projection operator in-place on the input vector
     */
    virtual void apply_projection(Eigen::Ref<Vector_t> vec) = 0;

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
    virtual Vector_ref evaluate_projected_directional_stiffness
      (Eigen::Ref<const Vector_t> delF) = 0;


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
     * set uniform strain (typically used to initialise problems
     */
    virtual void set_uniform_strain(const Eigen::Ref<const Matrix_t> &) = 0;

    //! get a sparse matrix view on the cell
    virtual Adaptor get_adaptor() = 0;
  protected:
    bool initialised{false}; //!< to handle double initialisation right

  private:
  };
  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template <Dim_t DimS, Dim_t DimM=DimS>
  class CellBase: public Cell
  {
  public:
    using Parent = Cell;
    using Ccoord = Ccoord_t<DimS>; //!< cell coordinates type
    using Rcoord = Rcoord_t<DimS>; //!< physical coordinates type
    //! global field collection
    using FieldCollection_t = GlobalFieldCollection<DimS>;
    //! the collection is handled in a `std::unique_ptr`
    using Collection_ptr = std::unique_ptr<FieldCollection_t>;
    //! polymorphic base material type
    using Material_t = MaterialBase<DimS, DimM>;
    //! materials handled through `std::unique_ptr`s
    using Material_ptr = std::unique_ptr<Material_t>;
    //! polymorphic base projection type
    using Projection_t = ProjectionBase<DimS, DimM>;
    //! projections handled through `std::unique_ptr`s
    using Projection_ptr = std::unique_ptr<Projection_t>;
    //! dynamic global fields
    template <typename T>
    using Field_t = TypedField<FieldCollection_t, T>;
    //! expected type for strain fields
    using StrainField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for stress fields
    using StressField_t =
      TensorField<FieldCollection_t, Real, secondOrder, DimM>;
    //! expected type for tangent stiffness fields
    using TangentField_t =
      TensorField<FieldCollection_t, Real, fourthOrder, DimM>;
    //! combined stress and tangent field
    using FullResponse_t =
      std::tuple<const StressField_t&, const TangentField_t&>;
    //! iterator type over all cell pixel's
    using iterator = typename CcoordOps::Pixels<DimS>::iterator;

    //! dynamic vector type for interactions with numpy/scipy/solvers etc.
    using Vector_t = typename Parent::Vector_t;

    //! ref to constant vector
    using ConstVector_ref = typename Parent::ConstVector_ref;

    //! output vector reference for solvers
    using Vector_ref = typename Parent::Vector_ref;

    //! dynamic array type for interactions with numpy/scipy/solvers, etc.
    template <typename T>
    using Array_t = typename Parent::Array_t<T>;

    //! dynamic array type for interactions with numpy/scipy/solvers, etc.
    template <typename T>
    using Array_ref = typename Parent::Array_ref<T>;

    //! sparse matrix emulation
    using Adaptor = Parent::Adaptor;

    //! Default constructor
    CellBase() = delete;

    //! constructor using sizes and resolution
    CellBase(Projection_ptr projection);

    //! Copy constructor
    CellBase(const CellBase &other) = delete;

    //! Move constructor
    CellBase(CellBase &&other);

    //! Destructor
    virtual ~CellBase() = default;

    //! Copy assignment operator
    CellBase& operator=(const CellBase &other) = delete;

    //! Move assignment operator
    CellBase& operator=(CellBase &&other) = default;

    /**
     * Materials can only be moved. This is to assure exclusive
     * ownership of any material by this cell
     */
    Material_t & add_material(Material_ptr mat);


    /**
     * returns a writable map onto the strain field of this cell. This
     * corresponds to the unknowns in a typical solve cycle.
     */
    virtual Vector_ref get_strain_vector() override;

    /**
     * returns a read-only map onto the stress field of this
     * cell. This corresponds to the intermediate (and finally, total)
     * solution in a typical solve cycle
     */
    virtual ConstVector_ref get_stress_vector() const override;

    /**
     * evaluates and returns the stress for the currently set strain
     */
    virtual ConstVector_ref evaluate_stress() override;

    /**
     * evaluates and returns the stress and stiffness for the currently set strain
     */
    virtual std::array<ConstVector_ref, 2> evaluate_stress_tangent() override;


    /**
     * evaluates the directional and projected stiffness (this
     * corresponds to G:K:δF in de Geus 2017,
     * http://dx.doi.org/10.1016/j.cma.2016.12.032). It seems that
     * this operation needs to be implemented with a copy in oder to
     * be compatible with scipy and EigenCG etc. (At the very least,
     * the copy is only made once)
     */
    virtual Vector_ref evaluate_projected_directional_stiffness
      (Eigen::Ref<const Vector_t> delF) override;

    //! return the template param DimM (required for polymorphic use of `Cell`
    Dim_t get_material_dim() const override final {return DimM;}

    /**
     * returns the number of rows and cols for the strain matrix type
     * (for full storage, the strain is stored in material_dim ×
     * material_dim matrices, but in symmetriy storage, it is a column
     * vector)
     */
    std::array<Dim_t, 2> get_strain_shape() const override final;

    /**
     * applies the projection operator in-place on the input vector
     */
    void apply_projection(Eigen::Ref<Vector_t> vec) override final;



    /**
     * set uniform strain (typically used to initialise problems
     */
    void set_uniform_strain(const Eigen::Ref<const Matrix_t> &) override;


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
     * vectorized version for eigen solvers, no copy, but only works
     * when fields have ArrayStore=false
     */
    Vector_ref directional_stiffness_vec(const Eigen::Ref<const Vector_t> & delF);
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

    //! returns a ref to the cell's strain field
    StrainField_t & get_strain();

    //! returns a ref to the cell's stress field
    const StressField_t & get_stress() const;

    //! returns a ref to the cell's tangent stiffness field
    const TangentField_t & get_tangent(bool create = false);

    //! returns a ref to a temporary field managed by the cell
    StrainField_t & get_managed_T2_field(std::string unique_name);

    //! returns a ref to a temporary field of real values managed by the cell
    Field_t<Real> & get_managed_real_field(std::string unique_name,
                                           size_t nb_components);

    //! returns a Array ref to a temporary field of real values managed by the cell
    virtual Array_ref<Real>
    get_managed_real_array(std::string unique_name,
                           size_t nb_components) override final;

    /**
     * returns a global field filled from local (internal) fields of
     * the materials. see `Cell::get_globalised_internal_array` for
     * details.
     */
    Field_t<Real> & get_globalised_internal_real_field(const std::string & unique_name);

    //! see `Cell::get_globalised_internal_array` for details
    Array_ref<Real>
    get_globalised_internal_real_array(const std::string & unique_name) override final;

    /**
     * general initialisation; initialises the projection and
     * fft_engine (i.e. infrastructure) but not the materials. These
     * need to be initialised separately
     */
    void initialise(FFT_PlanFlags flags = FFT_PlanFlags::estimate);

    /**
     * for materials with state variables, these typically need to be
     * saved/updated an the end of each load increment, this function
     * calls this update for each material in the cell
     */
    void save_history_variables() override final;

    iterator begin(); //!< iterator to the first pixel
    iterator end();  //!< iterator past the last pixel
    //! number of pixels in the cell
    size_t size() const override final{return pixels.size();}

    //! return the subdomain resolutions of the cell
    const Ccoord & get_subdomain_resolutions() const {
      return this->subdomain_resolutions;}
    //! return the subdomain locations of the cell
    const Ccoord & get_subdomain_locations() const {
      return this->subdomain_locations;}
    //! return the domain resolutions of the cell
    const Ccoord & get_domain_resolutions() const {
      return this->domain_resolutions;}
    //! return the sizes of the cell
    const Rcoord & get_domain_lengths() const {return this->domain_lengths;}

    /**
     * formulation is hard set by the choice of the projection class
     */
    const Formulation & get_formulation() const override final {
      return this->projection->get_formulation();}

    /**
     * get a reference to the projection object. should only be
     * required for debugging
     */
    Eigen::Map<Eigen::ArrayXXd> get_projection() {
      return this->projection->get_operator();}

    //! returns the spatial size
    constexpr static Dim_t get_sdim() {return DimS;};

    //! return a sparse matrix adaptor to the cell
    Adaptor get_adaptor() override;
    //! returns the number of degrees of freedom in the cell
    Dim_t get_nb_dof() const override {return this->size()*ipow(DimS, 2);};

    //! return the communicator object
    virtual const Communicator & get_communicator() const override {
      return this->projection->get_communicator();
    }

  protected:
    //! make sure that every pixel is assigned to one and only one material
    void check_material_coverage();

    const Ccoord & subdomain_resolutions; //!< the cell's subdomain resolutions
    const Ccoord & subdomain_locations; //!< the cell's subdomain resolutions
    const Ccoord & domain_resolutions; //!< the cell's domain resolutions
    CcoordOps::Pixels<DimS> pixels; //!< helper to iterate over the pixels
    const Rcoord & domain_lengths; //!< the cell's lengths
    Collection_ptr fields; //!< handle for the global fields of the cell
    StrainField_t & F; //!< ref to strain field
    StressField_t & P; //!< ref to stress field
    //! Tangent field might not even be required; so this is an
    //! optional ref_wrapper instead of a ref
    optional<std::reference_wrapper<TangentField_t>> K{};
    //! container of the materials present in the cell
    std::vector<Material_ptr> materials{};
    Projection_ptr projection; //!< handle for the projection operator

  private:
  };


  /**
   * lightweight resource handle wrapping a `muSpectre::Cell` or
   * a subclass thereof into `Eigen::EigenBase`, so it can be
   * interpreted as a sparse matrix by Eigen solvers
   */
  template <class Cell>
  class CellAdaptor: public Eigen::EigenBase<CellAdaptor<Cell>> {

  public:
    using Scalar = double;     //!< sparse matrix traits
    using RealScalar = double; //!< sparse matrix traits
    using StorageIndex = int;  //!< sparse matrix traits
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic,
      RowsAtCompileTime = Eigen::Dynamic,
      MaxRowsAtCompileTime = Eigen::Dynamic,
      IsRowMajor = false
    };

    //! constructor
    CellAdaptor(Cell & cell):cell{cell}{}
    //!returns the number of logical rows
    Eigen::Index rows() const {return this->cell.get_nb_dof();}
    //!returns the number of logical columns
    Eigen::Index cols() const {return this->rows();}

    //! implementation of the evaluation
    template<typename Rhs>
    Eigen::Product<CellAdaptor,Rhs,Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs>& x) const {
      return Eigen::Product<CellAdaptor,Rhs,Eigen::AliasFreeProduct>
        (*this, x.derived());
    }
    Cell & cell; //!< ref to the cell
  };

}  // muSpectre


namespace Eigen {
  namespace internal {
    //! Implementation of `muSpectre::CellAdaptor` * `Eigen::DenseVector` through a
    //! specialization of `Eigen::internal::generic_product_impl`:
    template<typename Rhs, class CellAdaptor>
    struct generic_product_impl<CellAdaptor, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
      : generic_product_impl_base<CellAdaptor,Rhs,generic_product_impl<CellAdaptor,Rhs> >
    {
      //! undocumented
      typedef typename Product<CellAdaptor,Rhs>::Scalar Scalar;

      //! undocumented
      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const CellAdaptor& lhs, const Rhs& rhs, const Scalar& /*alpha*/)
      {
        // This method should implement "dst += alpha * lhs * rhs" inplace,
        // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
        // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
        dst.noalias() += const_cast<CellAdaptor&>(lhs).cell.
          evaluate_projected_directional_stiffness(rhs);
      }
    };
  }
}


#endif /* CELL_BASE_H */
