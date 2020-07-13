/**
 * @file   field_typed.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Aug 2019
 *
 * @brief  Field classes for which the scalar type has been defined
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_FIELD_TYPED_HH_
#define SRC_LIBMUGRID_FIELD_TYPED_HH_

#include "field.hh"
#include "grid_common.hh"

#include <Eigen/Dense>

#include <vector>
#include <memory>

namespace muGrid {

  //! forward declaration
  template <typename T, Mapping Mutability>
  class FieldMap;
  //! forward declaration
  template <typename T>
  class TypedFieldBase;

  template <typename T>
  class TypedFieldBase : public Field {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");

   protected:
    /**
     * Simple structure used to allow for lazy evaluation of the unary '-' sign.
     * When assiging the the negative of a field to another, as in field_a =
     * -field_b, this structure allows to implement this operation without
     * needing a temporary object holding the negative value of field_b.
     */
    struct Negative {
      //! field on which the unary '-' was applied
      const TypedFieldBase & field;
    };

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a `FieldCollection. The `Field` constructor is protected to
     * ensure this. Fields are instantiated through the `register_field`
     * methods FieldCollection.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param nb_components number of components to store per quadrature
     *        point
     */
    TypedFieldBase(const std::string & unique_name,
                   FieldCollection & collection, Index_t nb_components,
                   const std::string & sub_division, const Unit & unit)
        : Parent{unique_name,
                 collection,
                 nb_components,
                 sub_division,
                 unit} {}

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a `FieldCollection. The `Field` constructor is protected to
     * ensure this. Fields are instantiated through the `register_field`
     * methods FieldCollection.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param components_shape number of components to store per quadrature
     * point
     */
    TypedFieldBase(const std::string & unique_name,
                   FieldCollection & collection,
                   const Shape_t & components_shape,
                   const std::string & sub_division, const Unit & unit)
        : Parent{unique_name,
                 collection,
                 components_shape,
                 sub_division,
                 unit} {}

   public:
    //! stored scalar type
    using Scalar = T;

    //! Eigen type used to represent the field's data
    using EigenRep_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    //! eigen map (handle for EigenRep_t)
    using Eigen_map = Eigen::Map<EigenRep_t>;

    //! eigen const map (handle for EigenRep_t)
    using Eigen_cmap = Eigen::Map<const EigenRep_t>;

    //! base class
    using Parent = Field;

    //! Default constructor
    TypedFieldBase() = delete;

    //! Copy constructor
    TypedFieldBase(const TypedFieldBase & other) = delete;

    //! Move constructor
    TypedFieldBase(TypedFieldBase && other) = default;

    //! Destructor
    virtual ~TypedFieldBase() = default;

    //! Move assignment operator
    TypedFieldBase & operator=(TypedFieldBase && other) = delete;

    //! Copy assignment operator
    TypedFieldBase & operator=(const TypedFieldBase & other);

    //! Copy assignment operator
    TypedFieldBase & operator=(const Negative & other);

    //! Copy assignment operators
    TypedFieldBase & operator=(const EigenRep_t & other);

    //! Unary negative
    Negative operator-() const;

    //! addition assignment
    TypedFieldBase & operator+=(const TypedFieldBase & other);

    //! subtraction assignment
    TypedFieldBase & operator-=(const TypedFieldBase & other);

    const std::type_info & get_stored_typeid() const final { return typeid(T); }

    //! return a vector map onto the underlying data
    Eigen_map eigen_vec();
    //! return a const vector map onto the underlying data
    Eigen_cmap eigen_vec() const;

    /**
     * return a matrix map onto the underlying data with one column per
     * quadrature point
     */
    Eigen_map eigen_sub_pt();
    /**
     * return a const matrix map onto the underlying data with one column per
     * quadrature point
     */
    Eigen_cmap eigen_sub_pt() const;

    /**
     * return a matrix map onto the underlying data with one column per
     * pixel
     */
    Eigen_map eigen_pixel();
    /**
     * return a const matrix map onto the underlying data with one column per
     * pixel
     */
    Eigen_cmap eigen_pixel() const;

    template <typename T_int, Mapping Mutability>
    friend class FieldMap;

    /**
     * convenience function returns a map of this field, iterable per pixel.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components`
     * × `nb_quad_pts` is used
     */
    FieldMap<T, Mapping::Mut> get_pixel_map(const Index_t & nb_rows = Unknown);

    /**
     * convenience function returns a const map of this field, iterable per
     * pixel.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components`
     * × `nb_quad_pts` is used
     */
    FieldMap<T, Mapping::Const>
    get_pixel_map(const Index_t & nb_rows = Unknown) const;

    /**
     * convenience function returns a map of this field, iterable per quadrature
     * point.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    FieldMap<T, Mapping::Mut>
    get_sub_pt_map(const Index_t & nb_rows = Unknown);

    /**
     * convenience function returns a const  map of this field, iterable per
     * quadrature point.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    FieldMap<T, Mapping::Const>
    get_sub_pt_map(const Index_t & nb_rows = Unknown) const;

    //! get the raw data ptr. don't use unless interfacing with external libs
    T * data() const;

   protected:
    //! back-end for the public non-const eigen_XXX functions
    Eigen_map eigen_map(const Index_t & nb_rows, const Index_t & nb_cols);
    //! back-end for the public const eigen_XXX functions
    Eigen_cmap eigen_map(const Index_t & nb_rows,
                         const Index_t & nb_cols) const;
    //! set the data_ptr
    void set_data_ptr(T * ptr);
    /**
     * in order to accomodate both registered fields (who own and
     * manage their data) and unregistered temporary field proxies
     * (piggy-backing on a chunk of existing memory as e.g., a numpy
     * array) *efficiently*, the `get_ptr_to_entry` methods need to be
     * branchless. this means that we cannot decide on the fly whether
     * to return pointers pointing into values or into alt_values, we
     * need to maintain an (shudder) raw data pointer that is set
     * either at construction (for unregistered fields) or at any
     * resize event (which may invalidate existing pointers). For the
     * coder, this means that they need to be absolutely vigilant that
     * *any* operation on the values vector that invalidates iterators
     * needs to be followed by an update of data_ptr, or we will get
     * super annoying memory bugs.
     */
    T * data_ptr{};
  };

  /**
   * A `muGrid::TypedField` holds a certain number of components (scalars of
   * type `T` per quadrature point of a `muGrid::FieldCollection`'s domain.
   *
   * @tparam T type of scalar to hold. Must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`.
   */
  template <typename T>
  class TypedField : public TypedFieldBase<T> {
   protected:
    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_components number of components to store per quadrature
     * point
     * @param collection reference to the holding field collection.
     */
    TypedField(const std::string & unique_name, FieldCollection & collection,
               const Index_t & nb_components,
               const std::string & sub_division, const Unit & unit)
        : Parent{unique_name, collection, nb_components, sub_division,
                 unit} {}

    /**
     * `Field`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a FieldCollection. The `Field` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param collection reference to the holding field collection.
     * @param components_shape number of components to store per quadrature
     * point
     * @param storage_oder in-memory storage order of the components
     */
    TypedField(const std::string & unique_name, FieldCollection & collection,
               const Shape_t & components_shape,
               const std::string & sub_division, const Unit & unit)
        : Parent{unique_name, collection, components_shape, sub_division,
                 unit} {}

   public:
    //! base class
    using Parent = TypedFieldBase<T>;

    //! Eigen type to represent the field's data
    using EigenRep_t = typename Parent::EigenRep_t;

    //! convenience alias
    using Negative = typename Parent::Negative;

    //! Default constructor
    TypedField() = delete;

    //! Copy constructor
    TypedField(const TypedField & other) = delete;

    //! Move constructor
    TypedField(TypedField && other) = delete;

    //! Destructor
    virtual ~TypedField() = default;

    //! Move assignment operator
    TypedField & operator=(TypedField && other) = delete;

    //! Copy assignment operator
    TypedField & operator=(const TypedField & other);

    //! Copy assignment operator
    TypedField & operator=(const Parent & other);

    //! Copy assignment operator
    TypedField & operator=(const Negative & other);

    //! Copy assignment operator
    TypedField & operator=(const EigenRep_t & other);

    void set_zero() final;
    void set_pad_size(const size_t & pad_size) final;

    //! cast a reference to a base type to this type, with full checks
    static TypedField & safe_cast(Field & other);

    //! cast a const reference to a base type to this type, with full checks
    static const TypedField & safe_cast(const Field & other);

    /**
     * cast a reference to a base type to this type safely, plus check whether
     * it has the right number of components
     */
    static TypedField & safe_cast(Field & other,
                                  const Index_t & nb_components,
                                  const std::string & sub_division);

    /**
     * cast a const reference to a base type to this type safely, plus check
     * whether it has the right number of components
     */
    static const TypedField & safe_cast(const Field & other,
                                        const Index_t & nb_components,
                                        const std::string & sub_division);

    size_t get_buffer_size() const final;

    /**
     * add a new scalar value at the end of the field (incurs runtime cost, do
     * not use this in any hot loop)
     */
    void push_back(const T & value);

    /**
     * add a new non-scalar value at the end of the field (incurs runtime cost,
     * do not use this in any hot loop)
     */
    void
    push_back(const Eigen::Ref<
              const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> & value);

    /**
     * perform a full copy of a field
     * @param new_name the name under which the new field will be stored
     * @param allow_overwrite by default, this function throws an error if the
     * destination field already exists to avoid accidental clobbering of
     * fields. If set to true, the copy will be made into the existing field.
     */
    TypedField & clone(const std::string & new_name,
                       const bool & allow_overwrite = false) const;

    //! give access to collections
    friend FieldCollection;

   protected:
    void resize() final;

    //! storage of the raw field data
    std::vector<T> values{};
  };

  /**
   * Wrapper class providing a field view of existing memory. This is
   * particularly useful when  dealing with input from external libraries (e.g.,
   * numpy arrays)
   */
  template <typename T>
  class WrappedField : public TypedFieldBase<T> {
   public:
    //! base class
    using Parent = TypedFieldBase<T>;
    //! convenience alias to the Eigen representation of this field's data
    using EigenRep_t = typename Parent::EigenRep_t;

   public:
    /**
     * constructor from a raw pointer. Typically, this would be a reference
     * to a numpy array from the python bindings.
     */
    WrappedField(const std::string & unique_name, FieldCollection & collection,
                 const Index_t & nb_components, const size_t & size, T * ptr,
                 const std::string & sub_division,
                 const Unit & unit = Unit::unitless(),
                 const Shape_t & strides = {});

    /**
     * constructor from a raw pointer. Typically, this would be a reference
     * to a numpy array from the python bindings.
     */
    WrappedField(const std::string & unique_name, FieldCollection & collection,
                 const Shape_t & components_shape, const size_t & size, T * ptr,
                 const std::string & sub_division,
                 const Unit & unit = Unit::unitless(),
                 const Shape_t & strides = {});

    /**
     * constructor from an eigen array ref.
     */
    WrappedField(const std::string & unique_name, FieldCollection & collection,
                 const Index_t & nb_components, Eigen::Ref<EigenRep_t> values,
                 const std::string & sub_division,
                 const Unit & unit = Unit::unitless(),
                 const Shape_t & strides = {});

    /**
     * constructor from an eigen array ref.
     */
    WrappedField(const std::string & unique_name, FieldCollection & collection,
                 const Shape_t & components_shape,
                 Eigen::Ref<EigenRep_t> values,
                 const std::string & sub_division,
                 const Unit & unit = Unit::unitless(),
                 const Shape_t & strides = {});

    //! Default constructor
    WrappedField() = delete;

    //! Copy constructor
    WrappedField(const WrappedField & other) = delete;

    //! Move constructor
    WrappedField(WrappedField && other) = default;

    //! Destructor
    virtual ~WrappedField() = default;

    //! Move assignment operator
    WrappedField & operator=(WrappedField && other) = delete;

    //! Copy assignment operator
    WrappedField & operator=(const Parent & other);

    /**
     * Emulation of a const constructor
     */
    static std::unique_ptr<const WrappedField> make_const(
        const std::string & unique_name, FieldCollection & collection,
        const Index_t & nb_components,
        const Eigen::Ref<const EigenRep_t> values,
        const std::string & sub_division, const Unit & unit = Unit::unitless(),
        const Shape_t & strides = {});

    void set_zero() final;
    void set_pad_size(const size_t & pad_size) final;

    size_t get_buffer_size() const final;

    Shape_t get_strides(
        const IterUnit & iter_type, Index_t element_size = 1) const final;

    StorageOrder get_storage_order() const final;

    //! give access to collections
    friend FieldCollection;

   protected:
    void resize() final;

    //! size of the wrapped buffer
    size_t size;

    /**
     * Strides of the wrapped field when iterating over sub-points, they are
     * decoupled from the underlying `FieldCollection`. If they are empty,
     * then the natural muGrid storage order applies.
     */
    Shape_t strides;
  };

  //! Alias for real-valued fields
  using RealField = TypedField<Real>;
  //! Alias for complex-valued fields
  using ComplexField = TypedField<Complex>;
  //! Alias for integer-valued fields
  using IntField = TypedField<Int>;
  //! Alias for unsigned integer-valued fields
  using UintField = TypedField<Uint>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_TYPED_HH_
