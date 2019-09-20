/**
 * @file   nfield_typed.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Aug 2019
 *
 * @brief  NField classes for which the scalar type has been defined
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
 * General Public License for more details.
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
 */

#ifndef SRC_LIBMUGRID_NFIELD_TYPED_HH_
#define SRC_LIBMUGRID_NFIELD_TYPED_HH_

#include "nfield.hh"
#include "grid_common.hh"

#include <Eigen/Dense>

#include <vector>
#include <memory>

namespace muGrid {

  //! forward declaration
  template <typename T, Mapping Mutability>
  class NFieldMap;
  //! forward declaration
  template <typename T>
  class TypedNFieldBase;

  template <typename T>
  class TypedNFieldBase : public NField {
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
      const TypedNFieldBase & field;
    };
    /**
     * `NField`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a `NFieldCollection. The `NField` constructor is protected to
     * ensure this. Fields are instantiated through the `register_field`
     * methods NFieldCollection.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_components number of components to store per quadrature point
     * @param collection reference to the holding field collection.
     */
    TypedNFieldBase(const std::string & unique_name,
                    NFieldCollection & collection, Dim_t nb_components)
        : Parent{unique_name, collection, nb_components} {}

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
    using Parent = NField;

    //! Default constructor
    TypedNFieldBase() = delete;

    //! Copy constructor
    TypedNFieldBase(const TypedNFieldBase & other) = delete;

    //! Move constructor
    TypedNFieldBase(TypedNFieldBase && other) = delete;

    //! Destructor
    virtual ~TypedNFieldBase() = default;

    //! Move assignment operator
    TypedNFieldBase & operator=(TypedNFieldBase && other) = delete;

    //! Copy assignment operator
    TypedNFieldBase & operator=(const TypedNFieldBase & other);

    //! Copy assignment operator
    TypedNFieldBase & operator=(const Negative & other);

    //! Copy assignment operators
    TypedNFieldBase & operator=(const EigenRep_t & other);

    //! Unary negative
    Negative operator-() const;

    //! addition assignment
    TypedNFieldBase & operator+=(const TypedNFieldBase & other);

    const std::type_info & get_stored_typeid() const final { return typeid(T); }

    //! return a vector map onto the underlying data
    Eigen_map eigen_vec();
    //! return a const vector map onto the underlying data
    Eigen_cmap eigen_vec() const;

    /**
     * return a matrix map onto the underlying data with one column per
     * quadrature point
     */
    Eigen_map eigen_quad_pt();
    /**
     * return a const matrix map onto the underlying data with one column per
     * quadrature point
     */
    Eigen_cmap eigen_quad_pt() const;

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
    friend class NFieldMap;

    /**
     * convenience function returns a map of this field, iterable per pixel.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components` ×
     * `nb_quad_pts` is used
     */
    NFieldMap<T, Mapping::Mut> get_pixel_map(const Dim_t & nb_rows = Unknown);

    /**
     * convenience function returns a const map of this field, iterable per
     * pixel.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a matrix of shape `nb_components` ×
     * `nb_quad_pts` is used
     */
    NFieldMap<T, Mapping::Const>
    get_pixel_map(const Dim_t & nb_rows = Unknown) const;

    /**
     * convenience function returns a map of this field, iterable per quadrature
     * point.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    NFieldMap<T, Mapping::Mut> get_quad_pt_map(const Dim_t & nb_rows = Unknown);

    /**
     * convenience function returns a const  map of this field, iterable per
     * quadrature point.
     *
     * @param nb_rows optional specification of the number of rows for the
     * iterate. If left to default value, a column vector is used
     */
    NFieldMap<T, Mapping::Const>
    get_quad_pt_map(const Dim_t & nb_rows = Unknown) const;

    //! get the raw data ptr. don't use unless interfacing with external libs
    T * data() const;

   protected:
    //! back-end for the public non-const eigen_XXX functions
    Eigen_map eigen_map(const Dim_t & nb_rows, const Dim_t & nb_cols);
    //! back-end for the public const eigen_XXX functions
    Eigen_cmap eigen_map(const Dim_t & nb_rows, const Dim_t & nb_cols) const;
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
   * A `muGrid::TypedNField` holds a certain number of components (scalars of
   * type `T` per quadrature point of a `muGrid::NFieldCollection`'s domain.
   *
   * @tparam T type of scalar to hold. Must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`.
   */
  template <typename T>
  class TypedNField : public TypedNFieldBase<T> {
   protected:
    /**
     * `NField`s are supposed to only exist in the form of `std::unique_ptr`s
     * held by a NFieldCollection. The `NField` constructor is protected to
     * ensure this.
     * @param unique_name unique field name (unique within a collection)
     * @param nb_components number of components to store per quadrature point
     * @param collection reference to the holding field collection.
     */
    TypedNField(const std::string & unique_name, NFieldCollection & collection,
                Dim_t nb_components)
        : Parent{unique_name, collection, nb_components} {}

   public:
    //! base class
    using Parent = TypedNFieldBase<T>;

    //! Eigen type to represent the field's data
    using EigenRep_t = typename Parent::EigenRep_t;

    //! convenience alias
    using Negative = typename Parent::Negative;

    //! Default constructor
    TypedNField() = delete;

    //! Copy constructor
    // TypedNField(const TypedNField & other) = delete;

    //! Move constructor
    TypedNField(TypedNField && other) = delete;

    //! Destructor
    virtual ~TypedNField() = default;

    //! Move assignment operator
    TypedNField & operator=(TypedNField && other) = delete;

    //! Copy assignment operator
    TypedNField & operator=(const TypedNField & other);

    //! Copy assignment operator
    TypedNField & operator=(const Negative & other);

    //! Copy assignment operator
    TypedNField & operator=(const EigenRep_t & other);

    void set_zero() final;
    void set_pad_size(size_t pad_size) final;

    //! cast a reference to a base type to this type, with full checks
    static TypedNField & safe_cast(NField & other);

    //! cast a const reference to a base type to this type, with full checks
    static const TypedNField & safe_cast(const NField & other);

    /**
     * cast a reference to a base type to this type safely, plus check whether
     * it has the right number of components
     */
    static TypedNField & safe_cast(NField & other, const Dim_t & nb_components);

    /**
     * cast a const reference to a base type to this type safely, plus check
     * whether it has the right number of components
     */
    static const TypedNField & safe_cast(const NField & other,
                                         const Dim_t & nb_components);

    size_t buffer_size() const final;

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

    //! give access to collections
    friend NFieldCollection;

   protected:
    void resize(size_t size) final;

    //! storage of the raw field data
    std::vector<T> values{};
  };

  /**
   * Wrapper class providing a field view of existing memory. This is
   * particularly useful when  dealing with input from external libraries (e.g.,
   * numpy arrays)
   */
  template <typename T>
  class WrappedNField : public TypedNFieldBase<T> {
   public:
    //! base class
    using Parent = TypedNFieldBase<T>;
    //! convenience alias to the Eigen representation of this field's data
    using EigenRep_t = typename Parent::EigenRep_t;

   public:
    /**
     * constructor from an eigen array ref. Typically, this would be a reference
     * to a numpy array from the python bindings.
     */
    WrappedNField(const std::string & unique_name,
                  NFieldCollection & collection, Dim_t nb_components,
                  Eigen::Ref<EigenRep_t> values);

    //! Default constructor
    WrappedNField() = delete;

    //! Copy constructor
    WrappedNField(const WrappedNField & other) = delete;

    //! Move constructor
    WrappedNField(WrappedNField && other) = delete;

    //! Destructor
    virtual ~WrappedNField() = default;

    //! Copy assignment operator
    WrappedNField & operator=(const WrappedNField & other) = delete;

    //! Move assignment operator
    WrappedNField & operator=(WrappedNField && other) = delete;

    //! Emulation of a const constructor
    static std::unique_ptr<const WrappedNField>
    make_const(const std::string & unique_name, NFieldCollection & collection,
               Dim_t nb_components, const Eigen::Ref<const EigenRep_t> values);

    void set_zero() final;
    void set_pad_size(size_t pad_size) final;

    size_t buffer_size() const final;

    //! give access to collections
    friend NFieldCollection;

   protected:
    /**
     * an unregistered typed field can be mapped onto an array of
     * existing values
     */
    Eigen::Ref<EigenRep_t> values{};

    void resize(size_t size) final;
  };

  //! Alias for real-valued fields
  using RealNField = TypedNField<Real>;
  //! Alias for complex-valued fields
  using ComplexNField = TypedNField<Complex>;
  //! Alias for integer-valued fields
  using IntNField = TypedNField<Int>;
  //! Alias for unsigned integer-valued fields
  using UintNField = TypedNField<Uint>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_TYPED_HH_
