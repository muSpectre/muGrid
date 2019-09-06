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

namespace muGrid {

  //! forward declaration
  template <typename T, Mapping Mutability>
  class NFieldMap;

  template <typename T>
  class TypedNFieldBase : public NField {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");

   protected:
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
    using Element_t = T;
    using EigenRep_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Eigen_map = Eigen::Map<EigenRep_t>;
    using Eigen_cmap = Eigen::Map<const EigenRep_t>;
    using Parent = NField;
    //! Default constructor
    TypedNFieldBase() = delete;

    //! Copy constructor
    TypedNFieldBase(const TypedNFieldBase & other) = delete;

    //! Move constructor
    TypedNFieldBase(TypedNFieldBase && other) = delete;

    //! Destructor
    virtual ~TypedNFieldBase() = default;

    //! Copy assignment operator
    TypedNFieldBase & operator=(const TypedNFieldBase & other) = delete;

    //! Move assignment operator
    TypedNFieldBase & operator=(TypedNFieldBase && other) = delete;

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

    NFieldMap<T, Mapping::Mut> get_pixel_map();
    NFieldMap<T, Mapping::Const> get_pixel_map() const;

    NFieldMap<T, Mapping::Mut> get_quad_pt_map();
    NFieldMap<T, Mapping::Const> get_quad_pt_map() const;

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

  /* ---------------------------------------------------------------------- */
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
    using Parent = TypedNFieldBase<T>;
    using EigenRep_t = typename Parent::EigenRep_t;

    //! Default constructor
    TypedNField() = delete;

    //! Copy constructor
    // TypedNField(const TypedNField & other) = delete;

    //! Move constructor
    TypedNField(TypedNField && other) = delete;

    //! Destructor
    virtual ~TypedNField() = default;

    //! Copy assignment operator
    TypedNField & operator=(const TypedNField & other) = delete;

    //! Move assignment operator
    TypedNField & operator=(TypedNField && other) = delete;

    void set_zero() final;
    void set_pad_size(size_t pad_size) final;

    static TypedNField & safe_cast(NField & other);
    static const TypedNField & safe_cast(const NField & other);

    static TypedNField & safe_cast(NField & other, const Dim_t & nb_components);
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
    T * get_ptr_to_pixel(const size_t & pixel_index);
    const T * get_ptr_to_pixel(const size_t & pixel_index) const;
    T * get_ptr_to_quad_pt(const size_t & quad_pt_index);
    const T * get_ptr_to_quad_pt(const size_t & quad_pt_index) const;

    void resize(size_t size) final;
    std::vector<T> values{};
  };

  /* ---------------------------------------------------------------------- */
  template <typename T>
  class WrappedNField : public TypedNFieldBase<T> {
   public:
    using Parent = TypedNFieldBase<T>;
    using EigenRep_t = typename Parent::EigenRep_t;

   protected:
    /**
     * constructor from an eigen array ref. Typically, this would be a reference
     * to a numpy array from the python bindings.
     */
    WrappedNField(const std::string & unique_name,
                  NFieldCollection & collection, Dim_t nb_components,
                  Eigen::Ref<EigenRep_t> values);

   public:
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

  using RealNField = TypedNField<Real>;
  using ComplexNField = TypedNField<Complex>;
  using IntNField = TypedNField<Int>;
  using UintNField = TypedNField<Uint>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_TYPED_HH_
