/**
 * @file   field_collection.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   10 Aug 2019
 *
 * @brief  Base class for field collections
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

#ifndef SRC_LIBMUGRID_FIELD_COLLECTION_HH_
#define SRC_LIBMUGRID_FIELD_COLLECTION_HH_

#include "exception.hh"
#include "grid_common.hh"
#include "units.hh"

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace muGrid {

  //! forward declaration of the `muSpectre::Field`
  class Field;

  //! forward declaration of the `muSpectre::TypedField`
  template <typename T>
  class TypedField;

  //! forward declaration of the state field
  class StateField;

  //! forward declaration of the state field
  template <typename T>
  class TypedStateField;

  //! forward declaration of the field collection
  class FieldCollection;

  //! forward declacation of the field's destructor-functor
  template <class DefaultDestroyable>
  struct FieldDestructor {
    //! deletes the held field
    void operator()(DefaultDestroyable * field);
  };

  /**
   * base class for field collection-related exceptions
   */
  class FieldCollectionError : public RuntimeError {
   public:
    //! constructor
    explicit FieldCollectionError(const std::string & what)
        : RuntimeError(what) {}
    //! constructor
    explicit FieldCollectionError(const char * what) : RuntimeError(what) {}
  };

  /**
   * Base class for both `muGrid::GlobalFieldCollection` and
   * `muGrid::LocalFieldCollection`. Manages the a group of fields with the
   * same domain of validity (i.e., global fields, or local fields defined on
   * the same pixels).
   */
  class FieldCollection {
   public:
    //! unique_ptr for holding fields
    using Field_ptr = std::unique_ptr<Field, FieldDestructor<Field>>;
    //! map to hold nb_sub_pts by tag
    using SubPtMap_t = std::map<std::string, Index_t>;
    //! unique_ptr for holding state fields
    using StateField_ptr =
        std::unique_ptr<StateField, FieldDestructor<StateField>>;
    //! domain of validity of the managed fields
    enum class ValidityDomain { Global, Local };

    class IndexIterable;
    //! convenience alias
    using QuadPtIndexIterable = IndexIterable;
    class PixelIndexIterable;

   protected:
    /**
     * Constructor (not called by user, who constructs either a
     * LocalFieldCollection or a GlobalFieldCollection
     * @param domain Domain of validity, can be global or local
     * @param spatial_dimension spatial dimension of the field (can be
     *                    muGrid::Unknown, e.g., in the case of the local fields
     *                    for storing internal material variables)
     * @param nb_sub_pts Specification of pixel subdivision. This is a map that
     *                    of a string (the name of the subdivision scheme) to
     *                    the number of subdivisions
     * @param storage_order Storage order of the pixels vs subdivision portion
     *                    of the field. In a column-major storage order, the
     *                    pixel subdivision (i.e. the components of the field)
     *                    are stored next to each other in memory, file in a
     *                    row-major storage order for each component the
     *                    pixels are stored next to each other in memory.
     *                    (This is also sometimes called the array of structures
     *                    vs. structure of arrays storage order.)
     *                    Important: The pixels or subpoints have their own
     *                    storage order that is not affected by this setting.
     */
    FieldCollection(ValidityDomain domain, const Index_t & spatial_dimension,
                    const SubPtMap_t & nb_sub_pts,
                    StorageOrder storage_order =
                        StorageOrder::ArrayOfStructures);

   public:
    //! Default constructor
    FieldCollection() = delete;

    //! Copy constructor
    FieldCollection(const FieldCollection & other) = delete;

    //! Move constructor
    FieldCollection(FieldCollection && other) = default;

    //! Destructor
    virtual ~FieldCollection() = default;

    //! Copy assignment operator
    FieldCollection & operator=(const FieldCollection & other) = delete;

    //! Move assignment operator
    FieldCollection & operator=(FieldCollection && other) = default;

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per sub-point
     * (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a scalar
     * field)
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     */
    template <typename T>
    TypedField<T> &
    register_field(const std::string & unique_name,
                   const Index_t & nb_components,
                   const std::string & sub_division_tag = PixelTag,
                   const Unit & unit = Unit::unitless()) {
      static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                    "You can only register fields templated with one of the "
                    "numeric types Real, Complex, Int, or UInt");
      return this->register_field_helper<T>(unique_name, nb_components,
                                            sub_division_tag, unit);
    }

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param components_shape number of components to store per quadrature
     * point
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     * @param storage_oder in-memory storage order of the components
     */
    template <typename T>
    TypedField<T> &
    register_field(const std::string & unique_name,
                   const Shape_t & components_shape,
                   const std::string & sub_division_tag = PixelTag,
                   const Unit & unit = Unit::unitless()) {
      static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                    "You can only register fields templated with one of the "
                    "numeric types Real, Complex, Int, or UInt");
      return this->register_field_helper<T>(
          unique_name, components_shape, sub_division_tag, unit);
    }

    /**
     * place a new real-valued field  in the responsibility of this collection
     * (Note, because fields have protected constructors, users can't create
     * them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per sub-point
     * (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a scalar
     * field)
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     */
    TypedField<Real> &
    register_real_field(const std::string & unique_name,
                        const Index_t & nb_components,
                        const std::string & sub_division_tag = PixelTag,
                        const Unit & unit = Unit::unitless());

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param components_shape number of components to store per quadrature
     * point
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     * @param storage_oder in-memory storage order of the components
     */
    TypedField<Real> &
    register_real_field(const std::string & unique_name,
                        const Shape_t & components_shape,
                        const std::string & sub_division_tag = PixelTag,
                        const Unit & unit = Unit::unitless());

      /**
       * place a new complex-valued field  in the responsibility of this
       * collection (Note, because fields have protected constructors, users can't
       * create them
       * @param unique_name unique identifier for this field
       * @param nb_components number of components to be stored per sub-point
       * (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a scalar
       * field)
       * @param sub_division_tag unique identifier of the subdivision scheme
       * @param unit phyiscal unit of this field
       */
    TypedField<Complex> &
    register_complex_field(const std::string & unique_name,
                           const Index_t & nb_components,
                           const std::string & sub_division_tag = PixelTag,
                           const Unit & unit = Unit::unitless());

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param components_shape number of components to store per quadrature
     * point
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     * @param storage_oder in-memory storage order of the components
     */
    TypedField<Complex> &
    register_complex_field(const std::string & unique_name,
                           const Shape_t & components_shape,
                           const std::string & sub_division_tag = PixelTag,
                           const Unit & unit = Unit::unitless());

    /**
     * place a new integer-valued field  in the responsibility of this
     * collection (Note, because fields have protected constructors, users can't
     * create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per sub-point
     * (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a scalar
     * field)
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     */
    TypedField<Int> &
    register_int_field(const std::string & unique_name,
                       const Index_t & nb_components,
                       const std::string & sub_division_tag = PixelTag,
                       const Unit & unit = Unit::unitless());

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param components_shape number of components to store per quadrature
     * point
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     * @param storage_oder in-memory storage order of the components
     */
    TypedField<Int> &
    register_int_field(const std::string & unique_name,
                       const Shape_t & components_shape,
                       const std::string & sub_division_tag = PixelTag,
                       const Unit & unit = Unit::unitless());

    /**
     * place a new unsigned integer-valued field  in the responsibility of this
     * collection (Note, because fields have protected constructors, users can't
     * create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per sub-point
     * (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a scalar
     * field)
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     */
    TypedField<Uint> &
    register_uint_field(const std::string & unique_name,
                        const Index_t & nb_components,
                        const std::string & sub_division_tag = PixelTag,
                        const Unit & unit = Unit::unitless());

    /**
     * place a new field in the responsibility of this collection (Note, because
     * fields have protected constructors, users can't create them
     * @param unique_name unique identifier for this field
     * @param components_shape number of components to store per quadrature
     * point
     * @param sub_division_tag unique identifier of the subdivision scheme
     * @param unit phyiscal unit of this field
     * @param storage_oder in-memory storage order of the components
     */
    TypedField<Uint> &
    register_uint_field(const std::string & unique_name,
                        const Shape_t & components_shape,
                        const std::string & sub_division_tag = PixelTag,
                        const Unit & unit = Unit::unitless());

    /**
     * place a new state field in the responsibility of this collection (Note,
     * because state fields have protected constructors, users can't create them
     */
    template <typename T>
    TypedStateField<T> &
    register_state_field(const std::string & unique_prefix,
                         const Index_t & nb_memory,
                         const Index_t & nb_components,
                         const std::string & sub_division_tag = PixelTag,
                         const Unit & unit = Unit::unitless()) {
      static_assert(
          std::is_scalar<T>::value or std::is_same<T, Complex>::value,
          "You can only register state fields templated with one of the "
          "numeric types Real, Complex, Int, or UInt");
      return this->register_state_field_helper<T>(
          unique_prefix, nb_memory, nb_components, sub_division_tag, unit);
    }

    /**
     * place a new real-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per
     * quadrature point
     */
    TypedStateField<Real> &
    register_real_state_field(const std::string & unique_prefix,
                              const Index_t & nb_memory,
                              const Index_t & nb_components,
                              const std::string & sub_division_tag = PixelTag,
                              const Unit & unit = Unit::unitless());

    /**
     * place a new complex-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per
     * quadrature point
     */
    TypedStateField<Complex> & register_complex_state_field(
        const std::string & unique_prefix, const Index_t & nb_memory,
        const Index_t & nb_components,
        const std::string & sub_division_tag = PixelTag,
        const Unit & unit = Unit::unitless());

    /**
     * place a new integer-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per
     * quadrature point
     */
    TypedStateField<Int> &
    register_int_state_field(const std::string & unique_prefix,
                             const Index_t & nb_memory,
                             const Index_t & nb_components,
                             const std::string & sub_division_tag = PixelTag,
                             const Unit & unit = Unit::unitless());

    /**
     * place a new unsigned integer-valued state field in the responsibility of
     * this collection (Note, because state fields have protected constructors,
     * users can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per
     * quadrature point
     */
    TypedStateField<Uint> &
    register_uint_state_field(const std::string & unique_prefix,
                              const Index_t & nb_memory,
                              const Index_t & nb_components,
                              const std::string & sub_division_tag = PixelTag,
                              const Unit & unit = Unit::unitless());

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool field_exists(const std::string & unique_name) const;

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool state_field_exists(const std::string & unique_prefix) const;

    //! returns the number of pixels present in the collection
    Index_t get_nb_pixels() const;

    /**
     * returns the number of (virtual) pixels required to store the underlying
     * data that may involve padding regions
     */
    Index_t get_nb_buffer_pixels() const;

    /**
     * Check whether the number of subdivision points peir pixel/voxel has been
     * set for a given tags
     */
    bool has_nb_sub_pts(const std::string & tag) const;

    /**
     * set the number of sub points per pixel/voxel for a given tag. Can only be
     * done once per tag
     */

    void set_nb_sub_pts(const std::string & tag,
                        const Index_t & nb_sub_pts_per_pixel);

    /**
     * return the number of subpoints per pixel/voxel for a given tag
     */
    const Index_t & get_nb_sub_pts(const std::string & tag);
    /**
     * return the number of subpoints per pixel/voxel for a given tag
     */
    const Index_t & get_nb_sub_pts(const std::string & tag) const;

    /**
     * return the spatial dimension of the underlying discretisation grid
     */
    const Index_t & get_spatial_dim() const;

    /**
     * return the domain of validity (i.e., wher the fields are defined globally
     * (`muGrid::FieldCollection::ValidityDomain::Global`) or locally
     * (`muGrid::FieldCollection::ValidityDomain::Local`)
     */
    const ValidityDomain & get_domain() const;

    //! return shape of the pixels
    virtual Shape_t get_pixels_shape() const = 0;

    //! return strides of the pixels
    virtual Shape_t get_pixels_strides(Index_t element_size = 1) const = 0;

    //! return the storage order of the pixels vs. subpoints
    const StorageOrder & get_storage_order() const;

    //! check whether two field collections have the same memory layout
    bool has_same_memory_layout(const FieldCollection & other) const;

    /**
     * whether the collection has been properly initialised (i.e., it knows the
     * number of quadrature points and all its pixels/voxels
     */
    bool is_initialised() const;

    /**
     * return an iterable proxy to the collection which allows to efficiently
     * iterate over the indices fo the collection's pixels
     */
    PixelIndexIterable get_pixel_indices_fast() const;

    /**
     * return an iterable proxy to the collection which allows to iterate over
     * the indices fo the collection's pixels
     */
    IndexIterable get_pixel_indices() const;

    /**
     * return an iterable proxy to the collection which allows to iterate over
     * the indices fo the collection's quadrature points
     */
    IndexIterable get_sub_pt_indices(const std::string & tag) const;

    std::vector<size_t> get_pixel_ids() { return this->pixel_indices; }

    /**
     * returns a (base-type) reference to the field identified by `unique_name`.
     * Throws a `muGrid::FieldCollectionError` if the field does not exist.
     */
    Field & get_field(const std::string & unique_name);

    /**
     * returns a (base-type) reference to the state field identified by
     * `unique_prefix`. Throws a `muGrid::FieldCollectionError` if the state
     * field does not exist.
     */
    StateField & get_state_field(const std::string & unique_prefix);

    //! returns a vector of all field names
    std::vector<std::string> list_fields() const;

    //! preregister a map for latent initialisation
    void preregister_map(std::shared_ptr<std::function<void()>> & call_back);

    /**
     * run-time checker for nb_sub_pts: checks whether the number of sub-points
     * (e.g., quadrature points) is compatible with the sub-division scheme).
     * Attention: this does allow `Unknown`  as valid values for
     * `IterUnit::SubPt`, if the tag is defined, since these values can
     * be specified for the entire FieldCollection at a later point, before
     * initialisation. Hence, this function cannot be used for checking
     * nb_sub_pts for iterators, which need a known value. Use
     * `check_initialised_nb_sub_pts()` instead for that.
     */
    Index_t check_nb_sub_pts(const Index_t & nb_sub_pts,
                             const IterUnit & iteration_type,
                             const std::string & tag) const;

    /**
     * run-time checker for nb_sub_pts: checks whether the number of sub-points
     * (e.g., quadrature points) is compatible with the sub-division scheme),
     * and set to a positive integer value (i.e., not `Unknown`).
     */
    size_t check_initialised_nb_sub_pts(const Index_t & nb_sub_pts,
                                        const IterUnit & iteration_type,
                                        const std::string & tag) const;

   protected:
    //! internal worker function called by register_<T>_field
    template <typename T>
    TypedField<T> & register_field_helper(const std::string & unique_name,
                                          const Index_t & nb_components,
                                          const std::string & sub_division_tag,
                                          const Unit & unit);

    //! internal worker function called by register_<T>_field
    template <typename T>
    TypedField<T> & register_field_helper(const std::string & unique_name,
                                          const Shape_t & components_shape,
                                          const std::string & sub_division_tag,
                                          const Unit & unit);

    //! internal worker function called by register_<T>_state_field
    template <typename T>
    TypedStateField<T> & register_state_field_helper(
        const std::string & unique_prefix, const Index_t & nb_memory,
        const Index_t & nb_components, const std::string & sub_division_tag,
        const Unit & unit);

    /**
     * loop through all fields and allocate their memory. Is exclusively
     * called by the daughter classes' `initialise` member function.
     */
    void allocate_fields();

    /**
     * initialise all preregistered maps
     */
    void initialise_maps();

    //! storage container for fields
    std::map<std::string, Field_ptr> fields{};
    //! storage container for state fields
    std::map<std::string, StateField_ptr> state_fields{};

    //! Maps registered before initialisation which will need their data_ptr set
    std::vector<std::weak_ptr<std::function<void()>>> init_callbacks{};
    //! domain of validity
    ValidityDomain domain;
    //! spatial dimension
    Index_t spatial_dim;

    //! number of subpoints per pixel/voxel, stored by tag
    SubPtMap_t nb_sub_pts;

    //! total number of pixels
    Index_t nb_pixels{Unknown};

    //! total number of pixels for the buffer (including padding regions)
    Index_t nb_buffer_pixels{Unknown};

    //! storage oder
    StorageOrder storage_order;

    //! keeps track of whether the collection has already been initialised
    bool initialised{false};
    /**
     * Storage for indices of the stored quadrature points in the global field
     * collection. Note that these are not truly global indices, but rather
     * absolute indices within the domain of the local processor. I.e., they
     * are universally valid to address any quadrature point on the local
     * processor, and not for any quadrature point located on another
     * processor.
     */
    std::vector<size_t> pixel_indices{};
  };

  /**
   * Lightweight proxy class providing iteration over the pixel indices of a
   * `muGrid::FieldCollection`
   */
  class FieldCollection::PixelIndexIterable {
   public:
    //! stl
    using iterator = typename std::vector<size_t>::const_iterator;
    //! Default constructor
    PixelIndexIterable() = delete;

    //! Copy constructor
    PixelIndexIterable(const PixelIndexIterable & other) = delete;

    //! Move constructor
    PixelIndexIterable(PixelIndexIterable && other) = default;

    //! Destructor
    virtual ~PixelIndexIterable() = default;

    //! Copy assignment operator
    PixelIndexIterable & operator=(const PixelIndexIterable & other) = delete;

    //! Move assignment operator
    PixelIndexIterable & operator=(PixelIndexIterable && other) = delete;

    //! stl
    iterator begin() const;

    //! stl
    iterator end() const;

    //! stl
    size_t size() const;

   protected:
    //! allow field collections to call the procted constructor of this iterable
    friend FieldCollection;

    //! Constructor is protected, because no one ever need to construct this
    //! except the fieldcollection
    explicit PixelIndexIterable(const FieldCollection & collection);

    //! reference back to the proxied collection
    const FieldCollection & collection;
  };

  /**
   * Iterate class for iterating over quadrature point indices of a field
   * collection (i.e. the iterate you get when iterating over the result of
   * `muGrid::FieldCollection::get_quad_pt_indices`).
   */
  class FieldCollection::IndexIterable {
   public:
    class iterator;
    //! Default constructor
    IndexIterable() = delete;

    //! Copy constructor
    IndexIterable(const IndexIterable & other) = delete;

    //! Move constructor
    IndexIterable(IndexIterable && other) = default;

    //! Destructor
    virtual ~IndexIterable() = default;

    //! Copy assignment operator
    IndexIterable & operator=(const IndexIterable & other) = delete;

    //! Move assignment operator
    IndexIterable & operator=(IndexIterable && other) = delete;

    //! stl
    iterator begin() const;

    //! stl
    iterator end() const;

    //! stl
    size_t size() const;

   protected:
    /**
     * allow the field collection to create
     * `muGrid::FieldCollection::IndexIterable`s
     */
    friend FieldCollection;

    /**
     * Constructor is protected, because no one ever need to construct this
     * except the fieldcollection. Constructor for sub_point iteration
     */
    IndexIterable(const FieldCollection & collection, const std::string & tag,
                  const Index_t & stride = Unknown);

    /**
     * Constructor is protected, because no one ever need to construct this
     * except the fieldcollection. Constructor for pixel iteration
     */
    explicit IndexIterable(const FieldCollection & collection,
                           const Index_t & stride = Unknown);

    //! reference back to the proxied collection
    const FieldCollection & collection;

    //! whether to iterate over pixels or quadrature points
    const IterUnit iteration_type;

    //! stride for the slow moving index
    size_t stride;
  };

  /**
   * iterator class for iterating over quadrature point indices or pixel indices
   * of a `muGrid::FieldCollection::IndexIterable`. Dereferences to an index.
   */
  class FieldCollection::IndexIterable::iterator final {
   public:
    //! convenience alias
    using PixelIndexIterator_t = typename std::vector<size_t>::const_iterator;
    //! Default constructor
    iterator() = delete;

    //! constructor
    iterator(const PixelIndexIterator_t & pixel_index_iterator,
             const size_t & stride);

    //! Copy constructor
    iterator(const iterator & other) = default;

    //! Move constructor
    iterator(iterator && other) = default;

    //! Destructor
    ~iterator() = default;

    //! pre-increment
    iterator & operator++() {
      // increment the offset and keep only the modulo
      (++this->offset) %= this->stride;
      // conditionally increment the pixel if the offset has recycled to zero
      this->pixel_index_iterator += size_t(this->offset == 0);
      return *this;
    }

    //! comparison
    bool operator!=(const iterator & other) const {
      return (this->pixel_index_iterator != other.pixel_index_iterator) or
             (this->offset != other.offset);
    }

    //! comparison (required by akantu::iterators)
    bool operator==(const iterator & other) const {
      return not(*this != other);
    }

    //! dereference
    size_t operator*() {
      return *(this->pixel_index_iterator) * this->stride + this->offset;
    }

   protected:
    //! stride for the slow moving index
    size_t stride;  // store the value rather than the const ref in order
                    //  to allow the IndexIterable to be destroyed and still use
                    //  the iterator
    //! fast-moving index
    size_t offset{};
    //! iterator of slow moving index
    PixelIndexIterator_t pixel_index_iterator;
  };

  /* ---------------------------------------------------------------------- */

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_COLLECTION_HH_
