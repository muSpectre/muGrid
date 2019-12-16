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

#include "grid_common.hh"

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
  class FieldCollectionError : public std::runtime_error {
   public:
    //! constructor
    explicit FieldCollectionError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit FieldCollectionError(const char * what)
        : std::runtime_error(what) {}
  };

  /**
   * Base class for both `muGrid::GlobalFieldCollection` and
   * `muGrid::LocalFieldCollection`. Manages the a group of fields with the
   * same domain of validitiy (i.e., global fields, or local fields defined on
   * the same pixels).
   */
  class FieldCollection {
   public:
    //! unique_ptr for holding fields
    using Field_ptr = std::unique_ptr<Field, FieldDestructor<Field>>;
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
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    FieldCollection(ValidityDomain domain, Dim_t spatial_dimension,
                     Dim_t nb_quad_pts);

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
     * @param nb_components number of components to be stored per quadrature
     * point (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a
     * scalar field)
     */
    template <typename T>
    TypedField<T> & register_field(const std::string & unique_name,
                                    const Dim_t & nb_components) {
      static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                    "You can only register fields templated with one of the "
                    "numeric types Real, Complex, Int, or UInt");
      return this->register_field_helper<T>(unique_name, nb_components);
    }

    /**
     * place a new real-valued field  in the responsibility of this collection
     * (Note, because fields have protected constructors, users can't create
     * them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per quadrature
     * point (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a
     * scalar field)
     */
    TypedField<Real> & register_real_field(const std::string & unique_name,
                                            const Dim_t & nb_components);
    /**
     * place a new complex-valued field  in the responsibility of this
     * collection (Note, because fields have protected constructors, users can't
     * create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per quadrature
     * point (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a
     * scalar field)
     */
    TypedField<Complex> &
    register_complex_field(const std::string & unique_name,
                           const Dim_t & nb_components);
    /**
     * place a new integer-valued field  in the responsibility of this
     * collection (Note, because fields have protected constructors, users can't
     * create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per quadrature
     * point (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a
     * scalar field)
     */
    TypedField<Int> & register_int_field(const std::string & unique_name,
                                          const Dim_t & nb_components);
    /**
     * place a new unsigned integer-valued field  in the responsibility of this
     * collection (Note, because fields have protected constructors, users can't
     * create them
     * @param unique_name unique identifier for this field
     * @param nb_components number of components to be stored per quadrature
     * point (e.g., 4 for a two-dimensional second-rank tensor, or 1 for a
     * scalar field)
     */
    TypedField<Uint> & register_uint_field(const std::string & unique_name,
                                            const Dim_t & nb_components);

    /**
     * place a new state field in the responsibility of this collection (Note,
     * because state fields have protected constructors, users can't create them
     */
    template <typename T>
    TypedStateField<T> &
    register_state_field(const std::string & unique_prefix,
                         const Dim_t & nb_memory, const Dim_t & nb_components) {
      static_assert(
          std::is_scalar<T>::value or std::is_same<T, Complex>::value,
          "You can only register state fields templated with one of the "
          "numeric types Real, Complex, Int, or UInt");
      return this->register_state_field_helper<T>(unique_prefix, nb_memory,
                                                  nb_components);
    }

    /**
     * place a new real-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per quadrature
     * point
     */
    TypedStateField<Real> &
    register_real_state_field(const std::string & unique_prefix,
                              const Dim_t & nb_memory,
                              const Dim_t & nb_components);

    /**
     * place a new complex-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per quadrature
     * point
     */
    TypedStateField<Complex> &
    register_complex_state_field(const std::string & unique_prefix,
                                 const Dim_t & nb_memory,
                                 const Dim_t & nb_components);

    /**
     * place a new integer-valued state field in the responsibility of this
     * collection (Note, because state fields have protected constructors, users
     * can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per quadrature
     * point
     */
    TypedStateField<Int> &
    register_int_state_field(const std::string & unique_prefix,
                             const Dim_t & nb_memory,
                             const Dim_t & nb_components);

    /**
     * place a new unsigned integer-valued state field in the responsibility of
     * this collection (Note, because state fields have protected constructors,
     * users can't create them
     *
     * @param unique_prefix unique idendifier for this state field
     * @param nb_memory number of previous values of this field to store
     * @param nb_components number of scalar components to store per quadrature
     * point
     */
    TypedStateField<Uint> &
    register_uint_state_field(const std::string & unique_prefix,
                              const Dim_t & nb_memory,
                              const Dim_t & nb_components);

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool field_exists(const std::string & unique_name) const;

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool state_field_exists(const std::string & unique_prefix) const;

    /**
     * returns the number of entries held by any given field in this
     * collection. This corresponds to nb_pixels × nb_quad_pts, (I.e., a scalar
     * field field and a vector field sharing the the same collection have the
     * same number of entries, even though the vector field has more scalar
     * values.)
     */
    const Dim_t & get_nb_entries() const;

    //! returns the number of pixels present in the collection
    size_t get_nb_pixels() const;

    /**
     * check whether the number of quadrature points per pixel/voxel has ben
     * set
     */
    bool has_nb_quad() const;

    /**
     * set the number of quadrature points per pixel/voxel. Can only be done
     * once.
     */
    void set_nb_quad(Dim_t nb_quad_pts_per_pixel);

    /**
     * return the number of quadrature points per pixel
     */
    const Dim_t & get_nb_quad() const;

    /**
     * return the spatial dimension of the underlying discretisation grid
     */
    const Dim_t & get_spatial_dim() const;

    /**
     * return the domain of validity (i.e., wher the fields are defined globally
     * (`muGrid::FieldCollection::ValidityDomain::Global`) or locally
     * (`muGrid::FieldCollection::ValidityDomain::Local`)
     */
    const ValidityDomain & get_domain() const;

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
    IndexIterable get_quad_pt_indices() const;

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

   protected:
    //! internal worker function called by register_<T>_field
    template <typename T>
    TypedField<T> & register_field_helper(const std::string & unique_name,
                                           const Dim_t & nb_components);

    //! internal worker function called by register_<T>_state_field
    template <typename T>
    TypedStateField<T> &
    register_state_field_helper(const std::string & unique_prefix,
                                const Dim_t & nb_memory,
                                const Dim_t & nb_components);

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
    Dim_t spatial_dim;
    //! number of quadrature points per pixel/voxel
    Dim_t nb_quad_pts;
    //! total number of entries
    Dim_t nb_entries{Unknown};
    //! keeps track of whether the collection has already been initialised
    bool initialised{false};
    /**
     * Storage for indices of the stored quadrature points in the global field
     * collection. Note that these are not truly global indices, but rather
     * absolute indices within the domain of the local processor. I.e., they
     * are universally valid to address any quadrature point on the local
     * processor, and not for any quadrature point located on anothe
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
     * evaluate and return the stride with with the fast index of the iterators
     * over the indices of this collection rotate
     */
    Dim_t get_stride() const {
      return (this->iteration_type == Iteration::QuadPt)
                 ? this->collection.get_nb_quad()
                 : 1;
    }

    /**
     * allow the field collection to create
     * `muGrid::FieldCollection::IndexIterable`s
     */
    friend FieldCollection;
    //! Constructor is protected, because no one ever need to construct this
    //! except the fieldcollection
    IndexIterable(const FieldCollection & collection,
                  const Iteration & iteration_type);

    //! reference back to the proxied collection
    const FieldCollection & collection;

    //! whether to iterate over pixels or quadrature points
    const Iteration iteration_type;
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

    //! Copy assignment operator
    iterator & operator=(const iterator & other) = default;

    //! Move assignment operator
    iterator & operator=(iterator && other) = default;

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
    size_t stride;
    //! fast-moving index
    size_t offset{};
    //! iterator of slow moving index
    PixelIndexIterator_t pixel_index_iterator;
  };

  /* ---------------------------------------------------------------------- */

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_COLLECTION_HH_
