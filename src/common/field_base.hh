/**
 * file   field_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   10 Apr 2018
 *
 * @brief  Virtual base class for fields
 *
 * Copyright © 2018 Till Junge
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


#ifndef FIELD_BASE_H
#define FIELD_BASE_H

#include <string>
#include <stdexcept>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  /**
   * base class for field collection-related exceptions
   */
  class FieldCollectionError: public std::runtime_error {
  public:
    //! constructor
    explicit FieldCollectionError(const std::string& what)
      :std::runtime_error(what){}
    //! constructor
    explicit FieldCollectionError(const char * what)
      :std::runtime_error(what){}
  };

  /// base class for field-related exceptions
  class FieldError: public FieldCollectionError {
    using Parent = FieldCollectionError;
  public:
    //! constructor
    explicit FieldError(const std::string& what)
      :Parent(what){}
    //! constructor
    explicit FieldError(const char * what)
      :Parent(what){}
  };

  /**
   * Thrown when a associating a field map to and incompatible field
   * is attempted
   */
  class FieldInterpretationError: public FieldError
  {
  public:
    //! constructor
    explicit FieldInterpretationError(const std::string & what)
      :FieldError(what){}
    //! constructor
    explicit FieldInterpretationError(const char * what)
      :FieldError(what){}
  };

  namespace internal{

    /* ---------------------------------------------------------------------- */
    /**
     * Virtual base class for all fields. A field represents
     * meta-information for the per-pixel storage for a scalar, vector
     * or tensor quantity and is therefore the abstract class defining
     * the field. It is used for type and size checking at runtime and
     * for storage of polymorphic pointers to fully typed and sized
     * fields. `FieldBase` (and its children) are templated with a
     * specific `FieldCollection` (derived from
     * `muSpectre::FieldCollectionBase`). A `FieldCollection` stores
     * multiple fields that all apply to the same set of
     * pixels. Addressing and managing the data for all pixels is
     * handled by the `FieldCollection`.  Note that `FieldBase` does
     * not know anything about about mathematical operations on the
     * data or how to iterate over all pixels. Mapping the raw data
     * onto for instance Eigen maps and iterating over those is
     * handled by the `FieldMap`.
     */
    template <class FieldCollection>
    class FieldBase
    {

    protected:
      //! constructor
      //! unique name (whithin Collection)
      //! number of components
      //! collection to which this field belongs (eg, material, cell)
      FieldBase(std::string unique_name,
                size_t nb_components,
                FieldCollection & collection);

    public:
      using collection_t = FieldCollection; //!< for type checks

      //! Copy constructor
      FieldBase(const FieldBase &other) = delete;

      //! Move constructor
      FieldBase(FieldBase &&other) = delete;

      //! Destructor
      virtual ~FieldBase() = default;

      //! Copy assignment operator
      FieldBase& operator=(const FieldBase &other) = delete;

      //! Move assignment operator
      FieldBase& operator=(FieldBase &&other) = delete;

      /* ---------------------------------------------------------------------- */
      //!Identifying accessors
      //! return field name
      inline const std::string & get_name() const;
      //! return field type
      //inline const Field_t & get_type() const;
      //! return my collection (for iterating)
      inline const FieldCollection & get_collection() const;
      //! return number of components (e.g., dimensions) of this field
      inline const size_t & get_nb_components() const;
      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const = 0;

      //! number of pixels in the field
      virtual size_t size() const = 0;

      //! add a pad region to the end of the field buffer; required for
      //! using this as e.g. an FFT workspace
      virtual void set_pad_size(size_t pad_size_) = 0;

      //! pad region size
      virtual size_t get_pad_size() const {return this->pad_size;};

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      virtual void set_zero() = 0;

      //! give access to collections
      friend FieldCollection;
      //! give access to collection's base class
      using FParent_t = typename FieldCollection::Parent;
      friend FParent_t;

    protected:
      /* ---------------------------------------------------------------------- */
      //! allocate memory etc
      virtual void resize(size_t size) = 0;
      const std::string name; //!< the field's unique name
      const size_t nb_components; //!< number of components per entry
      //! reference to the collection this field belongs to
      const FieldCollection & collection;
      size_t pad_size; //!< size of padding region at end of buffer
    private:
    };


    /* ---------------------------------------------------------------------- */
    // Implementations
    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    FieldBase<FieldCollection>::FieldBase(std::string unique_name,
                                          size_t nb_components_,
                                          FieldCollection & collection_)
      :name(unique_name), nb_components(nb_components_),
    collection(collection_), pad_size{0} {}

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const std::string & FieldBase<FieldCollection>::get_name() const {
      return this->name;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const FieldCollection & FieldBase<FieldCollection>::
    get_collection() const {
      return this->collection;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const size_t & FieldBase<FieldCollection>::
    get_nb_components() const {
      return this->nb_components;
    }

  }  // internal


}  // muSpectre

#endif /* FIELD_BASE_H */
