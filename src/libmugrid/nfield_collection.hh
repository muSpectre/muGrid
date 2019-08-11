/**
 * @file   nfield_collection.hh
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

#ifndef SRC_LIBMUGRID_NFIELD_COLLECTION_HH_
#define SRC_LIBMUGRID_NFIELD_COLLECTION_HH_

#include "grid_common.hh"

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace muGrid {

  //! forward declaration of the field
  class NField;
  //! forward declaration of the state field
  class StateNField;
  //! forward declaration of the field collection
  class NFieldCollection;
  //! forward declacation of the field's destructor-functor
  template <class DefaultDestroyable>
  struct NFieldDestructor {
    void operator()(DefaultDestroyable * field);
  };

  /**
   * base class for field collection-related exceptions
   */
  class NFieldCollectionError : public std::runtime_error {
   public:
    //! constructor
    explicit NFieldCollectionError(const std::string & what)
        : std::runtime_error(what) {}
    //! constructor
    explicit NFieldCollectionError(const char * what)
        : std::runtime_error(what) {}
  };

  /* ---------------------------------------------------------------------- */
  class NFieldCollection {
   public:
    //! unique_ptr for holding fields
    using NField_ptr = std::unique_ptr<NField, NFieldDestructor<NField>>;
    //! unique_ptr for holding state fields
    using StateNField_ptr =
        std::unique_ptr<StateNField, NFieldDestructor<StateNField>>;
    enum class Domain { Global, Local };
    using iterator = typename std::vector<size_t>::const_iterator;

   protected:
    /**
     * Constructor (not called by user, who constructs either a
     * LocalNFieldCollection or a GlobalNFieldCollection
     * @param domain Domain of validity, can be global or local
     * @param spatial_dim spatial dimension of the field (can be
     *                    muGrid::Unknown, e.g., in the case of the local fields
     *                    for storing internal material variables)
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    NFieldCollection(Domain domain, Dim_t spatial_dimension, Dim_t nb_quad_pts);

   public:
    //! Default constructor
    NFieldCollection() = delete;

    //! Copy constructor
    NFieldCollection(const NFieldCollection & other) = delete;

    //! Move constructor
    NFieldCollection(NFieldCollection && other) = default;

    //! Destructor
    virtual ~NFieldCollection() = default;

    //! Copy assignment operator
    NFieldCollection & operator=(const NFieldCollection & other) = delete;

    //! Move assignment operator
    NFieldCollection & operator=(NFieldCollection && other) = default;

    //! place a new field in the responsibility of this collection (Note,
    //! because fields have protected constructors, users can't create them
    template <class NFieldType, typename... Args>
    NFieldType & register_field(const std::string & unique_name,
                                Args &&... args);

    //! place a new state field in the responsibility of this collection (Note,
    //! because state fields have protected constructors, users can't create
    //! them
    template <class StateNFieldType, typename... Args>
    StateNFieldType & register_state_field(const std::string & unique_prefix,
                                           Args &&... args);

    //! check whether a field of name 'unique_name' has already been registered
    bool field_exists(const std::string & unique_name) const;

    //! check whether a field of name 'unique_name' has already been registered
    bool state_field_exists(const std::string & unique_prefix) const;

    /**
     * returns the number of entries held by any given field in this collection.
     * This correspons nb_pixels × nb_quad_pts, (I.e., a scalar field field and
     * a vector field sharing the the same collection have the same number of
     * entries, even though the vector field has more scalar values.)
     */
    const Dim_t & size() const;

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
    void set_nb_quad(const Dim_t & nb_quad_pts_per_pixel);

    /**
     * returns the number of quadrature points
     */
    const Dim_t & get_nb_quad() const;

    const Domain & get_domain() const;

    bool is_initialised() const;

    //! iterator over indices
    iterator begin() const;
    //! iterator to end of indices
    iterator end() const;

    NField & get_field(const std::string & unique_name);
    StateNField & get_state_field(const std::string & unique_prefix);

   protected:
    /**
     * loop through all fields and allocate their memory. Is exclusively called
     * by the daughter classes' `initialise` member function.
     */
    void allocate_fields();
    //! storage container for fields
    std::map<std::string, NField_ptr> fields{};
    //! storage container for state fields
    std::map<std::string, StateNField_ptr> state_fields{};
    //! domain of validity
    const Domain domain;
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
     * absolute indices within the domain of the local processor. I.e., they are
     * universally valid to address any quadrature point on the local processor,
     * and not for any quadrature point located on anothe processor.
     */
    std::vector<size_t> indices{};
  };

  /* ---------------------------------------------------------------------- */
  template <class NFieldType, typename... Args>
  NFieldType & NFieldCollection::register_field(const std::string & unique_name,
                                                Args &&... args) {
    if (this->field_exists(unique_name)) {
      std::stringstream error{};
      error << "A NField of name '" << unique_name
            << "' is already registered in this field collection. "
            << "Currently registered fields: ";
      std::string prelude{""};
      for (const auto & name_field_pair : this->fields) {
        error << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw NFieldCollectionError(error.str());
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedNField with
    //! the number of components specified in 'int' rather than 'size_t'.
    NFieldType * raw_ptr{new NFieldType{unique_name, *this, args...}};
    NFieldType & retref{*raw_ptr};
    NField_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize(this->size());
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

  template <class StateNFieldType, typename... Args>
  StateNFieldType &
  NFieldCollection::register_state_field(const std::string & unique_prefix,
                                         Args &&... args) {
    if (this->state_field_exists(unique_prefix)) {
      std::stringstream error{};
      error << "A StateNField of name '" << unique_prefix
            << "' is already registered in this field collection. "
            << "Currently registered state fields: ";
      std::string prelude{""};
      for (const auto & name_field_pair : this->state_fields) {
        error << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw NFieldCollectionError(error.str());
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedNField with
    //! the number of components specified in 'int' rather than 'size_t'.
    StateNFieldType * raw_ptr{
        new StateNFieldType{unique_prefix, *this, args...}};
    StateNFieldType & retref{*raw_ptr};
    StateNField_ptr field{raw_ptr};
    this->state_fields[unique_prefix] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_COLLECTION_HH_
