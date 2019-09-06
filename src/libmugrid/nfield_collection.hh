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
  template <typename T>
  class TypedNField;
  //! forward declaration of the state field
  class StateNField;
  //! forward declaration of the state field
  template <typename T>
  class TypedStateNField;
  //! forward declaration of the wrapped field
  template <typename T>
  class WrappedNField;
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
    template <typename T>
    TypedNField<T> & register_field(const std::string & unique_name,
                                    Dim_t nb_components) {
      static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                    "You can only register fields templated with one of the "
                    "numeric types Real, Complex, Int, or UInt");
      return this->register_field_helper<T>(unique_name, nb_components);
    }

    TypedNField<Real> & register_real_field(const std::string & unique_name,
                                            Dim_t nb_components);
    TypedNField<Complex> &
    register_complex_field(const std::string & unique_name,
                           Dim_t nb_components);
    TypedNField<Int> & register_int_field(const std::string & unique_name,
                                          Dim_t nb_components);
    TypedNField<Uint> & register_uint_field(const std::string & unique_name,
                                            Dim_t nb_components);

    //! place a new state field in the responsibility of this collection
    //! (Note, because state fields have protected constructors, users can't
    //! create them
    template <typename T>
    TypedStateNField<T> &
    register_state_field(const std::string & unique_prefix,
                         Dim_t nb_memory, Dim_t nb_components) {
      static_assert(
          std::is_scalar<T>::value or std::is_same<T, Complex>::value,
          "You can only register state fields templated with one of the "
          "numeric types Real, Complex, Int, or UInt");
      return this->register_state_field_helper<T>(unique_prefix, nb_memory,
                                                  nb_components);
    }

    TypedStateNField<Real> &
    register_real_state_field(const std::string & unique_name,
                              Dim_t nb_memory,
                              Dim_t nb_components);
    TypedStateNField<Complex> &
    register_complex_state_field(const std::string & unique_name,
                                 Dim_t nb_memory,
                                 Dim_t nb_components);
    TypedStateNField<Int> &
    register_int_state_field(const std::string & unique_name,
                             Dim_t nb_memory,
                             Dim_t nb_components);
    TypedStateNField<Uint> &
    register_uint_state_field(const std::string & unique_name,
                              Dim_t nb_memory,
                              Dim_t nb_components);

    //! place a new field in the responsibility of this collection (Note,
    //! because fields have protected constructors, users can't create them
    template <typename T>
    WrappedNField<T> & register_wrapped_field(
        const std::string & unique_name, Dim_t nb_components,
        Eigen::Ref<typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        values) {
      static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                    "You can only register wrapped fields templated with one "
                    "of the numeric types Real, Complex, Int, or UInt");
      return this->register_wrapped_field_helper<T>(unique_name, nb_components, values);
    }

    WrappedNField<Real> &
    register_real_wrapped_field(
        const std::string & unique_name, Dim_t nb_components,
        Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> values);
    WrappedNField<Complex> &
    register_complex_wrapped_field(
        const std::string & unique_name, Dim_t nb_components,
        Eigen::Ref<Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>> values);
    WrappedNField<Int> &
    register_int_wrapped_field(
        const std::string & unique_name, Dim_t nb_components,
        Eigen::Ref<Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>> values);
    WrappedNField<Uint> &
    register_uint_wrapped_field(
        const std::string & unique_name, Dim_t nb_components,
        Eigen::Ref<Eigen::Matrix<Uint, Eigen::Dynamic, Eigen::Dynamic>> values);

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool field_exists(const std::string & unique_name) const;

    //! check whether a field of name 'unique_name' has already been
    //! registered
    bool state_field_exists(const std::string & unique_prefix) const;

    /**
     * returns the number of entries held by any given field in this
     * collection. This correspons nb_pixels × nb_quad_pts, (I.e., a scalar
     * field field and a vector field sharing the the same collection have the
     * same number of entries, even though the vector field has more scalar
     * values.)
     */
    Dim_t size() const;

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
     * returns the number of quadrature points
     */
    Dim_t get_nb_quad() const;

    const Domain & get_domain() const;

    bool is_initialised() const;

    //! iterator over indices
    iterator begin() const;
    //! iterator to end of indices
    iterator end() const;

    NField & get_field(const std::string & unique_name);
    StateNField & get_state_field(const std::string & unique_prefix);

    const std::map<std::string, NField_ptr> & get_fields() const;

    std::vector<std::string> list_fields() const;

   protected:
    template <typename T>
    TypedNField<T> & register_field_helper(const std::string & unique_name,
                                           Dim_t nb_components);
    template <typename T>
    TypedStateNField<T> &
    register_state_field_helper(const std::string & unique_prefix,
                                Dim_t nb_memory, Dim_t nb_components);
    template <typename T>
    WrappedNField<T> &
    register_wrapped_field_helper(
        const std::string & unique_prefix, Dim_t nb_components,
        Eigen::Ref<typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        values);

    /**
     * loop through all fields and allocate their memory. Is exclusively
     * called by the daughter classes' `initialise` member function.
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
     * absolute indices within the domain of the local processor. I.e., they
     * are universally valid to address any quadrature point on the local
     * processor, and not for any quadrature point located on anothe
     * processor.
     */
    std::vector<size_t> indices{};
  };

  /* ---------------------------------------------------------------------- */

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_COLLECTION_HH_
