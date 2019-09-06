/**
 * @file   nfield_collection.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Implementations for field collections
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

#include "nfield_collection.hh"
#include "nfield.hh"
#include "state_nfield.hh"
#include "nfield_typed.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <class DefaultDestroyable>
  void NFieldDestructor<DefaultDestroyable>::
  operator()(DefaultDestroyable * field) {
    delete field;
  }

  /* ---------------------------------------------------------------------- */
  template struct NFieldDestructor<NField>;
  template struct NFieldDestructor<StateNField>;

  /* ---------------------------------------------------------------------- */
  NFieldCollection::NFieldCollection(Domain domain, Dim_t spatial_dimension,
                                     Dim_t nb_quad_pts)
      : domain{domain}, spatial_dim{spatial_dimension}, nb_quad_pts{
                                                            nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> &
  NFieldCollection::register_field_helper(const std::string & unique_name,
                                          Dim_t nb_components) {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");
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
    TypedNField<T> * raw_ptr{
        new TypedNField<T>{unique_name, *this, nb_components}};
    TypedNField<T> & retref{*raw_ptr};
    NField_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize(this->size());
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  TypedNField<Real> &
  NFieldCollection::register_real_field(const std::string & unique_name,
                                        Dim_t nb_components) {
    return this->register_field_helper<Real>(unique_name, nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedNField<Complex> &
  NFieldCollection::register_complex_field(const std::string & unique_name,
                                           Dim_t nb_components) {
    return this->register_field_helper<Complex>(unique_name, nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedNField<Int> &
  NFieldCollection::register_int_field(const std::string & unique_name,
                                       Dim_t nb_components) {
    return this->register_field_helper<Int>(unique_name, nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedNField<Uint> &
  NFieldCollection::register_uint_field(const std::string & unique_name,
                                        Dim_t nb_components) {
    return this->register_field_helper<Uint>(unique_name, nb_components);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedStateNField<T> & NFieldCollection::register_state_field_helper(
      const std::string & unique_prefix, Dim_t nb_memory, Dim_t nb_components) {
    static_assert(
        std::is_scalar<T>::value or std::is_same<T, Complex>::value,
        "You can only register state fields templated with one of the "
        "numeric types Real, Complex, Int, or UInt");
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
    //! following line, please check whether you are creating a TypedNField
    //! with the number of components specified in 'int' rather than 'size_t'.
    TypedStateNField<T> * raw_ptr{new TypedStateNField<T>{
        unique_prefix, *this, nb_memory, nb_components}};
    TypedStateNField<T> & retref{*raw_ptr};
    StateNField_ptr field{raw_ptr};
    this->state_fields[unique_prefix] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  TypedStateNField<Real> &
  NFieldCollection::register_real_state_field(const std::string & unique_name,
                                              Dim_t nb_memory,
                                              Dim_t nb_components) {
    return this->register_state_field_helper<Real>(unique_name, nb_memory,
                                                   nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateNField<Complex> & NFieldCollection::register_complex_state_field(
      const std::string & unique_name, Dim_t nb_memory,
      Dim_t nb_components) {
    return this->register_state_field_helper<Complex>(unique_name, nb_memory,
                                                      nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateNField<Int> &
  NFieldCollection::register_int_state_field(const std::string & unique_name,
                                             Dim_t nb_memory,
                                             Dim_t nb_components) {
    return this->register_state_field_helper<Int>(unique_name, nb_memory,
                                                  nb_components);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateNField<Uint> &
  NFieldCollection::register_uint_state_field(const std::string & unique_name,
                                              Dim_t nb_memory,
                                              Dim_t nb_components) {
    return this->register_state_field_helper<Uint>(unique_name, nb_memory,
                                                   nb_components);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  WrappedNField<T> &
  NFieldCollection::register_wrapped_field_helper(
      const std::string & unique_name, Dim_t nb_components,
      Eigen::Ref<typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      values) {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");
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
    WrappedNField<T> * raw_ptr{
        new WrappedNField<T>{unique_name, *this, nb_components, values}};
    WrappedNField<T> & retref{*raw_ptr};
    NField_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize(this->size());
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  WrappedNField<Real> &
  NFieldCollection::register_real_wrapped_field(
      const std::string & unique_name, Dim_t nb_components,
      Eigen::Ref<typename Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
      values) {
    return this->register_wrapped_field_helper<Real>(unique_name, nb_components,
                                                     values);
  }

  /* ---------------------------------------------------------------------- */
  WrappedNField<Complex> &
  NFieldCollection::register_complex_wrapped_field(
      const std::string & unique_name, Dim_t nb_components,
      Eigen::Ref<typename Eigen::Matrix<Complex, Eigen::Dynamic,
                 Eigen::Dynamic>> values) {
    return this->register_wrapped_field_helper<Complex>(unique_name, nb_components,
                                                     values);
  }

  /* ---------------------------------------------------------------------- */
  WrappedNField<Int> &
  NFieldCollection::register_int_wrapped_field(
      const std::string & unique_name, Dim_t nb_components,
      Eigen::Ref<typename Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>>
      values) {
    return this->register_wrapped_field_helper<Int>(unique_name, nb_components,
                                                     values);
  }

  /* ---------------------------------------------------------------------- */
  WrappedNField<Uint> &
  NFieldCollection::register_uint_wrapped_field(
      const std::string & unique_name, Dim_t nb_components,
      Eigen::Ref<typename Eigen::Matrix<Uint, Eigen::Dynamic, Eigen::Dynamic>>
      values) {
    return this->register_wrapped_field_helper<Uint>(unique_name, nb_components,
                                                     values);
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::field_exists(const std::string & unique_name) const {
    return this->fields.find(unique_name) != this->fields.end();
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::state_field_exists(
      const std::string & unique_prefix) const {
    return this->state_fields.find(unique_prefix) != this->state_fields.end();
  }

  /* ---------------------------------------------------------------------- */
  Dim_t NFieldCollection::size() const { return this->nb_entries; }

  /* ---------------------------------------------------------------------- */
  size_t NFieldCollection::get_nb_pixels() const {
    return this->nb_entries / this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::has_nb_quad() const {
    return not(this->nb_quad_pts == Unknown);
  }

  /* ---------------------------------------------------------------------- */
  void NFieldCollection::set_nb_quad(Dim_t nb_quad_pts_per_pixel) {
    if (this->has_nb_quad() and (this->nb_quad_pts != nb_quad_pts_per_pixel)) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel has already been set "
               "to "
            << this->nb_quad_pts << " and cannot be changed";
      throw NFieldCollectionError(error.str());
    }
    if (nb_quad_pts_per_pixel < 1) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel must be positive. "
            << "You chose " << nb_quad_pts_per_pixel;
      throw NFieldCollectionError(error.str());
    }
    this->nb_quad_pts = nb_quad_pts_per_pixel;
  }

  /* ---------------------------------------------------------------------- */
  Dim_t NFieldCollection::get_nb_quad() const {
    return this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::get_domain() const -> const Domain & {
    return this->domain;
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::is_initialised() const { return this->initialised; }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::begin() const -> iterator {
    if (not this->is_initialised()) {
      throw NFieldCollectionError(
          "Can't iterate over a collection before it's initialised");
    }
    return this->indices.begin();
  }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::end() const -> iterator { return this->indices.end(); }

  /* ---------------------------------------------------------------------- */
  void NFieldCollection::allocate_fields() {
    for (auto && item : this->fields) {
      auto && field{*item.second};
      const auto field_size = field.size();
      if ((field_size != 0) and (field_size != size_t(this->size()))) {
        std::stringstream err_stream;
        err_stream << "NField '" << field.get_name() << "' contains "
                   << field_size << " entries, but the field collection "
                   << "has " << this->size() << " pixels";
        throw NFieldCollectionError(err_stream.str());
      }
      // resize is being called unconditionally, because it alone guarantees
      // the exactness of the field's `data_ptr`
      field.resize(this->size());
    }
  }

  /* ---------------------------------------------------------------------- */
  NField & NFieldCollection::get_field(const std::string & unique_name) {
    if (not this->field_exists(unique_name)) {
      std::stringstream err_stream{};
      err_stream << "The field '" << unique_name << "' does not exist";
      throw NFieldCollectionError(err_stream.str());
    }
    return *this->fields[unique_name];
  }

  /* ---------------------------------------------------------------------- */
  StateNField &
  NFieldCollection::get_state_field(const std::string & unique_prefix) {
    if (not this->state_field_exists(unique_prefix)) {
      std::stringstream err_stream{};
      err_stream << "The state field '" << unique_prefix << "' does not exist";
      throw NFieldCollectionError(err_stream.str());
    }
    return *this->state_fields[unique_prefix];
  }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::get_fields() const
      -> const std::map<std::string, NField_ptr> & {
    return this->fields;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> NFieldCollection::list_fields() const {
    std::vector<std::string> field_names;
    for (auto & f: this->get_fields()) {
      field_names.push_back(std::get<0>(f));
    }
    return field_names;
  }

  /**
   * Technically, these explicit instantiations are not necessary, as they are
   * implicitly instantiated when the register_<T>field(...) member functions
   * are compiled.
   */
  template TypedNField<Real> &
  NFieldCollection::register_field<Real>(const std::string &, Dim_t);

  template TypedNField<Complex> &
  NFieldCollection::register_field<Complex>(const std::string &, Dim_t);

  template TypedNField<Int> &
  NFieldCollection::register_field<Int>(const std::string &, Dim_t);

  template TypedNField<Uint> &
  NFieldCollection::register_field<Uint>(const std::string &, Dim_t);
}  // namespace muGrid
