/**
 * @file   field_collection.cc
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

#include "field_collection.hh"
#include "field.hh"
#include "state_field.hh"
#include "field_typed.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <class DefaultDestroyable>
  void FieldDestructor<DefaultDestroyable>::
  operator()(DefaultDestroyable * field) {
    delete field;
  }

  /* ---------------------------------------------------------------------- */
  template struct FieldDestructor<Field>;
  template struct FieldDestructor<StateField>;

  /* ---------------------------------------------------------------------- */
  FieldCollection::FieldCollection(ValidityDomain domain,
                                   const Dim_t & spatial_dimension,
                                   const Dim_t & nb_quad_pts)
      : domain{domain}, spatial_dim{spatial_dimension}, nb_quad_pts{
                                                            nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> &
  FieldCollection::register_field_helper(const std::string & unique_name,
                                         const Dim_t & nb_dof_per_quad_pt) {
    static_assert(std::is_scalar<T>::value or std::is_same<T, Complex>::value,
                  "You can only register fields templated with one of the "
                  "numeric types Real, Complex, Int, or UInt");
    if (this->field_exists(unique_name)) {
      std::stringstream error{};
      error << "A Field of name '" << unique_name
            << "' is already registered in this field collection. "
            << "Currently registered fields: ";
      std::string prelude{""};
      for (const auto & name_field_pair : this->fields) {
        error << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw FieldCollectionError(error.str());
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedField with
    //! the number of components specified in 'int' rather than 'size_t'.
    TypedField<T> * raw_ptr{
        new TypedField<T>{unique_name, *this, nb_dof_per_quad_pt}};
    TypedField<T> & retref{*raw_ptr};
    Field_ptr field{raw_ptr};
    if (this->initialised) {
      retref.resize(this->get_nb_entries());
    }
    this->fields[unique_name] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  TypedField<Real> &
  FieldCollection::register_real_field(const std::string & unique_name,
                                       const Dim_t & nb_dof_per_quad_pt) {
    return this->register_field_helper<Real>(unique_name, nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedField<Complex> &
  FieldCollection::register_complex_field(const std::string & unique_name,
                                          const Dim_t & nb_dof_per_quad_pt) {
    return this->register_field_helper<Complex>(unique_name,
                                                nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedField<Int> &
  FieldCollection::register_int_field(const std::string & unique_name,
                                      const Dim_t & nb_dof_per_quad_pt) {
    return this->register_field_helper<Int>(unique_name, nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedField<Uint> &
  FieldCollection::register_uint_field(const std::string & unique_name,
                                       const Dim_t & nb_dof_per_quad_pt) {
    return this->register_field_helper<Uint>(unique_name, nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedStateField<T> & FieldCollection::register_state_field_helper(
      const std::string & unique_prefix, const Dim_t & nb_memory,
      const Dim_t & nb_dof_per_quad_pt) {
    static_assert(
        std::is_scalar<T>::value or std::is_same<T, Complex>::value,
        "You can only register state fields templated with one of the "
        "numeric types Real, Complex, Int, or UInt");
    if (this->state_field_exists(unique_prefix)) {
      std::stringstream error{};
      error << "A StateField of name '" << unique_prefix
            << "' is already registered in this field collection. "
            << "Currently registered state fields: ";
      std::string prelude{""};
      for (const auto & name_field_pair : this->state_fields) {
        error << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw FieldCollectionError(error.str());
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedField
    //! with the number of components specified in 'int' rather than 'size_t'.
    TypedStateField<T> * raw_ptr{
        new TypedStateField<T>{unique_prefix, *this, nb_memory,
                               nb_dof_per_quad_pt}};
    TypedStateField<T> & retref{*raw_ptr};
    StateField_ptr field{raw_ptr};
    this->state_fields[unique_prefix] = std::move(field);
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  TypedStateField<Real> &
  FieldCollection::register_real_state_field(const std::string & unique_name,
                                             const Dim_t & nb_memory,
                                             const Dim_t & nb_dof_per_quad_pt) {
    return this->register_state_field_helper<Real>(unique_name, nb_memory,
                                                   nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateField<Complex> &
  FieldCollection::register_complex_state_field(
      const std::string & unique_name, const Dim_t & nb_memory,
      const Dim_t & nb_dof_per_quad_pt) {
    return this->register_state_field_helper<Complex>(unique_name, nb_memory,
                                                      nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateField<Int> &
  FieldCollection::register_int_state_field(const std::string & unique_name,
                                            const Dim_t & nb_memory,
                                            const Dim_t & nb_dof_per_quad_pt) {
    return this->register_state_field_helper<Int>(unique_name, nb_memory,
                                                  nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  TypedStateField<Uint> &
  FieldCollection::register_uint_state_field(const std::string & unique_name,
                                             const Dim_t & nb_memory,
                                             const Dim_t & nb_dof_per_quad_pt) {
    return this->register_state_field_helper<Uint>(unique_name, nb_memory,
                                                   nb_dof_per_quad_pt);
  }

  /* ---------------------------------------------------------------------- */
  bool FieldCollection::field_exists(const std::string & unique_name) const {
    return this->fields.find(unique_name) != this->fields.end();
  }

  /* ---------------------------------------------------------------------- */
  bool
  FieldCollection::state_field_exists(const std::string & unique_prefix) const {
    return this->state_fields.find(unique_prefix) != this->state_fields.end();
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FieldCollection::get_nb_entries() const {
    return this->nb_entries;
  }

  /* ---------------------------------------------------------------------- */
  Dim_t FieldCollection::get_nb_pixels() const {
    assert(this->initialised);
    return this->nb_entries / this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  bool FieldCollection::has_nb_quad_pts() const {
    return not(this->nb_quad_pts == Unknown);
  }

  /* ---------------------------------------------------------------------- */
  void FieldCollection::set_nb_quad_pts(Dim_t nb_quad_pts_per_pixel) {
    if (this->has_nb_quad_pts() and
        (this->nb_quad_pts != nb_quad_pts_per_pixel)) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel has already been set "
               "to "
            << this->nb_quad_pts << " and cannot be changed";
      throw FieldCollectionError(error.str());
    }
    if (nb_quad_pts_per_pixel < 1) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel must be positive. "
            << "You chose " << nb_quad_pts_per_pixel;
      throw FieldCollectionError(error.str());
    }
    this->nb_quad_pts = nb_quad_pts_per_pixel;
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FieldCollection::get_nb_quad_pts() const {
    return this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FieldCollection::get_spatial_dim() const {
    return this->spatial_dim;
  }
  /* ---------------------------------------------------------------------- */
  auto FieldCollection::get_domain() const -> const ValidityDomain & {
    return this->domain;
  }

  /* ---------------------------------------------------------------------- */
  bool FieldCollection::is_initialised() const { return this->initialised; }

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::get_pixel_indices_fast() const -> PixelIndexIterable {
    return PixelIndexIterable{*this};
  }

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::get_pixel_indices() const -> IndexIterable {
    return IndexIterable{*this, Iteration::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::get_quad_pt_indices() const -> IndexIterable {
    return IndexIterable{*this, Iteration::QuadPt};
  }

  /* ---------------------------------------------------------------------- */
  void FieldCollection::allocate_fields() {
    for (auto && item : this->fields) {
      auto && field{*item.second};
      const auto field_size = field.size();
      if ((field_size != 0) and
          (field_size != size_t(this->get_nb_entries()))) {
        std::stringstream err_stream;
        err_stream << "Field '" << field.get_name() << "' contains "
                   << field_size << " entries, but the field collection "
                   << "has " << this->get_nb_entries() << " pixels";
        throw FieldCollectionError(err_stream.str());
      }
      // resize is being called unconditionally, because it alone guarantees
      // the exactness of the field's `data_ptr`
      field.resize(this->get_nb_entries());
    }
  }

  /* ---------------------------------------------------------------------- */
  void FieldCollection::initialise_maps() {
    for (auto & weak_callback : this->init_callbacks) {
      if (auto shared_callback{weak_callback.lock()}) {
        auto && callback{*shared_callback};
        callback();
      }
    }
    this->init_callbacks.clear();
  }

  /* ---------------------------------------------------------------------- */
  Field & FieldCollection::get_field(const std::string & unique_name) {
    if (not this->field_exists(unique_name)) {
      std::stringstream err_stream{};
      err_stream << "The field '" << unique_name << "' does not exist";
      throw FieldCollectionError(err_stream.str());
    }
    return *this->fields[unique_name];
  }

  /* ---------------------------------------------------------------------- */
  StateField &
  FieldCollection::get_state_field(const std::string & unique_prefix) {
    if (not this->state_field_exists(unique_prefix)) {
      std::stringstream err_stream{};
      err_stream << "The state field '" << unique_prefix << "' does not exist";
      throw FieldCollectionError(err_stream.str());
    }
    return *this->state_fields[unique_prefix];
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> FieldCollection::list_fields() const {
    std::vector<std::string> field_names;
    for (const auto & f : this->fields) {
      field_names.push_back(std::get<0>(f));
    }
    return field_names;
  }

  /* ---------------------------------------------------------------------- */
  void FieldCollection::preregister_map(
      std::shared_ptr<std::function<void()>> & call_back) {
    if (this->initialised) {
      throw FieldCollectionError("Collection is already initialised");
    }
    this->init_callbacks.push_back(call_back);
  }

  /**
   * Technically, these explicit instantiations are not necessary, as they are
   * implicitly instantiated when the register_<T>field(...) member functions
   * are compiled.
   */
  template TypedField<Real> &
  FieldCollection::register_field<Real>(const std::string &, const Dim_t &);

  template TypedField<Complex> &
  FieldCollection::register_field<Complex>(const std::string &, const Dim_t &);

  template TypedField<Int> &
  FieldCollection::register_field<Int>(const std::string &, const Dim_t &);

  template TypedField<Uint> &
  FieldCollection::register_field<Uint>(const std::string &, const Dim_t &);

  /* ---------------------------------------------------------------------- */
  FieldCollection::PixelIndexIterable::PixelIndexIterable(
      const FieldCollection & collection)
      : collection{collection} {}

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::PixelIndexIterable::begin() const -> iterator {
    return this->collection.pixel_indices.begin();
  }

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::PixelIndexIterable::end() const -> iterator {
    return this->collection.pixel_indices.end();
  }

  /* ---------------------------------------------------------------------- */
  size_t FieldCollection::PixelIndexIterable::size() const {
    return this->collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  FieldCollection::IndexIterable::IndexIterable(
      const FieldCollection & collection, const Iteration & iteration_type)
      : collection{collection}, iteration_type{iteration_type} {}

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::IndexIterable::begin() const -> iterator {
    return iterator(this->collection.pixel_indices.begin(), this->get_stride());
  }

  /* ---------------------------------------------------------------------- */
  auto FieldCollection::IndexIterable::end() const -> iterator {
    return iterator(this->collection.pixel_indices.end(), this->get_stride());
  }

  /* ---------------------------------------------------------------------- */
  size_t FieldCollection::IndexIterable::size() const {
    return (this->iteration_type == Iteration::QuadPt)
               ? this->collection.get_nb_entries()
               : this->collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  FieldCollection::IndexIterable::iterator::iterator(
      const PixelIndexIterator_t & pixel_index_iterator, const size_t & stride)
      : stride{stride}, pixel_index_iterator{pixel_index_iterator} {}
}  // namespace muGrid
