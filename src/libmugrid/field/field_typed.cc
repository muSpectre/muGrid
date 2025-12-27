/**
 * @file   field_typed.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Aug 2019
 *
 * @brief  Implementation for typed fields
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

#include <sstream>

#include "grid/index_ops.hh"
#include "field/field_typed.hh"
#include "collection/field_collection.hh"
#include "field/field_map.hh"
#include "util/tensor_algebra.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  void * TypedFieldBase<T, MemorySpace>::get_void_data_ptr(
      bool assert_host_memory) const {
    if (assert_host_memory && !is_host_space_v<MemorySpace>) {
      throw FieldError(
          "get_void_data_ptr called on device-space field with "
          "assert_host_memory=true. Set assert_host_memory=false only when "
          "passing to CUDA-aware libraries (e.g., CUDA-aware MPI).");
    }
    return static_cast<void *>(const_cast<T *>(this->values.data()));
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::operator=(const TypedField & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::operator=(const Parent & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::operator=(const Negative & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::operator=(const EigenRep_t & other) {
    Parent::operator=(other);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  void TypedField<T, MemorySpace>::set_zero() {
    // Use our deep_copy that handles scalar fill
    muGrid::deep_copy(this->values, T{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::safe_cast(Field & other) {
    try {
      return dynamic_cast<TypedField<T, MemorySpace> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Can not cast field '" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_typeid().name()
            << "'.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  const TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::safe_cast(const Field & other) {
    try {
      return dynamic_cast<const TypedField<T, MemorySpace> &>(other);
    } catch (const std::bad_cast &) {
      std::stringstream error{};
      error << "Can not cast field '" << other.get_name()
            << "' to a typed field of type '" << typeid(T).name()
            << "', because it is of type '" << other.get_typeid().name()
            << "'.";
      throw FieldError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::safe_cast(Field & other,
                                        const Index_t & nb_components,
                                        const std::string & sub_division) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream err_msg{};
      err_msg << "Can not cast field '" << other.get_name()
              << "', because it has " << other.get_nb_components()
              << " degrees of freedom per sub-point, rather than the "
              << nb_components << " components which are requested.";
      throw FieldError(err_msg.str());
    }
    if (other.get_sub_division_tag() != sub_division) {
      std::stringstream err_msg{};
      err_msg << "Can not cast field '" << other.get_name()
              << "', because it's subdivision is '"
              << other.get_sub_division_tag() << "', rather than "
              << sub_division << ", which are requested.";
      throw FieldError(err_msg.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  const TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::safe_cast(const Field & other,
                                        const Index_t & nb_components,
                                        const std::string & sub_division) {
    if (other.get_nb_components() != nb_components) {
      std::stringstream err_msg{};
      err_msg << "Can not cast field '" << other.get_name()
              << "', because it has " << other.get_nb_components()
              << " degrees of freedom per sub-point, rather than the "
              << nb_components << " components which are requested.";
      throw FieldError(err_msg.str());
    }
    if (other.get_sub_division_tag() != sub_division) {
      std::stringstream err_msg{};
      err_msg << "Can not cast field '" << other.get_name()
              << "', because it's subdivision is '"
              << other.get_sub_division_tag() << "', rather than "
              << sub_division << ", which are requested.";
      throw FieldError(err_msg.str());
    }
    return TypedField::safe_cast(other);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  void TypedField<T, MemorySpace>::resize() {
    if (not this->has_nb_sub_pts()) {
      std::stringstream error_message{};
      error_message << "Can't compute the size of field '" << this->get_name()
                    << "' because the number of points per pixel for "
                       "subdivisions tagged '"
                    << this->get_sub_division_tag() << "' is not yet known.";
      throw FieldError(error_message.str());
    }

    auto && size{this->nb_sub_pts * this->get_nb_buffer_pixels()};
    const auto expected_size{size * this->get_nb_components()};
    if (this->values.size() != static_cast<size_t>(expected_size) or
        static_cast<Index_t>(this->current_nb_entries) != size) {
      this->current_nb_entries = size;
      // Use our resize function
      muGrid::resize(this->values, expected_size);
      // Zero-initialize the new memory
      this->values.fill_zero();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  size_t TypedField<T, MemorySpace>::get_buffer_size() const {
    return this->values.size();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, void>
  TypedField<T, MemorySpace>::push_back(const T & value) {
    if (this->is_global()) {
      throw FieldError("push_back() makes no sense on global fields (you can't "
                       "add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Can not push_back into a field before the number of "
                       "sub-division points has been set for it");
    }
    if (this->nb_components != 1) {
      throw FieldError("This is not a scalar field. push_back an array.");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_nb_entries += nb_sub;
    const auto old_size{static_cast<Index_t>(this->values.size())};
    const auto new_size{old_size + nb_sub};
    // Array resize doesn't preserve data, so we need a temp copy
    Array<T, MemorySpace> old_values(old_size);
    muGrid::deep_copy(old_values, this->values);
    muGrid::resize(this->values, new_size);
    // Copy old data back
    for (Index_t i{0}; i < old_size; ++i) {
      this->values[i] = old_values[i];
    }
    // Add new values
    for (Index_t sub_pt_id{0}; sub_pt_id < nb_sub; ++sub_pt_id) {
      this->values[old_size + sub_pt_id] = value;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, void>
  TypedField<T, MemorySpace>::push_back_single(const T & value) {
    if (this->is_global()) {
      throw FieldError("push_back_single() makes no sense on global fields "
                       "(you can't add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Can not push_back_single into a field before the "
                       "number of sub-division points has been set for it");
    }
    if (this->nb_components != 1) {
      throw FieldError("This is not a scalar field. push_back an array.");
    }
    this->current_nb_entries += 1;
    const auto old_size{static_cast<Index_t>(this->values.size())};
    const auto new_size{old_size + 1};
    // Array resize doesn't preserve data, so we need a temp copy
    Array<T, MemorySpace> old_values(old_size);
    muGrid::deep_copy(old_values, this->values);
    muGrid::resize(this->values, new_size);
    // Copy old data back
    for (Index_t i{0}; i < old_size; ++i) {
      this->values[i] = old_values[i];
    }
    this->values[old_size] = value;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, void>
  TypedField<T, MemorySpace>::push_back(
      const Eigen::Ref<
          const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> & value) {
    if (this->is_global()) {
      throw FieldError("push_back() makes no sense on global fields (you can't "
                       "add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Can not push_back into a field before the number of "
                       "sub-division points has been set for it");
    }
    if (this->nb_components != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << " components into a field with " << this->nb_components
            << " components.";
      throw FieldError(error.str());
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    this->current_nb_entries += nb_sub;
    const auto old_size{static_cast<Index_t>(this->values.size())};
    const auto new_size{old_size + this->nb_components * nb_sub};
    // Array resize doesn't preserve data, so we need a temp copy
    Array<T, MemorySpace> old_values(old_size);
    muGrid::deep_copy(old_values, this->values);
    muGrid::resize(this->values, new_size);
    // Copy old data back
    for (Index_t i{0}; i < old_size; ++i) {
      this->values[i] = old_values[i];
    }
    // Add new values
    for (Index_t sub_pt_id{0}; sub_pt_id < nb_sub; ++sub_pt_id) {
      for (Index_t i{0}; i < this->nb_components; ++i) {
        this->values[old_size + sub_pt_id * this->nb_components + i] = value.data()[i];
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, void>
  TypedField<T, MemorySpace>::push_back_single(
      const Eigen::Ref<
          const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> & value) {
    if (this->is_global()) {
      throw FieldError("push_back_single() makes no sense on global fields "
                       "(you can't add individual pixels");
    }
    if (not this->has_nb_sub_pts()) {
      throw FieldError("Can not push_back_single into a field before the number"
                       " of sub-division points has been set for it");
    }
    if (this->nb_components != value.size()) {
      std::stringstream error{};
      error << "You are trying to push an array with " << value.size()
            << " components into a field with " << this->nb_components
            << " components.";
      throw FieldError(error.str());
    }
    this->current_nb_entries += 1;
    const auto old_size{static_cast<Index_t>(this->values.size())};
    const auto new_size{old_size + this->nb_components};
    // Array resize doesn't preserve data, so we need a temp copy
    Array<T, MemorySpace> old_values(old_size);
    muGrid::deep_copy(old_values, this->values);
    muGrid::resize(this->values, new_size);
    // Copy old data back
    for (Index_t i{0}; i < old_size; ++i) {
      this->values[i] = old_values[i];
    }
    // Add new values
    for (Index_t i{0}; i < this->nb_components; ++i) {
      this->values[old_size + i] = value.data()[i];
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedField<T, MemorySpace> &
  TypedField<T, MemorySpace>::clone(const std::string & new_name,
                                    const bool & allow_overwrite) const {
    const bool field_exists{this->get_collection().field_exists(new_name)};

    if (field_exists and not allow_overwrite) {
      std::stringstream err_msg{};
      err_msg << "The field '" << new_name
              << "' already exists, and you did not set 'allow_overwrite' "
                 "to true";
      throw FieldError{err_msg.str()};
    }

    TypedField<T, MemorySpace> & other{
        field_exists
            ? this->safe_cast(this->get_collection().get_field(new_name),
                              this->nb_components, this->sub_division_tag)
            : this->safe_cast(
                  this->get_collection().template register_field<T>(
                      new_name, this->nb_components, this->sub_division_tag,
                      this->unit),
                  this->nb_components, this->sub_division_tag)};

    muGrid::deep_copy(other.view(), this->view());
    return other;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_map>
  TypedFieldBase<T, MemorySpace>::eigen_map(const Index_t & nb_rows,
                                            const Index_t & nb_cols) {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    if (not CcoordOps::is_buffer_contiguous(this->get_pixels_shape(),
                                            this->get_pixels_strides())) {
      throw FieldError("Eigen representation is only available for fields with "
                       "contiguous storage.");
    }
    return Eigen_map(this->values.data(), nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_cmap>
  TypedFieldBase<T, MemorySpace>::eigen_map(const Index_t & nb_rows,
                                            const Index_t & nb_cols) const {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    if (not CcoordOps::is_buffer_contiguous(this->get_pixels_shape(),
                                            this->get_pixels_strides())) {
      throw FieldError("Eigen representation is only available for fields with "
                       "contiguous storage.");
    }
    return Eigen_cmap(this->values.data(), nb_rows, nb_cols);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedFieldBase<T, MemorySpace> &
  TypedFieldBase<T, MemorySpace>::operator=(const TypedFieldBase & other) {
    // Use our deep_copy function
    muGrid::deep_copy(this->values, other.values);
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedFieldBase<T, MemorySpace> &
  TypedFieldBase<T, MemorySpace>::operator=(const Negative & other) {
    // This requires host access for Eigen operations
    if constexpr (!is_host_space_v<MemorySpace>) {
      throw FieldError("Negative assignment only available for host-space fields");
    } else {
      this->eigen_vec() = -other.field.eigen_vec();
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedFieldBase<T, MemorySpace> &
  TypedFieldBase<T, MemorySpace>::operator=(const EigenRep_t & other) {
    if constexpr (!is_host_space_v<MemorySpace>) {
      throw FieldError("Eigen assignment only available for host-space fields");
    } else {
      this->eigen_vec() = other;
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  auto TypedFieldBase<T, MemorySpace>::operator-() const -> Negative {
    return Negative{*this};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedFieldBase<T, MemorySpace> &
  TypedFieldBase<T, MemorySpace>::operator+=(const TypedFieldBase & other) {
    if constexpr (!is_host_space_v<MemorySpace>) {
      throw FieldError("+= only available for host-space fields");
    } else {
      this->eigen_vec() += other.eigen_vec();
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  TypedFieldBase<T, MemorySpace> &
  TypedFieldBase<T, MemorySpace>::operator-=(const TypedFieldBase & other) {
    if constexpr (!is_host_space_v<MemorySpace>) {
      throw FieldError("-= only available for host-space fields");
    } else {
      this->eigen_vec() -= other.eigen_vec();
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::EigenVec_map>
  TypedFieldBase<T, MemorySpace>::eigen_vec() {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    return EigenVec_map(this->values.data(),
                        this->get_nb_entries() * this->nb_components);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::EigenVec_cmap>
  TypedFieldBase<T, MemorySpace>::eigen_vec() const {
    if (not this->collection.is_initialised()) {
      std::stringstream error{};
      error << "The FieldCollection for field '" << this->name
            << "' has not been initialised";
      throw FieldError(error.str());
    }
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return EigenVec_cmap(this->values.data(),
                         this->get_nb_entries() * this->nb_components);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_map>
  TypedFieldBase<T, MemorySpace>::eigen_sub_pt() {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->template eigen_map<M>(this->nb_components,
                                       this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_cmap>
  TypedFieldBase<T, MemorySpace>::eigen_sub_pt() const {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    return this->template eigen_map<M>(this->nb_components,
                                       this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_map>
  TypedFieldBase<T, MemorySpace>::eigen_pixel() {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->template eigen_map<M>(this->nb_components * nb_sub,
                                       this->get_nb_entries() / nb_sub);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_cmap>
  TypedFieldBase<T, MemorySpace>::eigen_pixel() const {
    if (this->get_nb_entries() == Unknown) {
      throw FieldError("Field has unknown number of entries");
    }
    const auto & nb_sub{this->get_nb_sub_pts()};
    return this->template eigen_map<M>(this->nb_components * nb_sub,
                                       this->get_nb_entries() / nb_sub);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_map>
  TypedFieldBase<T, MemorySpace>::eigen_mat() {
    return this->template eigen_map<M>(this->nb_components,
                                       this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>,
                   typename TypedFieldBase<T, MemorySpace>::Eigen_cmap>
  TypedFieldBase<T, MemorySpace>::eigen_mat() const {
    return this->template eigen_map<M>(this->nb_components,
                                       this->get_nb_entries());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Mut>>
  TypedFieldBase<T, MemorySpace>::get_pixel_map(const Index_t & nb_rows) {
    return (nb_rows == Unknown)
               ? FieldMap<T, Mapping::Mut>{*this, IterUnit::Pixel}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, IterUnit::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Const>>
  TypedFieldBase<T, MemorySpace>::get_pixel_map(const Index_t & nb_rows) const {
    return (nb_rows == Unknown)
               ? FieldMap<T, Mapping::Const>{*this, IterUnit::Pixel}
               : FieldMap<T, Mapping::Const>{*this, nb_rows, IterUnit::Pixel};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Mut>>
  TypedFieldBase<T, MemorySpace>::get_sub_pt_map(const Index_t & nb_rows) {
    return (nb_rows == Unknown)
               ? FieldMap<T, Mapping::Mut>{*this, IterUnit::SubPt}
               : FieldMap<T, Mapping::Mut>{*this, nb_rows, IterUnit::SubPt};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename M>
  std::enable_if_t<is_host_space_v<M>, FieldMap<T, Mapping::Const>>
  TypedFieldBase<T, MemorySpace>::get_sub_pt_map(
      const Index_t & nb_rows) const {
    return (nb_rows == Unknown)
               ? FieldMap<T, Mapping::Const>{*this, IterUnit::SubPt}
               : FieldMap<T, Mapping::Const>{*this, nb_rows, IterUnit::SubPt};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, typename MemorySpace>
  template <typename OtherSpace>
  void TypedFieldBase<T, MemorySpace>::deep_copy_from(
      const TypedFieldBase<T, OtherSpace> & src) {
    muGrid::deep_copy(*this, src);
  }

  /* ---------------------------------------------------------------------- */
  // Explicit template instantiations for HostSpace only
  template class TypedFieldBase<Real, HostSpace>;
  template class TypedFieldBase<Complex, HostSpace>;
  template class TypedFieldBase<Int, HostSpace>;
  template class TypedFieldBase<Uint, HostSpace>;
  template class TypedFieldBase<Index_t, HostSpace>;

  template class TypedField<Real, HostSpace>;
  template class TypedField<Complex, HostSpace>;
  template class TypedField<Int, HostSpace>;
  template class TypedField<Uint, HostSpace>;
  template class TypedField<Index_t, HostSpace>;

  // Explicit instantiation of template member functions for HostSpace
  template TypedFieldBase<Real, HostSpace>::Eigen_map
  TypedFieldBase<Real, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &);
  template TypedFieldBase<Real, HostSpace>::Eigen_cmap
  TypedFieldBase<Real, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &) const;
  template TypedFieldBase<Real, HostSpace>::EigenVec_map
  TypedFieldBase<Real, HostSpace>::eigen_vec<HostSpace>();
  template TypedFieldBase<Real, HostSpace>::EigenVec_cmap
  TypedFieldBase<Real, HostSpace>::eigen_vec<HostSpace>() const;
  template TypedFieldBase<Real, HostSpace>::Eigen_map
  TypedFieldBase<Real, HostSpace>::eigen_sub_pt<HostSpace>();
  template TypedFieldBase<Real, HostSpace>::Eigen_cmap
  TypedFieldBase<Real, HostSpace>::eigen_sub_pt<HostSpace>() const;
  template TypedFieldBase<Real, HostSpace>::Eigen_map
  TypedFieldBase<Real, HostSpace>::eigen_pixel<HostSpace>();
  template TypedFieldBase<Real, HostSpace>::Eigen_cmap
  TypedFieldBase<Real, HostSpace>::eigen_pixel<HostSpace>() const;
  template TypedFieldBase<Real, HostSpace>::Eigen_map
  TypedFieldBase<Real, HostSpace>::eigen_mat<HostSpace>();
  template TypedFieldBase<Real, HostSpace>::Eigen_cmap
  TypedFieldBase<Real, HostSpace>::eigen_mat<HostSpace>() const;
  template FieldMap<Real, Mapping::Mut>
  TypedFieldBase<Real, HostSpace>::get_pixel_map<HostSpace>(const Index_t &);
  template FieldMap<Real, Mapping::Const>
  TypedFieldBase<Real, HostSpace>::get_pixel_map<HostSpace>(
      const Index_t &) const;
  template FieldMap<Real, Mapping::Mut>
  TypedFieldBase<Real, HostSpace>::get_sub_pt_map<HostSpace>(const Index_t &);
  template FieldMap<Real, Mapping::Const>
  TypedFieldBase<Real, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &) const;

  template TypedFieldBase<Complex, HostSpace>::Eigen_map
  TypedFieldBase<Complex, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &);
  template TypedFieldBase<Complex, HostSpace>::Eigen_cmap
  TypedFieldBase<Complex, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &) const;
  template TypedFieldBase<Complex, HostSpace>::EigenVec_map
  TypedFieldBase<Complex, HostSpace>::eigen_vec<HostSpace>();
  template TypedFieldBase<Complex, HostSpace>::EigenVec_cmap
  TypedFieldBase<Complex, HostSpace>::eigen_vec<HostSpace>() const;
  template TypedFieldBase<Complex, HostSpace>::Eigen_map
  TypedFieldBase<Complex, HostSpace>::eigen_sub_pt<HostSpace>();
  template TypedFieldBase<Complex, HostSpace>::Eigen_cmap
  TypedFieldBase<Complex, HostSpace>::eigen_sub_pt<HostSpace>() const;
  template TypedFieldBase<Complex, HostSpace>::Eigen_map
  TypedFieldBase<Complex, HostSpace>::eigen_pixel<HostSpace>();
  template TypedFieldBase<Complex, HostSpace>::Eigen_cmap
  TypedFieldBase<Complex, HostSpace>::eigen_pixel<HostSpace>() const;
  template TypedFieldBase<Complex, HostSpace>::Eigen_map
  TypedFieldBase<Complex, HostSpace>::eigen_mat<HostSpace>();
  template TypedFieldBase<Complex, HostSpace>::Eigen_cmap
  TypedFieldBase<Complex, HostSpace>::eigen_mat<HostSpace>() const;
  template FieldMap<Complex, Mapping::Mut>
  TypedFieldBase<Complex, HostSpace>::get_pixel_map<HostSpace>(const Index_t &);
  template FieldMap<Complex, Mapping::Const>
  TypedFieldBase<Complex, HostSpace>::get_pixel_map<HostSpace>(
      const Index_t &) const;
  template FieldMap<Complex, Mapping::Mut>
  TypedFieldBase<Complex, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &);
  template FieldMap<Complex, Mapping::Const>
  TypedFieldBase<Complex, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &) const;

  template TypedFieldBase<Int, HostSpace>::Eigen_map
  TypedFieldBase<Int, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &);
  template TypedFieldBase<Int, HostSpace>::Eigen_cmap
  TypedFieldBase<Int, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &) const;
  template TypedFieldBase<Int, HostSpace>::EigenVec_map
  TypedFieldBase<Int, HostSpace>::eigen_vec<HostSpace>();
  template TypedFieldBase<Int, HostSpace>::EigenVec_cmap
  TypedFieldBase<Int, HostSpace>::eigen_vec<HostSpace>() const;
  template TypedFieldBase<Int, HostSpace>::Eigen_map
  TypedFieldBase<Int, HostSpace>::eigen_sub_pt<HostSpace>();
  template TypedFieldBase<Int, HostSpace>::Eigen_cmap
  TypedFieldBase<Int, HostSpace>::eigen_sub_pt<HostSpace>() const;
  template TypedFieldBase<Int, HostSpace>::Eigen_map
  TypedFieldBase<Int, HostSpace>::eigen_pixel<HostSpace>();
  template TypedFieldBase<Int, HostSpace>::Eigen_cmap
  TypedFieldBase<Int, HostSpace>::eigen_pixel<HostSpace>() const;
  template TypedFieldBase<Int, HostSpace>::Eigen_map
  TypedFieldBase<Int, HostSpace>::eigen_mat<HostSpace>();
  template TypedFieldBase<Int, HostSpace>::Eigen_cmap
  TypedFieldBase<Int, HostSpace>::eigen_mat<HostSpace>() const;
  template FieldMap<Int, Mapping::Mut>
  TypedFieldBase<Int, HostSpace>::get_pixel_map<HostSpace>(const Index_t &);
  template FieldMap<Int, Mapping::Const>
  TypedFieldBase<Int, HostSpace>::get_pixel_map<HostSpace>(
      const Index_t &) const;
  template FieldMap<Int, Mapping::Mut>
  TypedFieldBase<Int, HostSpace>::get_sub_pt_map<HostSpace>(const Index_t &);
  template FieldMap<Int, Mapping::Const>
  TypedFieldBase<Int, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &) const;

  template TypedFieldBase<Uint, HostSpace>::Eigen_map
  TypedFieldBase<Uint, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &);
  template TypedFieldBase<Uint, HostSpace>::Eigen_cmap
  TypedFieldBase<Uint, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &) const;
  template TypedFieldBase<Uint, HostSpace>::EigenVec_map
  TypedFieldBase<Uint, HostSpace>::eigen_vec<HostSpace>();
  template TypedFieldBase<Uint, HostSpace>::EigenVec_cmap
  TypedFieldBase<Uint, HostSpace>::eigen_vec<HostSpace>() const;
  template TypedFieldBase<Uint, HostSpace>::Eigen_map
  TypedFieldBase<Uint, HostSpace>::eigen_sub_pt<HostSpace>();
  template TypedFieldBase<Uint, HostSpace>::Eigen_cmap
  TypedFieldBase<Uint, HostSpace>::eigen_sub_pt<HostSpace>() const;
  template TypedFieldBase<Uint, HostSpace>::Eigen_map
  TypedFieldBase<Uint, HostSpace>::eigen_pixel<HostSpace>();
  template TypedFieldBase<Uint, HostSpace>::Eigen_cmap
  TypedFieldBase<Uint, HostSpace>::eigen_pixel<HostSpace>() const;
  template TypedFieldBase<Uint, HostSpace>::Eigen_map
  TypedFieldBase<Uint, HostSpace>::eigen_mat<HostSpace>();
  template TypedFieldBase<Uint, HostSpace>::Eigen_cmap
  TypedFieldBase<Uint, HostSpace>::eigen_mat<HostSpace>() const;
  template FieldMap<Uint, Mapping::Mut>
  TypedFieldBase<Uint, HostSpace>::get_pixel_map<HostSpace>(const Index_t &);
  template FieldMap<Uint, Mapping::Const>
  TypedFieldBase<Uint, HostSpace>::get_pixel_map<HostSpace>(
      const Index_t &) const;
  template FieldMap<Uint, Mapping::Mut>
  TypedFieldBase<Uint, HostSpace>::get_sub_pt_map<HostSpace>(const Index_t &);
  template FieldMap<Uint, Mapping::Const>
  TypedFieldBase<Uint, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &) const;

  template TypedFieldBase<Index_t, HostSpace>::Eigen_map
  TypedFieldBase<Index_t, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &);
  template TypedFieldBase<Index_t, HostSpace>::Eigen_cmap
  TypedFieldBase<Index_t, HostSpace>::eigen_map<HostSpace>(
      const Index_t &, const Index_t &) const;
  template TypedFieldBase<Index_t, HostSpace>::EigenVec_map
  TypedFieldBase<Index_t, HostSpace>::eigen_vec<HostSpace>();
  template TypedFieldBase<Index_t, HostSpace>::EigenVec_cmap
  TypedFieldBase<Index_t, HostSpace>::eigen_vec<HostSpace>() const;
  template TypedFieldBase<Index_t, HostSpace>::Eigen_map
  TypedFieldBase<Index_t, HostSpace>::eigen_sub_pt<HostSpace>();
  template TypedFieldBase<Index_t, HostSpace>::Eigen_cmap
  TypedFieldBase<Index_t, HostSpace>::eigen_sub_pt<HostSpace>() const;
  template TypedFieldBase<Index_t, HostSpace>::Eigen_map
  TypedFieldBase<Index_t, HostSpace>::eigen_pixel<HostSpace>();
  template TypedFieldBase<Index_t, HostSpace>::Eigen_cmap
  TypedFieldBase<Index_t, HostSpace>::eigen_pixel<HostSpace>() const;
  template TypedFieldBase<Index_t, HostSpace>::Eigen_map
  TypedFieldBase<Index_t, HostSpace>::eigen_mat<HostSpace>();
  template TypedFieldBase<Index_t, HostSpace>::Eigen_cmap
  TypedFieldBase<Index_t, HostSpace>::eigen_mat<HostSpace>() const;
  template FieldMap<Index_t, Mapping::Mut>
  TypedFieldBase<Index_t, HostSpace>::get_pixel_map<HostSpace>(const Index_t &);
  template FieldMap<Index_t, Mapping::Const>
  TypedFieldBase<Index_t, HostSpace>::get_pixel_map<HostSpace>(
      const Index_t &) const;
  template FieldMap<Index_t, Mapping::Mut>
  TypedFieldBase<Index_t, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &);
  template FieldMap<Index_t, Mapping::Const>
  TypedFieldBase<Index_t, HostSpace>::get_sub_pt_map<HostSpace>(
      const Index_t &) const;

  // Explicit instantiation of push_back template member functions
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Real, HostSpace>::push_back<HostSpace>(const Real &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Real, HostSpace>::push_back_single<HostSpace>(const Real &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Real, HostSpace>::push_back<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>> &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Real, HostSpace>::push_back_single<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>> &);

  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Complex, HostSpace>::push_back<HostSpace>(const Complex &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Complex, HostSpace>::push_back_single<HostSpace>(const Complex &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Complex, HostSpace>::push_back<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic>> &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Complex, HostSpace>::push_back_single<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic>> &);

  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Int, HostSpace>::push_back<HostSpace>(const Int &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Int, HostSpace>::push_back_single<HostSpace>(const Int &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Int, HostSpace>::push_back<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Int, Eigen::Dynamic, Eigen::Dynamic>> &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Int, HostSpace>::push_back_single<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Int, Eigen::Dynamic, Eigen::Dynamic>> &);

  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Uint, HostSpace>::push_back<HostSpace>(const Uint &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Uint, HostSpace>::push_back_single<HostSpace>(const Uint &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Uint, HostSpace>::push_back<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Uint, Eigen::Dynamic, Eigen::Dynamic>> &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Uint, HostSpace>::push_back_single<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Uint, Eigen::Dynamic, Eigen::Dynamic>> &);

  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Index_t, HostSpace>::push_back<HostSpace>(const Index_t &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Index_t, HostSpace>::push_back_single<HostSpace>(const Index_t &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Index_t, HostSpace>::push_back<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Index_t, Eigen::Dynamic, Eigen::Dynamic>> &);
  template std::enable_if_t<is_host_space_v<HostSpace>, void>
  TypedField<Index_t, HostSpace>::push_back_single<HostSpace>(
      const Eigen::Ref<const Eigen::Array<Index_t, Eigen::Dynamic, Eigen::Dynamic>> &);

  // Device-space explicit template instantiations (for CUDA/HIP builds)
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
  // Base class instantiations for device space
  template class TypedFieldBase<Real, DefaultDeviceSpace>;
  template class TypedFieldBase<Complex, DefaultDeviceSpace>;
  template class TypedFieldBase<Int, DefaultDeviceSpace>;
  template class TypedFieldBase<Uint, DefaultDeviceSpace>;
  template class TypedFieldBase<Index_t, DefaultDeviceSpace>;

  // Derived class instantiations for device space
  template class TypedField<Real, DefaultDeviceSpace>;
  template class TypedField<Complex, DefaultDeviceSpace>;
  template class TypedField<Int, DefaultDeviceSpace>;
  template class TypedField<Uint, DefaultDeviceSpace>;
  template class TypedField<Index_t, DefaultDeviceSpace>;

  // Cross-space deep_copy_from instantiations (host -> device)
  template void TypedFieldBase<Real, DefaultDeviceSpace>::deep_copy_from<HostSpace>(
      const TypedFieldBase<Real, HostSpace> &);
  template void TypedFieldBase<Complex, DefaultDeviceSpace>::deep_copy_from<HostSpace>(
      const TypedFieldBase<Complex, HostSpace> &);
  template void TypedFieldBase<Int, DefaultDeviceSpace>::deep_copy_from<HostSpace>(
      const TypedFieldBase<Int, HostSpace> &);
  template void TypedFieldBase<Uint, DefaultDeviceSpace>::deep_copy_from<HostSpace>(
      const TypedFieldBase<Uint, HostSpace> &);
  template void TypedFieldBase<Index_t, DefaultDeviceSpace>::deep_copy_from<HostSpace>(
      const TypedFieldBase<Index_t, HostSpace> &);

  // Cross-space deep_copy_from instantiations (device -> host)
  template void TypedFieldBase<Real, HostSpace>::deep_copy_from<DefaultDeviceSpace>(
      const TypedFieldBase<Real, DefaultDeviceSpace> &);
  template void TypedFieldBase<Complex, HostSpace>::deep_copy_from<DefaultDeviceSpace>(
      const TypedFieldBase<Complex, DefaultDeviceSpace> &);
  template void TypedFieldBase<Int, HostSpace>::deep_copy_from<DefaultDeviceSpace>(
      const TypedFieldBase<Int, DefaultDeviceSpace> &);
  template void TypedFieldBase<Uint, HostSpace>::deep_copy_from<DefaultDeviceSpace>(
      const TypedFieldBase<Uint, DefaultDeviceSpace> &);
  template void TypedFieldBase<Index_t, HostSpace>::deep_copy_from<DefaultDeviceSpace>(
      const TypedFieldBase<Index_t, DefaultDeviceSpace> &);
#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

}  // namespace muGrid
