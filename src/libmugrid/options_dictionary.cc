/**
 * @file   options_dictionary.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jun 2021
 *
 * @brief  class mimicking the python dictionary
 *
 * Copyright © 2021 Till Junge
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

#include "options_dictionary.hh"

namespace muGrid {
  /* ---------------------------------------------------------------------- */
  Dictionary::Dictionary(std::shared_ptr<RuntimeValue> ptr) : ptr{ptr} {}

  /* ---------------------------------------------------------------------- */
  Dictionary::Dictionary()
      : ptr{std::make_shared<RuntimeValue>(RuntimeValue::Map_t{})} {};

  /* ---------------------------------------------------------------------- */
  Dictionary::Dictionary(const std::string & key, const Real & value)
      : ptr{std::make_shared<RuntimeValue>(RuntimeValue::Map_t{
            std::make_pair(key, std::make_shared<RuntimeValue>(value))})} {};

  /* ---------------------------------------------------------------------- */
  Dictionary::Dictionary(const std::string & key, const Int & value)
      : ptr{std::make_shared<RuntimeValue>(RuntimeValue::Map_t{
            std::make_pair(key, std::make_shared<RuntimeValue>(value))})} {};

  /* ---------------------------------------------------------------------- */
  Dictionary::Dictionary(const std::string & key,
                         const Eigen::Ref<const Eigen::MatrixXd> & value)
      : ptr{std::make_shared<RuntimeValue>(RuntimeValue::Map_t{
            std::make_pair(key, std::make_shared<RuntimeValue>(value))})} {};

  /* ---------------------------------------------------------------------- */
  Dictionary Dictionary::operator[](const std::string & key) const {
    return Dictionary{this->ptr->get_value(key)};
  }

  /* ---------------------------------------------------------------------- */
  void Dictionary::add(const std::string & key, const Dictionary & other) {
    this->ptr->add(key, other.ptr);
  }

  /* ---------------------------------------------------------------------- */
  void Dictionary::add(const std::string & key, const Int & value) {
    this->ptr->add(key, value);
  }

  /* ---------------------------------------------------------------------- */
  void Dictionary::add(const std::string & key, const Real & value) {
    this->ptr->add(key, value);
  }

  /* ---------------------------------------------------------------------- */
  void Dictionary::add(const std::string & key,
                       const Eigen::Ref<const Eigen::MatrixXd> & value) {
    this->ptr->add(key, value);
  }

  /* ---------------------------------------------------------------------- */
  const RuntimeValue::ValueType & Dictionary::get_value_type() const {
    return this->ptr->get_value_type();
  }

  /* ---------------------------------------------------------------------- */
  const Int & Dictionary::get_int() const { return this->ptr->get_int(); }

  /* ---------------------------------------------------------------------- */
  const Real & Dictionary::get_real() const { return this->ptr->get_real(); }

  /* ---------------------------------------------------------------------- */
  const Eigen::MatrixXd & Dictionary::get_matrix() const {
    return this->ptr->get_matrix();
  }

  /* ---------------------------------------------------------------------- */
  Dictionary & Dictionary::operator=(const Int & value) {
    *this->ptr = value;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  Dictionary & Dictionary::operator=(const Real & value) {
    *this->ptr = value;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  Dictionary &
  Dictionary::operator=(const Eigen::Ref<const Eigen::MatrixXd> & value) {
    *this->ptr = value;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue::RuntimeValue(const Int & value)
      : value_type{ValueType::Int}, variant(value) {}

  /* ---------------------------------------------------------------------- */
  RuntimeValue::RuntimeValue(const Real & value)
      : value_type{ValueType::Real}, variant(value) {}

  /* ---------------------------------------------------------------------- */
  RuntimeValue::RuntimeValue(const Eigen::Ref<const Eigen::MatrixXd> & value)
      : value_type{ValueType::Matrix}, variant(value) {}

  /* ---------------------------------------------------------------------- */
  RuntimeValue::RuntimeValue(const Map_t & value)
      : value_type{ValueType::Dictionary}, variant(value) {}

  /* ---------------------------------------------------------------------- */
  RuntimeValue & RuntimeValue::operator=(const Int & value) {
    this->potentially_destroy_non_trivial_member();
    this->variant.integer_value = value;
    this->value_type = ValueType::Int;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue & RuntimeValue::operator=(const Real & value) {
    this->potentially_destroy_non_trivial_member();
    this->variant.real_value = value;
    this->value_type = ValueType::Real;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue &
  RuntimeValue::operator=(const Eigen::Ref<const Eigen::MatrixXd> & value) {
    this->potentially_destroy_non_trivial_member();
    new (&this->variant.matrix_value) Eigen::MatrixXd(value);
    this->value_type = ValueType::Matrix;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue & RuntimeValue::operator=(const Map_t & value) {
    this->potentially_destroy_non_trivial_member();
    this->variant.dictionary = value;
    this->value_type = ValueType::Dictionary;
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue & RuntimeValue::operator=(const RuntimeValue & other) {
    this->potentially_destroy_non_trivial_member();
    this->value_type = other.value_type;
    switch (other.value_type) {
    case ValueType::Dictionary: {
      this->variant.dictionary = other.variant.dictionary;
      break;
    }
    case ValueType::Int: {
      this->variant.integer_value = other.variant.integer_value;
      break;
    }
    case ValueType::Real: {
      this->variant.real_value = other.variant.real_value;
      break;
    }
    case ValueType::Matrix: {
      this->variant.matrix_value = other.variant.matrix_value;
      break;
    }
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  void RuntimeValue::add(const std::string & key, T value) {
    if (this->value_type != ValueType::Dictionary) {
      throw ValueError("This is not a Dictionary value");
    }
    if (this->variant.dictionary.count(key) != 0) {
      std::stringstream error_stream{};
      error_stream << "The key '" << key
                   << "' is already present in this dictionary. did you mean "
                      "to assign rather than add?";
      throw KeyError(error_stream.str());
    }
    this->variant.dictionary.insert(
        std::make_pair(key, std::make_shared<RuntimeValue>(value)));
  }

  /* ---------------------------------------------------------------------- */
  void RuntimeValue::add(const std::string & key,
                         std::shared_ptr<RuntimeValue> other) {
    if (this->value_type != ValueType::Dictionary) {
      throw ValueError("This is not a Dictionary value");
    }
    if (this->variant.dictionary.count(key) != 0) {
      std::stringstream error_stream{};
      error_stream << "The key '" << key
                   << "' is already present in this dictionary. did you mean "
                      "to assign rather than add?";
      throw KeyError(error_stream.str());
    }
    this->variant.dictionary.insert(std::make_pair(key, other));
  }

  /* ---------------------------------------------------------------------- */
  const Int & RuntimeValue::get_int() const {
    if (this->value_type != ValueType::Int) {
      throw ValueError{"This is not an  integer value"};
    }
    return this->variant.integer_value;
  }

  /* ---------------------------------------------------------------------- */
  const Real & RuntimeValue::get_real() const {
    if (this->value_type != ValueType::Real) {
      throw ValueError{"This is not an  real value"};
    }
    return this->variant.real_value;
  }

  /* ---------------------------------------------------------------------- */
  const Eigen::MatrixXd & RuntimeValue::get_matrix() const {
    if (this->value_type != ValueType::Matrix) {
      throw ValueError{"This is not an  matrix value"};
    }
    return this->variant.matrix_value;
  }

  /* ---------------------------------------------------------------------- */
  std::shared_ptr<RuntimeValue> &
  RuntimeValue::get_value(const std::string & key) {
    if (this->value_type != ValueType::Dictionary) {
      throw ValueError{"This isn't a Dictionary value"};
    }
    return this->variant.dictionary.at(key);
  }

  /* ---------------------------------------------------------------------- */
  RuntimeValue::~RuntimeValue() {
    this->potentially_destroy_non_trivial_member();
  }

  /* ---------------------------------------------------------------------- */
  void RuntimeValue::potentially_destroy_non_trivial_member() {
    switch (this->value_type) {
    case ValueType::Dictionary: {
      this->variant.dictionary.~Map_t();
      break;
    }
    case ValueType::Matrix: {
      this->variant.matrix_value.Eigen::MatrixXd::~MatrixXd();
      break;
    }
    default:
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  auto RuntimeValue::get_value_type() const -> const ValueType & {
    return this->value_type;
  }

  template void RuntimeValue::add(const std::string &, const Int &);
  template void RuntimeValue::add(const std::string &, const Real &);
  template void RuntimeValue::add(const std::string &,
                                  const Eigen::Ref<const Eigen::MatrixXd> &);
}  // namespace muGrid
