/**
 * @file   options_dictionary.hh
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

#include "grid_common.hh"

#include <Eigen/Dense>

#include <map>
#include <string>
#include <memory>

#ifndef SRC_LIBMUGRID_OPTIONS_DICTIONARY_HH_
#define SRC_LIBMUGRID_OPTIONS_DICTIONARY_HH_

namespace muGrid {
  /* ---------------------------------------------------------------------- */
  class DictionaryError : public muGrid::RuntimeError {
    using muGrid::RuntimeError::RuntimeError;
  };

  /* ---------------------------------------------------------------------- */
  class ValueError : public DictionaryError {
    using DictionaryError::DictionaryError;
  };

  /* ---------------------------------------------------------------------- */
  class KeyError : public DictionaryError {
    using DictionaryError::DictionaryError;
  };

  using Eigen::MatrixXd;

  /**
   * this class holds the dictionary tree structure and protects acccess to the
   * union type holding the actual data
   */
  class RuntimeValue final {
   public:
    /**
     * Currently, the only types we can hold in an options dictionary are
     * integers, real numbers, matrices of real numbers and nested dictionaries
     * to which these restrictions also apply
     */
    enum class ValueType { Dictionary, Int, Real, Matrix };
    using Map_t = std::map<std::string, std::shared_ptr<RuntimeValue>>;

    //! constructors from values
    explicit RuntimeValue(const Int & value);
    explicit RuntimeValue(const Real & value);
    explicit RuntimeValue(const Eigen::Ref<const Eigen::MatrixXd> & value);
    explicit RuntimeValue(const RuntimeValue & value);
    explicit RuntimeValue(const Map_t & value);

    //! assignment operators
    RuntimeValue & operator=(const Int & value);
    RuntimeValue & operator=(const Real & value);
    RuntimeValue & operator=(const Eigen::Ref<const Eigen::MatrixXd> & value);
    RuntimeValue & operator=(const Map_t & value);
    RuntimeValue & operator=(const RuntimeValue & other);

    /**
     * add a new dictionary entry. throws a ValueError if this RuntimeValue
     * isn't of ^ValueType::Dictionary`
     */
    template <typename T>
    void add(const std::string & key, T value);
    void add(const std::string & key, std::shared_ptr<RuntimeValue> other);

    //! safely recover a typed value, throws ValueError if types mismatch
    const Int & get_int() const;
    const Real & get_real() const;
    const Eigen::MatrixXd & get_matrix() const;
    std::shared_ptr<RuntimeValue> & get_value(const std::string & key);

    ~RuntimeValue();
    void potentially_destroy_non_trivial_member();

    const ValueType & get_value_type() const;

   protected:
    ValueType value_type;
    union Variant {
      Variant(const Int & value) : integer_value{value} {}
      Variant(const Real & value) : real_value{value} {}
      Variant(const Eigen::Ref<const Eigen::MatrixXd> & value) {
        new (&this->matrix_value) Eigen::MatrixXd(value);
      }
      Variant(const Map_t & value) : dictionary{value} {}
      //! doesn't do anything, responsibility of ValueHolder
      ~Variant() {}
      Map_t dictionary{};
      Int integer_value;
      Real real_value;
      Eigen::MatrixXd matrix_value;
    };
    Variant variant;
  };

  /**
   * The Dictionary class holds a smart pointer to a RuntimeValue and provides
   * the interface to assign, modify and get typed values out of thhat
   * RuntimeValue. It behaves like a subset of the python dict class
   */
  class Dictionary {
    explicit Dictionary(std::shared_ptr<RuntimeValue> ptr);

   public:
    //! default constructor
    Dictionary();
    //! move constructor
    Dictionary(Dictionary &&) = default;
    Dictionary(const Dictionary & other) = default;

    //! constructor with a single key-value pair
    Dictionary(const std::string & key, const Real & value);
    Dictionary(const std::string & key, const Int & value);
    Dictionary(const std::string & key,
               const Eigen::Ref<const Eigen::MatrixXd> & value);
    ~Dictionary() = default;

    //! copy operator
    Dictionary & operator=(const Dictionary & other) = default;

    //! assignment to a single runtime value (rather than a dict)
    Dictionary & operator=(const Int & value);
    Dictionary & operator=(const Real & value);
    Dictionary & operator=(const Eigen::Ref<const Eigen::MatrixXd> & value);

    //! get a typed value, throws DictionaryError if the type is mismatched
    const Int & get_int() const;
    const Real & get_real() const;
    const Eigen::MatrixXd & get_matrix() const;

    /**
     * index operator returns a modifiable sub-dictionary (modifications
     * propagate back into the original dictionary)
     */
    Dictionary operator[](const std::string & name) const;

    //! add a value to the dictionary
    void add(const std::string & key, const Dictionary & other);
    void add(const std::string & key, const Int & value);
    void add(const std::string & key, const Real & value);
    void add(const std::string & key,
             const Eigen::Ref<const Eigen::MatrixXd> & value);
    const RuntimeValue::ValueType & get_value_type() const;

   protected:
    std::shared_ptr<RuntimeValue> ptr;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPTIONS_DICTIONARY_HH_
