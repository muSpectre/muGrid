/**
 * @file   optional_mapped_field.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Jan 2020
 *
 * @brief  Simple structure for optional mapped fields with low runtime
 * overhead. This is practical if some fields need be be stored depending on
 * some solver arguments. The original use-case is a stress field for storing
 * native stresses, which usually do not need to be stored, but are useful for
 * visualisation and analysis of stress states.
 *
 * Copyright © 2020 Till Junge
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

#ifndef SRC_LIBMUGRID_OPTIONAL_MAPPED_FIELD_HH_
#define SRC_LIBMUGRID_OPTIONAL_MAPPED_FIELD_HH_

namespace muGrid {

  /**
   * @param MappedField needs to be any of the template variants of
   * `muGrid::MappedField`, typically one of its aliases, e.g.,
   * `muGrid::MappedT2Field`
   */
  template <class MappedField>
  class OptionalMappedField {
   public:
    //! Default constructor
    OptionalMappedField() = delete;

    //! constructor
    OptionalMappedField(FieldCollection & collection,
                        const std::string & unique_name,
                        const std::string & sub_division_tag)
        : collection{collection}, unique_name{unique_name},
          sub_division_tag{sub_division_tag} {}

    //! Copy constructor
    OptionalMappedField(const OptionalMappedField & other) = delete;

    //! Move constructor
    OptionalMappedField(OptionalMappedField && other) = delete;

    //! Destructor
    virtual ~OptionalMappedField() = default;

    //! Copy assignment operator
    OptionalMappedField & operator=(const OptionalMappedField & other) = delete;

    //! Move assignment operator
    OptionalMappedField & operator=(OptionalMappedField && other) = delete;

    //! returns whether the field has been created
    bool has_value() const { return this->field_exists; }

    /**
     * returns a reference to the held mapped field. If the field has not yet
     * been created, this call will cause it to be.
     */
    MappedField & get() {
      if (not this->field_exists) {
        this->mapped_field = std::make_unique<MappedField>(
            this->unique_name, this->collection, this->sub_division_tag);
        this->field_exists = true;
      }
      return *this->mapped_field;
    }

   protected:
    bool field_exists{false};

    FieldCollection & collection;
    std::string unique_name;
    std::string sub_division_tag;

    std::unique_ptr<MappedField> mapped_field{nullptr};
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPTIONAL_MAPPED_FIELD_HH_
