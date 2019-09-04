/**
 * @file   mapped_nfield.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Sep 2019
 *
 * @brief  convenience class to deal with data structures common to most
 *         internal variable fields in materials
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

#ifndef SRC_LIBMUGRID_MAPPED_NFIELD_HH_
#define SRC_LIBMUGRID_MAPPED_NFIELD_HH_

#include "nfield_map_static.hh"
#include "nfield_collection.hh"
#include "nfield_typed.hh"

#include <string>

namespace muGrid {

  /**
   * MappedNFields are a combination of a field and an associated map, and as
   * such it does not introduce any new functionality that Fields and FieldMaps
   * do not already possess. They provide a convenience structure for the
   * default use case of internal variables, which are typically used only by a
   * single material and always the same way.
   */
  template <class FieldMapType>
  class MappedField {
   public:
    using Scalar_t = typename FieldMapType::Scalar_t;
    using Return_t =
        typename FieldMapType::template Return_t<FieldMapType::IsConstField()>;
    using iterator = typename FieldMapType::iterator;
    using const_iterator = typename FieldMapType::const_iterator;
    //! Default constructor
    MappedField() = delete;

    MappedField(const std::string & unique_name, NFieldCollection & collection)
        : nb_components{compute_nb_components(unique_name, collection)},
          field(collection.register_field<TypedNField<Scalar_t>>(
              unique_name, this->nb_components)),
          map{this->field} {}

    //! Copy constructor
    MappedField(const MappedField & other) = delete;

    //! Move constructor
    MappedField(MappedField && other) = default;

    //! Destructor
    virtual ~MappedField() = default;

    //! Copy assignment operator
    MappedField & operator=(const MappedField & other) = delete;

    //! Move assignment operator
    MappedField & operator=(MappedField && other) = default;

    Return_t operator[](size_t index) { return this->map[index]; }

    iterator begin() { return this->map.begin(); }
    iterator end() { return this->map.end(); }
    const_iterator begin() const { return this->map.begin(); }
    const_iterator end() const { return this->map.end(); }

    TypedNField<Scalar_t> & get_field() { return this->field; }
    FieldMapType & get_map() { return this->map; }

   protected:
    static Dim_t compute_nb_components(const std::string & unique_name,
                                       NFieldCollection & collection) {
      switch (FieldMapType::GetIterationType()) {
      case Iteration::QuadPt: {
        return FieldMapType::Stride();
        break;
      }
      case Iteration::Pixel: {
        if (not collection.has_nb_quad()) {
          throw NFieldMapError("Can't create a pixel map for field '" +
                               unique_name +
                               "' before the number of quadrature points has "
                               "been set for the field collection.");
        }
        return FieldMapType::Stride()/collection.get_nb_quad();
        break;
      }
      default:
        throw NFieldMapError("unknown iteration type");
        break;
      }
    }

    Dim_t nb_components;
    TypedNField<Scalar_t> & field;
    FieldMapType map;
  };

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using MappedMatrixNField =
      MappedField<MatrixNFieldMap<T, ConstField, NbRow, NbCol, IterationType>>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using MappedArrayNField =
      MappedField<ArrayNFieldMap<T, ConstField, NbRow, NbCol, IterationType>>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField>
  using MappedScalarNField = MappedField<ScalarNFieldMap<T, ConstField>>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t Dim>
  using MappedT2NField = MappedField<T2NFieldMap<T, ConstField, Dim>>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t Dim>
  using MappedT4NField = MappedField<T4NFieldMap<T, ConstField, Dim>>;

}  // namespace muGrid

#endif /* SRC_LIBMUGRID_MAPPED_NFIELD_HH_ */
