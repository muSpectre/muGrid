/**
 * @file   field_map_matrixlike.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  Eigen-Matrix and -Array maps over strongly typed fields
 *
 * Copyright © 2017 Till Junge
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

#ifndef FIELD_MAP_MATRIXLIKE_H
#define FIELD_MAP_MATRIXLIKE_H

#include "common/field_map_base.hh"
#include "common/T4_map_proxy.hh"

#include <Eigen/Dense>

#include <memory>

namespace muSpectre {

  namespace internal {

    /* ---------------------------------------------------------------------- */
    /**
     * lists all matrix-like types consideres by
     * `muSpectre::internal::MatrixLikeFieldMap`
     */
    enum class Map_t{
      Matrix,  //!< for wrapping `Eigen::Matrix`
      Array,   //!< for wrapping `Eigen::Array`
      T4Matrix //!< for wrapping `Eigen::T4Matrix`
    };

    /**
     * traits structure to define the name shown when a
     * `muSpectre::MatrixLikeFieldMap` output into an ostream
     */
    template<Map_t map_type>
    struct NameWrapper {
    };

    /// specialisation for `muSpectre::ArrayFieldMap`
    template<>
    struct NameWrapper<Map_t::Array> {
      //! string to use for printing
      static std::string field_info_root() {return "Array";}
    };

    /// specialisation for `muSpectre::MatrixFieldMap`
    template<>
    struct NameWrapper<Map_t::Matrix> {
      //! string to use for printing
      static std::string field_info_root() {return "Matrix";}
    };

    /// specialisation for `muSpectre::T4MatrixFieldMap`
    template<>
    struct NameWrapper<Map_t::T4Matrix> {
      //! string to use for printing
      static std::string field_info_root() {return "T4Matrix";}
    };

    /* ---------------------------------------------------------------------- */
    /*!
     * base class for maps of matrices, arrays and fourth-order
     * tensors mapped onty matrices
     *
     * It should never be necessary to call directly any of the
     * constructors if this class, but rather use the template aliases
     * `muSpectre::ArrayFieldMap`, `muSpectre::MatrixFieldMap`, and
     * `muSpectre::T4MatrixFieldMap`
     */
    template <class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type, bool ConstField>
    class MatrixLikeFieldMap: public FieldMap<FieldCollection,
                                              typename EigenArray::Scalar,
                                              EigenArray::SizeAtCompileTime,
                                              ConstField>
    {
    public:
      //! base class
      using Parent = FieldMap<FieldCollection,
                              typename EigenArray::Scalar,
                              EigenArray::SizeAtCompileTime, ConstField>;
      using T_t = EigenPlain; //!< plain Eigen type to map
      //! cell coordinates type
      using Ccoord = Ccoord_t<FieldCollection::spatial_dim()>;
      using value_type = EigenArray; //!< stl conformance
      using const_reference = EigenConstArray; //!< stl conformance
      //! stl conformance
      using reference = std::conditional_t<ConstField,
                                           const_reference,
                                           value_type>; // since it's a resource handle
      using size_type = typename Parent::size_type; //!< stl conformance
      using pointer = std::unique_ptr<EigenArray>; //!< stl conformance
      using TypedField = typename Parent::TypedField; //!< stl conformance
      using Field = typename TypedField::Base; //!< stl conformance
      //! stl conformance
      using const_iterator= typename Parent::template iterator<MatrixLikeFieldMap, true>;
      //! stl conformance
      using iterator = std::conditional_t<
        ConstField,
        const_iterator,
        typename Parent::template iterator<MatrixLikeFieldMap>>;
      using reverse_iterator = std::reverse_iterator<iterator>; //!< stl conformance
       //! stl conformance
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;
      //! stl conformance
      friend iterator;

      //! Default constructor
      MatrixLikeFieldMap() = delete;

      /**
       * Constructor using a (non-typed) field. Compatibility is enforced at
       * runtime.  This should not be a performance concern, as this constructor
       * will not be called in anny inner loops (if used as intended).
       */
      template <bool isntConst=!ConstField>
      MatrixLikeFieldMap(std::enable_if_t<isntConst, Field &> field);
      /**
       * Constructor using a (non-typed) field. Compatibility is enforced at
       * runtime.  This should not be a performance concern, as this constructor
       * will not be called in anny inner loops (if used as intended).
       */
      template <bool isConst=ConstField>
      MatrixLikeFieldMap(std::enable_if_t<isConst, const Field &> field);

      /**
       * Constructor using a typed field. Compatibility is enforced
       * statically. It is not always possible to call this constructor, as the
       * type of the field might not be known at compile time.
       */
      template<class FC, typename T2, Dim_t NbC>
      MatrixLikeFieldMap(TypedSizedFieldBase<FC, T2, NbC> & field);

      //! Copy constructor
      MatrixLikeFieldMap(const MatrixLikeFieldMap &other) = default;

      //! Move constructor
      MatrixLikeFieldMap(MatrixLikeFieldMap &&other) = default;

      //! Destructor
      virtual ~MatrixLikeFieldMap()  = default;

      //! Copy assignment operator
      MatrixLikeFieldMap& operator=(const MatrixLikeFieldMap &other) = delete;

      //! Move assignment operator
      MatrixLikeFieldMap& operator=(MatrixLikeFieldMap &&other) = delete;

      //! Assign a matrixlike value to every entry
      template <class Derived>
      inline MatrixLikeFieldMap & operator=(const Eigen::EigenBase<Derived> & val);

      //! give human-readable field map type
      inline std::string info_string() const override final;

      //! member access
      inline reference operator[](size_type index);
      //! member access
      inline reference operator[](const Ccoord& ccoord);

      //! member access
      inline const_reference operator[](size_type index) const;
      //! member access
      inline const_reference operator[](const Ccoord& ccoord) const;

      //! return an iterator to head of field for ranges
      inline iterator begin(){return iterator(*this);}
      //! return an iterator to head of field for ranges
      inline const_iterator cbegin() const {return const_iterator(*this);}
      //! return an iterator to head of field for ranges
      inline const_iterator begin() const {return this->cbegin();}
      //! return an iterator to tail of field for ranges
      inline iterator end(){return iterator(*this, false);};
      //! return an iterator to tail of field for ranges
      inline const_iterator cend() const {return const_iterator(*this, false);}
      //! return an iterator to tail of field for ranges
      inline const_iterator end() const {return this->cend();}

      //! evaluate the average of the field
      inline T_t mean() const;

    protected:
      //! for sad, legacy iterator use
      inline pointer ptr_to_val_t(size_type index);
      const static std::string field_info_root; //!< for printing and debug
    private:
    };

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    const std::string MatrixLikeFieldMap<FieldCollection, EigenArray,
                                         EigenConstArray, EigenPlain, map_type,
                                         ConstField>::
    field_info_root{NameWrapper<map_type>::field_info_root()};

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, class EigenConstArray,
             class EigenPlain, Map_t map_type,  bool ConstField>
    template <bool isntConst>
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    MatrixLikeFieldMap(std::enable_if_t<isntConst, Field &> field)
      :Parent(field) {
      static_assert((isntConst != ConstField),
                    "let the compiler deduce isntConst, this is a SFINAE "
                    "parameter");
      this->check_compatibility();
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    template <bool isConst>
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    MatrixLikeFieldMap(std::enable_if_t<isConst, const Field &> field)
      :Parent(field) {
      static_assert((isConst == ConstField),
                    "let the compiler deduce isntConst, this is a SFINAE "
                    "parameter");
      this->check_compatibility();
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    template<class FC, typename T2, Dim_t NbC>
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField>::
    MatrixLikeFieldMap(TypedSizedFieldBase<FC, T2, NbC> & field)
      :Parent(field) {
    }

    /* ---------------------------------------------------------------------- */
    //! human-readable field map type
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    std::string
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField>::
    info_string() const {
      std::stringstream info;
      info << field_info_root << "("
           << typeid(typename EigenArray::value_type).name() << ", "
           << EigenArray::RowsAtCompileTime << "x"
           << EigenArray::ColsAtCompileTime << ")";
      return info.str();
    }

    /* ---------------------------------------------------------------------- */
    //! member access
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::reference
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField>::
    operator[](size_type index) {
      return reference(this->get_ptr_to_entry(index));
    }

    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::reference
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    operator[](const Ccoord & ccoord) {
      size_t index{};
      index = this->collection.get_index(ccoord);
      return reference(this->get_ptr_to_entry(std::move(index)));
    }

    /* ---------------------------------------------------------------------- */
    //! member access
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::const_reference
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    operator[](size_type index) const {
      return const_reference(this->get_ptr_to_entry(index));
    }

    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::const_reference
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    operator[](const Ccoord & ccoord) const{
      size_t index{};
      index = this->collection.get_index(ccoord);
      return const_reference(this->get_ptr_to_entry(std::move(index)));
    }

    //----------------------------------------------------------------------------//
    template <class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    template <class Derived>
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField> &
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField>::
    operator=(const Eigen::EigenBase<Derived> & val) {
      for (auto && entry: *this) {
        entry = val;
      }
      return *this;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type, bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::T_t
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                       EigenPlain, map_type, ConstField>::
    mean() const {
      T_t mean{T_t::Zero()};
      for (auto && val: *this) {
        mean += val;
      }
      mean *= 1./Real(this->size());
      return mean;
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, class EigenArray, class EigenConstArray,
              class EigenPlain, Map_t map_type,  bool ConstField>
    typename MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray,
                                EigenPlain, map_type, ConstField>::pointer
    MatrixLikeFieldMap<FieldCollection, EigenArray, EigenConstArray, EigenPlain,
                       map_type, ConstField>::
    ptr_to_val_t(size_type index) {
      return std::make_unique<value_type>
        (this->get_ptr_to_entry(std::move(index)));
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen matrix map as iterate
  template <class FieldCollection, typename T, Dim_t NbRows, Dim_t NbCols,
            bool ConstField=false>
  using MatrixFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     std::conditional_t<ConstField,
                        Eigen::Map<const Eigen::Matrix<T, NbRows, NbCols>>,
                        Eigen::Map<Eigen::Matrix<T, NbRows, NbCols>>>,
     Eigen::Map<const Eigen::Matrix<T, NbRows, NbCols>>,
     Eigen::Matrix<T, NbRows, NbCols>,
     internal::Map_t::Matrix, ConstField>;

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen matrix map as iterate
  template <class FieldCollection, typename T, Dim_t Dim,
            bool MapConst=false>
  using T4MatrixFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     T4MatMap<T, Dim, MapConst>,
     T4MatMap<T, Dim, true>,
     T4Mat<T, Dim>,
     internal::Map_t::T4Matrix, MapConst>;

  /* ---------------------------------------------------------------------- */
  //! short-hand for an Eigen array map as iterate
  template <class FieldCollection, typename T, Dim_t NbRows, Dim_t NbCols,
            bool ConstField=false>
  using ArrayFieldMap = internal::MatrixLikeFieldMap
    <FieldCollection,
     std::conditional_t<ConstField,
                        Eigen::Map<const Eigen::Array<T, NbRows, NbCols>>,
                        Eigen::Map<Eigen::Array<T, NbRows, NbCols>>>,
     Eigen::Map<const Eigen::Array<T, NbRows, NbCols>>,
     Eigen::Array<T, NbRows, NbCols>,
     internal::Map_t::Array, ConstField>;


}  // muSpectre

#endif /* FIELD_MAP_MATRIXLIKE_H */
