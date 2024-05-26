/**
 * @file   grid_common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Small definitions of commonly used types throughout µgrid
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

#include "exception.hh"
#include "iterators.hh"

#include "Eigen/Dense"

#include <array>
#include <cmath>
#include <complex>
#include <type_traits>
#include <initializer_list>
#include <algorithm>
#include <vector>
#include <cstdint>

#ifndef SRC_LIBMUGRID_GRID_COMMON_HH_
#define SRC_LIBMUGRID_GRID_COMMON_HH_

namespace muGrid {

  /**
   * \defgroup Scalars Scalar types
   * @{
   */

  /**
   * @typedef Dim_t
   * @brief A type alias for signed integers used for static dimensions.
   *
   * This type alias is used to represent signed integers for static dimensions
   * in the µGrid codebase. It is used for consistency throughout the code. It
   * is also capable of representing -1, which is a requirement for Eigen.
   */
  using Dim_t = int;

  /**
   * @typedef Index_t
   * @brief A type alias for Eigen::Index used for size-related values.
   *
   * This type alias is used to represent size-related values in the µGrid
   * codebase. It is used for consistency with Eigen and to handle large arrays
   * that have more indices than can be counted with Dim_t. For example, arrays
   * with dimensions 65536 × 65536 would overflow Dim_t, so Index_t is used
   * instead.
   */
  using Index_t = Eigen::Index;

  using Uint = unsigned int;  //!< type to use in math for unsigned integers
  using Int = int;            //!< type to use in math for signed integers
  using Real = double;        //!< type to use in math for real numbers
  using Complex =
      std::complex<Real>;  //!< type to use in math for complex numbers

  /**@}*/

  constexpr Index_t oneD{1};    //!< constant for a one-dimensional problem
  constexpr Index_t twoD{2};    //!< constant for a two-dimensional problem
  constexpr Index_t threeD{3};  //!< constant for a three-dimensional problem
  constexpr Index_t zerothOrder{0};   //!< constant for scalars
  constexpr Index_t firstOrder{1};    //!< constant for vectors
  constexpr Index_t secondOrder{2};   //!< constant second-order tensors
  constexpr Index_t fourthOrder{4};   //!< constant fourth-order tensors
  constexpr Index_t OneQuadPt{1};     //!< constant for 1 quadrature point/pixel
  constexpr Index_t TwoQuadPts{2};    //!< constant for 2 quadrature point/pixel
  constexpr Index_t FourQuadPts{4};   //!< constant for 4 quadrature point/pixel
  constexpr Index_t FiveQuadPts{5};   //!< constant for 5 quadrature point/pixel
  constexpr Index_t SixQuadPts{6};    //!< constant for 6 quadrature point/pixel
  constexpr Index_t EightQuadPts{8};  //!< constant for 8 quadrature point/pixel
  constexpr Index_t OneNode{1};       //!< constant for 1 node per pixel

  /**
   * @enum IterUnit
   * @brief An enumeration class for iteration units.
   *
   * This enumeration class is used in two contexts within the µGrid codebase.
   * Firstly, it is used in `Field`s to specify the relative storage of data
   * with respect to pixels, quadrature points, or nodal points.
   * Secondly, it is used in `FieldMap`s to specify the unit of iteration,
   * whether it be over pixels, quadrature points, or nodal points.
   *
   * @var Pixel Represents degrees of freedom (dofs) relative to a pixel/voxel,
   * with no subdivision.
   * @var SubPt Represents dofs relative to sub-points (e.g. quadrature points).
   */
  enum class IterUnit {
    Pixel,  //!< dofs relative to a pixel/voxel, no subdivision
    SubPt   //!< dofs relative to sub-points (e.g. quadrature points)
  };

  /**
   * @enum StorageOrder
   * @brief An enumeration class for storage orders of field components.
   *
   * This enumeration class defines six types of storage orders: ColMajor,
   * ArrayOfStructures, RowMajor, StructurOfArrays, Unknown, and Automatic.
   * These storage orders can be used to determine the order in which field
   * components are stored in memory.
   *
   * @var ColMajor Represents a column-major storage order. In this order, the
   * first index changes fastest, and the last index changes slowest.
   * @var ArrayOfStructures Represents an array of structures storage order. In
   * this order, components are consecutive in memory. It is equivalent to
   * ColMajor.
   * @var RowMajor Represents a row-major storage order. In this order, the last
   * index changes fastest, and the first index changes slowest.
   * @var StructurOfArrays Represents a structure of arrays storage order. In
   * this order, pixels are consecutive in memory. It is equivalent to RowMajor.
   * @var Unknown Represents an unknown storage order. It is used only for
   * `WrappedField`.
   * @var Automatic Represents an automatic storage order. In this order, the
   * storage order is inherited from `FieldCollection`.
   */
  enum class StorageOrder {
    ColMajor,  //!< column-major storage order (first index is fast)
    ArrayOfStructures = ColMajor,  //!< components are consecutive in memory
    RowMajor,  //!< row-major storage order (last index is fast)
    StructurOfArrays = RowMajor,  //< pixels are consecutive in memory
    Unknown,   //!< storage order is unknown, only for `WrappedField`
    Automatic  //!< inherit storage order from `FieldCollection`
  };

  //! Type used for shapes and strides
  using Shape_t = std::vector<Index_t>;

  /**
   * this tag is always defined to one in every field collection
   */
  const std::string PixelTag{"pixel"};

  //! inserts `muGrid::IterUnit` into `std::ostream`s
  std::ostream & operator<<(std::ostream & os, const IterUnit & sub_division);

  //! inserts `muGrid::StorageOrder` into `std::ostream`s
  std::ostream & operator<<(std::ostream & os,
                            const StorageOrder & storage_order);

  /**
   * @enum Mapping
   * @brief An enumeration class for mapping types.
   *
   * This enumeration class defines two types of mappings: Const and Mut. These
   * mappings can be used to determine the type of access (constant or mutable)
   * to the mapped field through their iterators or access operators.
   *
   * @var Const Represents a constant mapping. It is used when the mapped field
   * should not be modified.
   * @var Mut Represents a mutable mapping. It is used when the mapped field can
   * be modified.
   */
  enum class Mapping { Const, Mut };

  /**
   * \defgroup Coordinates Coordinate types
   * @{
   */

  /**
   * @typedef Ccoord_t
   * @brief A type alias for cell coordinates.
   *
   * This type alias represents cell coordinates, which are up to three integer
   * numbers with a fixed dimension. The dimension is determined by the template
   * parameter Dim. The coordinates are stored in a std::array of type Index_t.
   *
   * @tparam Dim The dimension of the cell coordinates. It should be between 1
   * and 3.
   */
  template <size_t Dim>
  using Ccoord_t = std::array<Index_t, Dim>;
  /**
   * @typedef Rcoord_t
   * @brief A type alias for real space coordinates.
   *
   * This type alias represents real space coordinates, which are up to three
   * floating point numbers with a fixed dimension. The dimension is determined
   * by the template parameter Dim. The coordinates are stored in a std::array
   * of type Real.
   *
   * @tparam Dim The dimension of the real space coordinates. It should be
   * between 1 and 3.
   */
  template <size_t Dim>
  using Rcoord_t = std::array<Real, Dim>;

  /**@}*/

  /**
   * Class to represent integer (cell-) coordinates or real-valued coordinates.
   * This class can dynamically accept any spatial-dimension between 1 and
   * MaxDim, and DynCcoord references can be cast to `muGrid::Ccoord_t &` or
   * `muGrid::Rcoord_t &` references. These are used when templating with the
   * spatial dimension of the problem is undesireable/impossible.
   */
  template <size_t MaxDim, typename T = Index_t>
  class DynCcoord {
    template <size_t Dim, size_t... Indices>
    constexpr static std::array<T, MaxDim>
    fill_front_helper(const std::array<T, Dim> & ccoord,
                      std::index_sequence<Indices...>) {
      return std::array<T, MaxDim>{ccoord[Indices]...};
    }

    template <size_t Dim>
    constexpr std::array<T, MaxDim>
    fill_front(const std::array<T, Dim> & ccoord) {
      static_assert(Dim <= MaxDim, "Coord has more than MaxDim dimensions.");
      return fill_front_helper(ccoord, std::make_index_sequence<Dim>{});
    }

   public:
    /**
     * @typedef iterator
     * @brief A type alias for an iterator over the elements of a std::array.
     *
     * This type alias is used to create an iterator that can traverse the
     * elements of a std::array. The std::array is templated on type T and has a
     * maximum size of MaxDim. The iterator can be used to access and modify the
     * elements of the std::array.
     */
    using iterator = typename std::array<T, MaxDim>::iterator;

    /**
     * @typedef const_iterator
     * @brief A type alias for a constant iterator over the elements of a
     * std::array.
     *
     * This type alias is used to create a constant iterator that can traverse
     * the elements of a std::array. The std::array is templated on type T and
     * has a maximum size of MaxDim. The constant iterator can be used to access
     * the elements of the std::array, but cannot modify them.
     */
    using const_iterator = typename std::array<T, MaxDim>::const_iterator;

    //! default constructor
    DynCcoord() : dim{}, long_array{} {};

    /**
     * @brief Constructs a DynCcoord object from an initializer list.
     *
     * This constructor creates a DynCcoord object using an initializer list.
     * The length of the initializer list determines the spatial dimension of
     * the coordinate. The initializer list must have a length between 1 and
     * MaxDim.
     *
     * @param init_list Initializer list used to set the values of the DynCcoord
     * object. The length of the list becomes the spatial dimension of the
     * coordinate.
     * @throws RuntimeError If the length of the initializer list is greater
     * than MaxDim, a RuntimeError is thrown with a message indicating the
     * maximum dimension and the provided dimension.
     */
    DynCcoord(std::initializer_list<T> init_list)
        : dim(init_list.size()), long_array{} {
      if (this->dim > Dim_t(MaxDim)) {
        std::stringstream error{};
        error << "The maximum dimension representable by this dynamic array is "
              << MaxDim << ". You supplied an initialiser list with "
              << init_list.size() << " entries.";
        throw RuntimeError(error.str());
      }
      std::copy(init_list.begin(), init_list.end(), this->long_array.begin());
    }

    /**
     * @brief Constructs a DynCcoord object with a specified dimension.
     *
     * This constructor creates a DynCcoord object with a specified dimension.
     * The dimension must be between 1 and MaxDim. Note: This constructor
     * requires regular (round) braces '()'. Using curly braces '{}' will result
     * in the initializer list constructor being called and creating a DynCcoord
     * with spatial dimension 1.
     *
     * @param dim The spatial dimension of the DynCcoord object. It needs to be
     * between 1 and MaxDim.
     */
    explicit DynCcoord(Dim_t dim) : dim{dim}, long_array{} {}
    //! Constructor from a statically sized coord
    template <size_t Dim>
    explicit DynCcoord(const std::array<T, Dim> & ccoord)
        : dim{Dim}, long_array{fill_front(ccoord)} {
      static_assert(Dim <= MaxDim,
                    "Assigned Ccoord has more than MaxDim dimensions.");
    }

    /**
     * @brief Constructs a DynCcoord object from a std::vector.
     *
     * This constructor creates a DynCcoord object using a std::vector. The size
     * of the std::vector determines the spatial dimension of the coordinate.
     * The std::vector must have a size between 1 and MaxDim.
     *
     * @param ccoord std::vector used to set the values of the DynCcoord object.
     * The size of the vector becomes the spatial dimension of the coordinate.
     * @throws RuntimeError If the size of the std::vector is greater than
     * MaxDim, a RuntimeError is thrown with a message indicating the maximum
     * dimension and the provided dimension.
     */
    explicit DynCcoord(const std::vector<T> & ccoord)
        : dim{Dim_t(ccoord.size())}, long_array{} {
      if (this->dim > Dim_t(MaxDim)) {
        std::stringstream error{};
        error << "The maximum dimension representable by this dynamic array is "
              << MaxDim << ". You supplied a vector with " << ccoord.size()
              << " entries.";
        throw RuntimeError(error.str());
      }
      std::copy(ccoord.begin(), ccoord.end(), this->long_array.begin());
    }

    //! Copy constructor
    DynCcoord(const DynCcoord & other) = default;

    //! Move constructor
    DynCcoord(DynCcoord && other) = default;

    //! nonvirtual Destructor
    ~DynCcoord() = default;

    //! Assign arrays
    template <size_t Dim>
    DynCcoord & operator=(const std::array<T, Dim> & ccoord) {
      static_assert(Dim <= MaxDim,
                    "Assigned Ccoord has more than MaxDim dimensions.");
      this->dim = Dim;
      std::copy(ccoord.begin(), ccoord.end(), this->long_array.begin());
      return *this;
    }

    //! Copy assignment operator
    DynCcoord & operator=(const DynCcoord & other) = default;

    //! Move assignment operator
    DynCcoord & operator=(DynCcoord && other) = default;

    //! comparison operator
    template <size_t Dim2>
    bool operator==(const std::array<T, Dim2> & other) const {
      return ((this->get_dim() == Dim2) and
              (std::array<T, Dim2>(*this) == other));
    }

    //! comparison operator
    bool operator==(const DynCcoord & other) const {
      bool retval{this->get_dim() == other.get_dim()};
      for (int i{0}; i < this->get_dim(); ++i) {
        retval &= this->long_array[i] == other[i];
      }
      return retval;
    }

    //! element-wise division
    template <typename T2>
    DynCcoord<MaxDim, decltype(T{} / T2{})>
    operator/(const DynCcoord<MaxDim, T2> & other) const {
      if (this->get_dim() != other.get_dim()) {
        std::stringstream error{};
        error << "you are trying to divide a " << this->get_dim()
              << "-dimensional coord by a " << other.get_dim()
              << "-dimensional coord element-wise.";
        throw RuntimeError(error.str());
      }
      DynCcoord<MaxDim, decltype(T{} / T2{})> retval(this->get_dim());
      for (Dim_t i{0}; i < this->get_dim(); ++i) {
        retval[i] = this->operator[](i) / other[i];
      }
      return retval;
    }

    //! access operator
    T & operator[](const size_t & index) { return this->long_array[index]; }

    //! access operator
    const T & operator[](const size_t & index) const {
      return this->long_array[index];
    }

    //! push element to the end
    void push_back(const T & value) {
      if (static_cast<size_t>(this->dim) >= MaxDim) {
        throw RuntimeError("Dimension bounds exceeded");
      }
      this->long_array[this->dim] = value;
      this->dim++;
    }

    //! modulo assignment operator (mostly for periodic boundaries stuff)
    DynCcoord & operator%=(const DynCcoord & other) {
      for (auto && tup : akantu::zip(*this, other)) {
        std::get<0>(tup) %= std::get<1>(tup);
      }
      return *this;
    }

    //! modulo operator (mostly for periodic boundaries stuff)
    DynCcoord operator%(const DynCcoord & other) const {
      DynCcoord ret_val{*this};
      ret_val %= other;
      return ret_val;
    }

    //! conversion operator
    template <size_t Dim>
    operator std::array<T, Dim>() const {
      return this->template get<Dim>();
    }

    //! cast to a reference to a statically sized array
    template <Dim_t Dim>
    std::array<T, Dim> & get() {
      static_assert(Dim <= MaxDim,
                    "Requested Ccoord has more than MaxDim dimensions.");
      char * intermediate{reinterpret_cast<char *>(&this->long_array)};
      return reinterpret_cast<std::array<T, Dim> &>(*intermediate);
    }

    //! cast to a const reference to a statically sized array
    template <Dim_t Dim>
    const std::array<T, Dim> & get() const {
      static_assert(Dim <= MaxDim,
                    "Requested Ccoord has more than MaxDim dimensions.");
      const char * intermediate{
          reinterpret_cast<const char *>(&this->long_array)};
      return reinterpret_cast<const std::array<T, Dim> &>(*intermediate);
    }

    //! return the spatial dimension of this coordinate
    const Dim_t & get_dim() const { return this->dim; }

    //! return the spatial dimension of this coordinate, STL compatibility
    const Dim_t & size() const { return this->dim; }

    //! convert into a vector
    explicit operator std::vector<T>() const {
      std::vector<T> v;
      for (auto && el : *this) {
        v.push_back(el);
      }
      return v;
    }

    //! iterator to the first entry for iterating over only the valid entries
    iterator begin() { return this->long_array.begin(); }
    //! iterator past the dim-th entry for iterating over only the valid entries
    iterator end() { return this->long_array.begin() + this->dim; }
    //! const iterator to the first entry for iterating over only the valid
    //! entries
    const_iterator begin() const { return this->long_array.begin(); }
    //! const iterator past the dim-th entry for iterating over only the valid
    //! entries
    const_iterator end() const { return this->long_array.begin() + this->dim; }

    //! return the underlying data pointer
    T * data() { return this->long_array.data(); }
    //! return the underlying data pointer
    const T * data() const { return this->long_array.data(); }

    //! return a reference to the last valid entry
    T & back() { return this->long_array[this->dim - 1]; }
    //! return a const reference to the last valid entry
    const T & back() const { return this->long_array[this->dim - 1]; }

   protected:
    //! spatial dimension of the coordinate
    Dim_t dim;
    //! storage for coordinate components
    std::array<T, MaxDim> long_array;
  };

  /**
   * @brief Overloads the addition operator for two DynCcoord objects.
   *
   * This function overloads the addition operator to perform element-wise
   * addition of two DynCcoord objects. The DynCcoord objects must have the same
   * dimension, otherwise a RuntimeError is thrown. The result is a new
   * DynCcoord object with the same dimension as the input objects, where each
   * element is the sum of the corresponding elements in the input objects.
   *
   * @tparam MaxDim The maximum dimension of the DynCcoord objects.
   * @tparam T The type of the elements in the DynCcoord objects.
   * @param A The first DynCcoord object.
   * @param B The second DynCcoord object.
   * @return A new DynCcoord object that is the result of the element-wise
   * addition of A and B.
   * @throws RuntimeError If the dimensions of A and B do not match.
   */
  template <size_t MaxDim, typename T>
  DynCcoord<MaxDim, T> operator+(const DynCcoord<MaxDim, T> & A,
                                 const DynCcoord<MaxDim, T> & B) {
    if (A.get_dim() != B.get_dim()) {
      throw RuntimeError("Dimension mismatch");
    }
    // this needs to be parens, *not* curly braces
    DynCcoord<MaxDim, T> result(A.get_dim());
    assert(result.get_dim() == A.get_dim());
    for (Dim_t dim{0}; dim < A.get_dim(); ++dim) {
      result[dim] = A[dim] + B[dim];
    }
    return result;
  }

  /**
   * @brief Overloads the subtraction operator for two DynCcoord objects.
   *
   * This function overloads the subtraction operator to perform element-wise
   * subtraction of two DynCcoord objects. The DynCcoord objects must have the
   * same dimension, otherwise a RuntimeError is thrown. The result is a new
   * DynCcoord object with the same dimension as the input objects, where each
   * element is the difference of the corresponding elements in the input
   * objects.
   *
   * @tparam MaxDim The maximum dimension of the DynCcoord objects.
   * @tparam T The type of the elements in the DynCcoord objects.
   * @param A The first DynCcoord object.
   * @param B The second DynCcoord object.
   * @return A new DynCcoord object that is the result of the element-wise
   * subtraction of A and B.
   * @throws RuntimeError If the dimensions of A and B do not match.
   */
  template <size_t MaxDim, typename T>
  DynCcoord<MaxDim, T> operator-(const DynCcoord<MaxDim, T> & A,
                                 const DynCcoord<MaxDim, T> & B) {
    if (A.get_dim() != B.get_dim()) {
      throw RuntimeError("Dimension mismatch");
    }
    // this needs to be parens, *not* curly braces
    DynCcoord<MaxDim, T> result(A.get_dim());
    assert(result.get_dim() == A.get_dim());
    for (Dim_t i{0}; i < A.get_dim(); ++i) {
      result[i] = A[i] - B[i];
    }
    return result;
  }

  /**
   * Cell coordinates, i.e. up to three integer numbers with dynamic (determined
   * during runtime) dimension
   */
  using DynCcoord_t = DynCcoord<threeD>;

  /**
   * Real space coordinates, i.e. up to three floating point numbers with
   * dynamic (determined during runtime) dimension
   */
  using DynRcoord_t = DynCcoord<threeD, Real>;

  /**
   * Enum class for verbose-flag
   */
  enum class Verbosity { Silent = 0, Some = 1, Detailed = 2, Full = 3 };

  /**
   * comparison operators for Verbosity-class
   */
  bool operator<(const Verbosity v1, const Verbosity v2);
  bool operator>(const Verbosity v1, const Verbosity v2);
  bool operator<=(const Verbosity v1, const Verbosity v2);
  bool operator>=(const Verbosity v1, const Verbosity v2);

  /**
   * return a Eigen representation of the data stored in a std::array (e.g., for
   * doing vector operations on a coordinate)
   */
  template <typename T, size_t Dim>
  Eigen::Map<Eigen::Matrix<T, Dim, 1>> eigen(std::array<T, Dim> & coord) {
    return Eigen::Map<Eigen::Matrix<T, Dim, 1>>{coord.data()};
  }

  /**
   * return a constant  Eigen representation of the data stored in a std::array
   * (e.g., for doing vector operations on a coordinate)
   */
  template <typename T, size_t Dim>
  Eigen::Map<const Eigen::Matrix<T, Dim, 1>>
  eigen(const std::array<T, Dim> & coord) {
    return Eigen::Map<const Eigen::Matrix<T, Dim, 1>>{coord.data()};
  }

  /**
   * return a Eigen representation of the data stored in a std::array (e.g., for
   * doing vector operations on a coordinate)
   */
  template <typename T, size_t MaxDim>
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>
  eigen(DynCcoord<MaxDim, T> & coord) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>{coord.data(),
                                                           coord.get_dim()};
  }

  /**
   * return a const Eigen representation of the data stored in a std::array
   * (e.g., for doing vector operations on a coordinate)
   */
  template <typename T, size_t MaxDim>
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
  eigen(const DynCcoord<MaxDim, T> & coord) {
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>{
        coord.data(), coord.get_dim()};
  }
  //@}

  /**
   * Allows inserting `std::vector` into `std::ostream`s
   */
  template <typename T>
  std::ostream & operator<<(std::ostream & os, const std::vector<T> & values) {
    os << "(";
    if (values.size() > 0) {
      for (size_t i = 0; i < values.size() - 1; ++i) {
        os << values[i] << ", ";
      }
      os << values.back();
    }
    os << ")";
    return os;
  }

  /**
   * Allows inserting `muGrid::Ccoord_t` and `muGrid::Rcoord_t`
   * into `std::ostream`s
   */
  template <typename T, size_t dim>
  std::ostream & operator<<(std::ostream & os,
                            const std::array<T, dim> & values) {
    os << "(";
    for (size_t i = 0; i < dim - 1; ++i) {
      os << values[i] << ", ";
    }
    os << values.back() << ")";
    return os;
  }

  /**
   * Allows inserting `muGrid::DynCcoord` into `std::ostream`s
   */
  template <size_t MaxDim, typename T>
  std::ostream & operator<<(std::ostream & os,
                            const DynCcoord<MaxDim, T> & values) {
    os << "(";
    if (values.get_dim() > 0) {
      for (Dim_t i = 0; i < values.get_dim() - 1; ++i) {
        os << values[i] << ", ";
      }
      os << values.back();
    }
    os << ")";
    return os;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Rcoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! element-wise division
  template <size_t dim>
  Rcoord_t<dim> operator/(const Rcoord_t<dim> & a, const Ccoord_t<dim> & b) {
    Rcoord_t<dim> retval{a};
    for (size_t i = 0; i < dim; ++i) {
      retval[i] /= b[i];
    }
    return retval;
  }

  //! convenience definitions
  constexpr Real pi{3.1415926535897932384626433};
  //! constant used to explicitly denote unknown positive integers
  constexpr static Dim_t Unknown{-1};

  //! compile-time potentiation required for field-size computations
  template <typename R, typename I>
  constexpr R ipow(R base, I exponent) {
    static_assert(std::is_integral<I>::value, "Type must be integer");
    R retval{1};
    for (I i = 0; i < exponent; ++i) {
      retval *= base;
    }
    return retval;
  }

  namespace version {

    /**
     * @brief Returns a formatted text that can be printed to stdout or to
     * output files.
     *
     * This function generates a string that contains the git commit hash and
     * repository url used to compile µGrid. It also indicates whether the
     * current state was dirty or not.
     *
     * @return A formatted string containing the git commit hash, repository url
     * and the state of the repository.
     */
    std::string info();

    /**
     * @brief Returns the git commit hash.
     *
     * This function retrieves the git commit hash used to compile µGrid.
     *
     * @return A constant character pointer representing the git commit hash.
     */
    const char * hash();

    /**
     * @brief Returns the repository description.
     *
     * This function retrieves the repository description used to compile µGrid.
     *
     * @return A constant character pointer representing the repository
     * description.
     */
    const char * description();

    /**
     * @brief Checks if the current state was dirty.
     *
     * This function checks if the current state of the repository used to
     * compile µGrid was dirty or not.
     *
     * @return A boolean value indicating if the state was dirty (true) or not
     * (false).
     */
    bool is_dirty();
  }  // namespace version

}  // namespace muGrid

#include "cpp_compliance.hh"

#endif  // SRC_LIBMUGRID_GRID_COMMON_HH_
