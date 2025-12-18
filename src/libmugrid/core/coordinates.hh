/**
 * @file   core/coordinates.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Coordinate type definitions for muGrid
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

#ifndef SRC_LIBMUGRID_CORE_COORDINATES_HH_
#define SRC_LIBMUGRID_CORE_COORDINATES_HH_

#include "types.hh"
#include "../exception.hh"
#include "../iterators.hh"

#include "Eigen/Dense"

#include <array>
#include <algorithm>
#include <sstream>
#include <vector>

namespace muGrid {

    /**
     * \defgroup Coordinates Coordinate types
     * @{
     */

    /**
     * @typedef Ccoord_t
     * @brief A type alias for cell coordinates.
     *
     * This type alias represents cell coordinates, which are up to three
     * integer numbers with a fixed dimension. The dimension is determined by
     * the template parameter Dim. The coordinates are stored in a std::array of
     * type Index_t.
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
     * floating point numbers with a fixed dimension. The dimension is
     * determined by the template parameter Dim. The coordinates are stored in a
     * std::array of type Real.
     *
     * @tparam Dim The dimension of the real space coordinates. It should be
     * between 1 and 3.
     */
    template <size_t Dim>
    using Rcoord_t = std::array<Real, Dim>;

    /**@}*/

    /**
     * Class to represent integer (cell-) coordinates or real-valued
     * coordinates. This class can dynamically accept any spatial-dimension
     * between 1 and MaxDim, and DynCcoord references can be cast to
     * `muGrid::Ccoord_t &` or `muGrid::Rcoord_t &` references. These are used
     * when templating with the spatial dimension of the problem is
     * undesireable/impossible.
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
            static_assert(Dim <= MaxDim,
                          "Coord has more than MaxDim dimensions.");
            return fill_front_helper(ccoord, std::make_index_sequence<Dim>{});
        }

       public:
        /**
         * @typedef iterator
         * @brief A type alias for an iterator over the elements of a
         * std::array.
         *
         * This type alias is used to create an iterator that can traverse the
         * elements of a std::array. The std::array is templated on type T and
         * has a maximum size of MaxDim. The iterator can be used to access and
         * modify the elements of the std::array.
         */
        using iterator = typename std::array<T, MaxDim>::iterator;

        /**
         * @typedef const_iterator
         * @brief A type alias for a constant iterator over the elements of a
         * std::array.
         *
         * This type alias is used to create a constant iterator that can
         * traverse the elements of a std::array. The std::array is templated on
         * type T and has a maximum size of MaxDim. The constant iterator can be
         * used to access the elements of the std::array, but cannot modify
         * them.
         */
        using const_iterator = typename std::array<T, MaxDim>::const_iterator;

        //! default constructor
        DynCcoord() : dim{}, long_array{} {};

        /**
         * @brief Constructs a DynCcoord object from an initializer list.
         *
         * This constructor creates a DynCcoord object using an initializer
         * list. The length of the initializer list determines the spatial
         * dimension of the coordinate. The initializer list must have a length
         * between 1 and MaxDim.
         *
         * @param init_list Initializer list used to set the values of the
         * DynCcoord object. The length of the list becomes the spatial
         * dimension of the coordinate.
         * @throws RuntimeError If the length of the initializer list is greater
         * than MaxDim, a RuntimeError is thrown with a message indicating the
         * maximum dimension and the provided dimension.
         */
        DynCcoord(std::initializer_list<T> init_list)
            : dim(init_list.size()), long_array{} {
            if (this->dim > Dim_t(MaxDim)) {
                std::stringstream error{};
                error << "The maximum dimension representable by this dynamic "
                         "array is "
                      << MaxDim << ". You supplied an initialiser list with "
                      << init_list.size() << " entries.";
                throw RuntimeError(error.str());
            }
            std::copy(init_list.begin(), init_list.end(),
                      this->long_array.begin());
        }

        /**
         * @brief Constructs a DynCcoord object with a specified dimension.
         *
         * This constructor creates a DynCcoord object with a specified
         * dimension. The dimension must be between 1 and MaxDim. Note: This
         * constructor requires regular (round) braces '()'. Using curly braces
         * '{}' will result in the initializer list constructor being called and
         * creating a DynCcoord with spatial dimension 1.
         *
         * @param dim The spatial dimension of the DynCcoord object. It needs to
         * be between 1 and MaxDim.
         * @param value The value to fill the DynCcoord object with. (optional)
         */
        explicit DynCcoord(Dim_t dim, const T value = T{})
            : dim{dim}, long_array{} {
            std::fill(this->long_array.begin(), this->long_array.end(), value);
        }

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
         * This constructor creates a DynCcoord object using a std::vector. The
         * size of the std::vector determines the spatial dimension of the
         * coordinate. The std::vector must have a size between 1 and MaxDim.
         *
         * @param ccoord std::vector used to set the values of the DynCcoord
         * object. The size of the vector becomes the spatial dimension of the
         * coordinate.
         * @throws RuntimeError If the size of the std::vector is greater than
         * MaxDim, a RuntimeError is thrown with a message indicating the
         * maximum dimension and the provided dimension.
         */
        explicit DynCcoord(const std::vector<T> & ccoord)
            : dim{Dim_t(ccoord.size())}, long_array{} {
            if (this->dim > Dim_t(MaxDim)) {
                std::stringstream error{};
                error << "The maximum dimension representable by this dynamic "
                         "array is "
                      << MaxDim << ". You supplied a vector with "
                      << ccoord.size() << " entries.";
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

        //! comparison operator
        bool operator!=(const DynCcoord & other) const {
            return !(*this == other);
        }

        //! element-wise addition
        DynCcoord & operator+=(const DynCcoord & other) {
            if (this->get_dim() != other.get_dim()) {
                std::stringstream error{};
                error << "you are trying to add a " << this->get_dim()
                      << "-dimensional coord to a " << other.get_dim()
                      << "-dimensional coord element-wise.";
                throw RuntimeError(error.str());
            }
            for (auto && tup : akantu::zip(*this, other)) {
                std::get<0>(tup) += std::get<1>(tup);
            }
            return *this;
        }

        //! element-wise addition
        template <typename T2>
        DynCcoord<MaxDim, decltype(T{} + T2{})>
        operator+(const DynCcoord<MaxDim, T2> & other) const {
            if (this->get_dim() != other.get_dim()) {
                std::stringstream error{};
                error << "you are trying to add a " << this->get_dim()
                      << "-dimensional coord to a " << other.get_dim()
                      << "-dimensional coord element-wise.";
                throw RuntimeError(error.str());
            }
            DynCcoord<MaxDim, decltype(T{} + T2{})> retval(this->get_dim());
            for (Dim_t i{0}; i < this->get_dim(); ++i) {
                retval[i] = this->operator[](i) + other[i];
            }
            return retval;
        }

        //! element-wise subtraction
        DynCcoord & operator-=(const DynCcoord & other) {
            if (this->get_dim() != other.get_dim()) {
                std::stringstream error{};
                error << "you are trying to subtract a " << this->get_dim()
                      << "-dimensional coord from a " << other.get_dim()
                      << "-dimensional coord element-wise.";
                throw RuntimeError(error.str());
            }
            for (auto && tup : akantu::zip(*this, other)) {
                std::get<0>(tup) -= std::get<1>(tup);
            }
            return *this;
        }

        //! element-wise subtraction
        template <typename T2>
        DynCcoord<MaxDim, decltype(T{} - T2{})>
        operator-(const DynCcoord<MaxDim, T2> & other) const {
            if (this->get_dim() != other.get_dim()) {
                std::stringstream error{};
                error << "you are trying to subtract a " << this->get_dim()
                      << "-dimensional coord from a " << other.get_dim()
                      << "-dimensional coord element-wise.";
                throw RuntimeError(error.str());
            }
            DynCcoord<MaxDim, decltype(T{} - T2{})> retval(this->get_dim());
            for (Dim_t i{0}; i < this->get_dim(); ++i) {
                retval[i] = this->operator[](i) - other[i];
            }
            return retval;
        }

        //! element-wise subtraction
        template <typename T2>
        DynCcoord<MaxDim, decltype(T{} - T2{})> operator-(T2 other) const {
            DynCcoord<MaxDim, decltype(T{} - T2{})> retval(this->get_dim());
            for (Dim_t i{0}; i < this->get_dim(); ++i) {
                retval[i] = this->operator[](i) - other;
            }
            return retval;
        }

        //! element-wise multiplication
        template <typename T2>
        DynCcoord<MaxDim, decltype(T{} * T2{})>
        operator*(const DynCcoord<MaxDim, T2> & other) const {
            if (this->get_dim() != other.get_dim()) {
                std::stringstream error{};
                error << "you are trying to multiply a " << this->get_dim()
                      << "-dimensional coord by a " << other.get_dim()
                      << "-dimensional coord element-wise.";
                throw RuntimeError(error.str());
            }
            DynCcoord<MaxDim, decltype(T{} * T2{})> retval(this->get_dim());
            for (Dim_t i{0}; i < this->get_dim(); ++i) {
                retval[i] = this->operator[](i) * other[i];
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

        //! modulo assignment operator (mostly for periodic boundaries stuff)
        DynCcoord & operator%=(const DynCcoord & other) {
            for (auto && tup : akantu::zip(*this, other)) {
                std::get<0>(tup) %= std::get<1>(tup);
                if (std::get<0>(tup) < 0) {
                    std::get<0>(tup) += std::get<1>(tup);
                }
            }
            return *this;
        }

        //! modulo operator (mostly for periodic boundaries stuff)
        DynCcoord operator%(const DynCcoord & other) const {
            DynCcoord ret_val{*this};
            ret_val %= other;
            return ret_val;
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
        Dim_t get_dim() const { return this->dim; }

        //! return the spatial dimension of this coordinate, STL compatibility
        Dim_t size() const { return this->dim; }

        //! convert into a vector
        explicit operator std::vector<T>() const {
            std::vector<T> v;
            for (auto && el : *this) {
                v.push_back(el);
            }
            return v;
        }

        //! iterator to the first entry for iterating over only the valid
        //! entries
        iterator begin() { return this->long_array.begin(); }
        //! iterator past the dim-th entry for iterating over only the valid
        //! entries
        iterator end() { return this->long_array.begin() + this->dim; }
        //! const iterator to the first entry for iterating over only the valid
        //! entries
        const_iterator begin() const { return this->long_array.begin(); }
        //! const iterator past the dim-th entry for iterating over only the
        //! valid entries
        const_iterator end() const {
            return this->long_array.begin() + this->dim;
        }

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
     * Cell coordinates, i.e. up to three integer numbers with dynamic
     * (determined during runtime) dimension
     */
    using IntCoord_t = DynCcoord<fourD>;  // 4D for alignment

    /**
     * Real space coordinates, i.e. up to three floating point numbers with
     * dynamic (determined during runtime) dimension
     */
    using RealCoord_t = DynCcoord<fourD, Real>;  // 4D for alignment

    /**
     * return a Eigen representation of the data stored in a std::array (e.g.,
     * for doing vector operations on a coordinate)
     */
    template <typename T, size_t Dim>
    Eigen::Map<Eigen::Matrix<T, Dim, 1>> eigen(std::array<T, Dim> & coord) {
        return Eigen::Map<Eigen::Matrix<T, Dim, 1>>{coord.data()};
    }

    /**
     * return a constant  Eigen representation of the data stored in a
     * std::array (e.g., for doing vector operations on a coordinate)
     */
    template <typename T, size_t Dim>
    Eigen::Map<const Eigen::Matrix<T, Dim, 1>>
    eigen(const std::array<T, Dim> & coord) {
        return Eigen::Map<const Eigen::Matrix<T, Dim, 1>>{coord.data()};
    }

    /**
     * return a Eigen representation of the data stored in a std::array (e.g.,
     * for doing vector operations on a coordinate)
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

    /**
     * Allows inserting `std::vector` into `std::ostream`s
     */
    template <typename T>
    std::ostream & operator<<(std::ostream & os,
                              const std::vector<T> & values) {
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

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CORE_COORDINATES_HH_
