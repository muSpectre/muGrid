/**
 * @file   communicator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   07 Mar 2018
 *
 * @brief  abstraction layer for the distributed memory communicator object
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_LIBMUGRID_COMMUNICATOR_HH_
#define SRC_LIBMUGRID_COMMUNICATOR_HH_

#include <type_traits>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <Eigen/Dense>

#include "grid_common.hh"

namespace muGrid {

  template <typename T>
  using DynMatrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  // template <typename T>
  // using Matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

#ifdef WITH_MPI

  template <typename T, typename T2 = T>
  inline decltype(auto) mpi_type() {
    static_assert(std::is_same<T, T2>::value,
                   "T2 is a SFINAE parameter, do not touch");
    static_assert(std::is_same<T, T2>::value and not std::is_same<T, T2>::value,
                  "The type you're trying to map has not been declared.");
    return MPI_LONG;
  }
  template <>
  inline decltype(auto) mpi_type<char>() {
    return MPI_CHAR;
  }
  template <>
  inline decltype(auto) mpi_type<short>() {  // NOLINT
    return MPI_SHORT;
  }
  template <>
  inline decltype(auto) mpi_type<int>() {
    return MPI_INT;
  }
  template <>
  inline decltype(auto) mpi_type<long>() {  // NOLINT
    return MPI_LONG;
  }
  template <>
  inline decltype(auto) mpi_type<long long>() {  // NOLINT
    return MPI_LONG_LONG_INT;
  }
  template <>
  inline decltype(auto) mpi_type<unsigned char>() {
    return MPI_UNSIGNED_CHAR;
  }
  template <>
  inline decltype(auto) mpi_type<unsigned short>() {  // NOLINT
    return MPI_UNSIGNED_SHORT;
  }
  template <>
  inline decltype(auto) mpi_type<unsigned int>() {
    return MPI_UNSIGNED;
  }
  template <>
  inline decltype(auto) mpi_type<unsigned long>() {  // NOLINT
    return MPI_UNSIGNED_LONG;
  }
  template <>
  inline decltype(auto) mpi_type<float>() {
    return MPI_FLOAT;
  }
  template <>
  inline decltype(auto) mpi_type<double>() {
    return MPI_DOUBLE;
  }
  template <>
  inline decltype(auto) mpi_type<Complex>() {
    return MPI_DOUBLE_COMPLEX;
  }

  //! lightweight abstraction for the MPI communicator object
  class Communicator {
   public:
    explicit Communicator(MPI_Comm comm = MPI_COMM_NULL);
    Communicator(const Communicator & other);
    ~Communicator();

    Communicator & operator=(const Communicator & other);

    //! get rank of present process
    int rank() const {
      // This is necessary here and below to be able to use the communicator
      // without a previous call to MPI_Init. This happens if you want to use
      // a version compiled with MPI for serial calculations.
      // Note that the difference between MPI_COMM_NULL and MPI_COMM_SELF is
      // that the latter actually executes the library function.
      if (comm == MPI_COMM_NULL)
        return 0;
      int res;
      MPI_Comm_rank(this->comm, &res);
      return res;
    }

    //! get total number of processes
    int size() const {
      if (comm == MPI_COMM_NULL)
        return 1;
      int res;
      MPI_Comm_size(this->comm, &res);
      return res;
    }

    //! sum reduction on scalar types
    template <typename T>
    T sum(const T & arg) const {
      if (comm == MPI_COMM_NULL)
        return arg;
      T res;
      MPI_Allreduce(&arg, &res, 1, mpi_type<T>(), MPI_SUM, this->comm);
      return res;
    }

    //! sum reduction on EigenMatrix types
    template <typename T>
    DynMatrix_t<T> sum_mat(const Eigen::Ref<DynMatrix_t<T>> & arg) const;

    //! ordered partial cumulative sum on scalar types. With the nomenclatur p0
    //! = the processor with rank = 0 and so on the following example
    //! demonstrates the cumulative sum of arg='a' returned on res='b'
    //! p0 a=2, b=0;  p1 a=3, b=0;  p2 a=1, b=0;  p3 a=5, b=0
    //! after computing the cumulative sum we find:
    //! p0 a=2, b=2,  p1 a=3, b=5,  p2 a=1, b=6,  p3 a=5, b=11
    template <typename T>
    T cumulative_sum(const T & arg) const {
      if (comm == MPI_COMM_NULL) {
        return arg;
      }
      T res;
      MPI_Scan(&arg, &res, 1, mpi_type<T>(), MPI_SUM, this->comm);
      return res;
    }

    // //! sum reduction on EigenMatrix types
    // template <typename T>
    // DynMatrix_t<T> sum_mat(const Eigen::Ref<Matrix_t<T>> & arg) const;

    //! gather on EigenMatrix types
    template <typename T>
    DynMatrix_t<T> gather(const Eigen::Ref<DynMatrix_t<T>> & arg) const;

    //! broadcast of scalar types
    //! broadcasts arg from root to all processors and additionally returns the
    //! broadcasted value in res (this is an overhead but usefull for the python
    //! binding).
    template <typename T>
    T bcast(T & arg, const Int & root) {
      if (comm == MPI_COMM_NULL) {
        return arg;
      } else {
        MPI_Bcast(&arg, 1, mpi_type<T>(), root, this->comm);
        T res = arg;
        return res;
      }
    }

    MPI_Comm get_mpi_comm() { return this->comm; }

    //! find whether the underlying communicator is mpi
    // TODO(pastewka) why do we need this?
    static bool has_mpi() { return true; }

   private:
    MPI_Comm comm;
  };

#else   // WITH_MPI

  //! stub communicator object that doesn't communicate anything
  class Communicator {
   public:
    Communicator() {}
    ~Communicator() {}

    //! get rank of present process
    int rank() const { return 0; }

    //! get total number of processes
    int size() const { return 1; }

    //! sum reduction on scalar types
    template <typename T>
    T sum(const T & arg) const {
      return arg;
    }

    //! sum reduction on EigenMatrix types
    template <typename T>
    DynMatrix_t<T> sum_mat(const Eigen::Ref<DynMatrix_t<T>> & arg) const {
      return arg;
    }

    //! ordered partial cumulative sum on scalar types. Find more details in the
    //! doc of the into the parallel implementation.
    template <typename T>
    T cumulative_sum(const T & arg) const {
      return arg;
    }

    //! gather on EigenMatrix types
    template <typename T>
    DynMatrix_t<T> gather(const Eigen::Ref<DynMatrix_t<T>> & arg) const {
      return arg;
    }

    //! find whether the underlying communicator is mpi
    // TODO(pastewka) why do we need this?

    //! broadcast of scalar types
    template <typename T>
    T bcast(T & arg, const Int &) {
      return arg;
    }

    static bool has_mpi() { return false; }
  };

#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_COMMUNICATOR_HH_
