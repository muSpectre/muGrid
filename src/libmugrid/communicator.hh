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

#include "Eigen/Dense"

#include "grid_common.hh"

namespace muGrid {

  template <typename T>
  using DynMatrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

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
    explicit Communicator(MPI_Comm comm = MPI_COMM_SELF);
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

    //! Barrier syncronization, muGrids communicator is calling MPI's Barrier()
    //! A Barrier blocks until all processes in the communicator have reached
    //! this point
    void barrier() {
      if (comm == MPI_COMM_NULL) {
        return;
      }
      int message{MPI_Barrier(this->comm)};
      if (message != 0) {
        std::stringstream error{};
        error << "MPI_BArrier failed with " << message << " on rank "
              << this->rank();
        throw RuntimeError(error.str());
      }
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

    //! max reduction on scalar types
    template <typename T>
    T max(const T & arg) const {
      if (comm == MPI_COMM_NULL)
        return arg;
      T res;
      MPI_Allreduce(&arg, &res, 1, mpi_type<T>(), MPI_MAX, this->comm);
      return res;
    }

    //! sum reduction on Eigen::Matrix types
    template <typename T, int RowsAtCompileTime, int ColsAtCompileTime>
    Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime>
    sum(const Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> & arg)
        const {
      if (this->comm == MPI_COMM_NULL)
        return arg;
      Eigen::Matrix<T, RowsAtCompileTime, ColsAtCompileTime> res;
      res.setZero();
      const auto count{arg.size()};
      MPI_Allreduce(arg.data(), res.data(), count, mpi_type<T>(), MPI_SUM,
                    this->comm);
      return res;
    }

    //! sum reduction on Eigen::Matrix types
    template <typename T>
    DynMatrix_t<T> sum(const DynMatrix_t<T> & arg) const {
      if (this->comm == MPI_COMM_NULL)
        return arg;
      DynMatrix_t<T> res(arg.rows(), arg.cols());
      res.setZero();
      const auto count{arg.size()};
      MPI_Allreduce(arg.data(), res.data(), count, mpi_type<T>(), MPI_SUM,
                    this->comm);
      return res;
    }

    //! sum reduction on Eigen::Matrix types
    template <typename T>
    DynMatrix_t<T> sum(const Eigen::Ref<DynMatrix_t<T>> & arg) const {
      if (this->comm == MPI_COMM_NULL)
        return arg;
      DynMatrix_t<T> res(arg.rows(), arg.cols());
      res.setZero();
      const auto count{arg.size()};
      MPI_Allreduce(arg.data(), res.data(), count, mpi_type<T>(), MPI_SUM,
                    this->comm);
      return res;
    }

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

    //! gather on EigenMatrix types
    //! If the matrices are of different sizes they are collected along the
    //! column index (second index). This is even possible for 0x0 arguments on
    //! some cores. The following example shows how the gathered result is
    //! collected for different sized matrices on the different cores:
    //! core   argument matrix   result matrix
    //!   0         3 x 1          3 x (24/3)
    //!   1         3 x 5          3 x (24/3)
    //!   2         0 x 0          3 x (24/3)
    //!   3         3 x 2          3 x (24/3)
    //!
    //! where 24 (=3*1+3*5+0*0+3*2) is the total number of elements on all
    //! cores. You should always have the same number of points in the row index
    //! (first index) except for 0 if there is nothing to be gathered on the
    //! core, otherwise there could be wrong behaviour. This condition is only
    //! checked in Debug mode.
    template <typename T>
    DynMatrix_t<T> gather(const Eigen::Ref<DynMatrix_t<T>> & arg) const {
      if (this->comm == MPI_COMM_NULL)
        return arg;

      int comm_size = this->size();

      // gather the number of rows on each core to define the output shape
      // (nb_rows_default = max(nb_rows_all) of the result
      Index_t nb_rows_loc{arg.rows()};
      Index_t nb_rows_max{0};
      auto message{MPI_Allreduce(&nb_rows_loc, &nb_rows_max, 1,
                                 mpi_type<Index_t>(), MPI_MAX, this->comm)};
      if (message != 0) {
        std::stringstream error{};
        error << "MPI_Allreduce MPI_MAX failed with " << message << " on rank "
              << this->rank();
        throw RuntimeError(error.str());
      }

      // It is only allowed to have matrices with a fixed number of rows or
      // empty matrices. Hence either nb_rows_loc == nb_rows_max or nb_rows_loc
      // == 0. Otherwise you might have undefined behaviour because the output
      // gathered matrix might have a wrong shape.
      assert((nb_rows_loc == nb_rows_max) or (nb_rows_loc == 0));

      // gather the number of elements on each core to define the output shape
      // (nb_cols = nb_entries/nb_rows_max) of the result
      Index_t send_buf_size(arg.size());
      std::vector<int> arg_sizes(comm_size, 0);
      message = MPI_Allgather(&send_buf_size, 1, mpi_type<int>(),
                              arg_sizes.data(), 1, mpi_type<int>(), this->comm);
      if (message != 0) {
        std::stringstream error{};
        error << "MPI_Allgather failed with " << message << " on rank "
              << this->rank();
        throw RuntimeError(error.str());
      }

      int nb_entries = 0;
      for (auto i = 0; i < comm_size; ++i) {
        nb_entries += arg_sizes[i];
      }

      // check if by accident a vector was handed over, rows=0, cols!=0. Thus
      // nb_rows is zero everywhere and so nb_rows_max is zero. However, there
      // are columns != 0, which leads to nb_entries != 0 and would lead in the
      // following to a zero division.
      assert(!((nb_rows_max == 0) and (nb_entries != 0)));

      // compute the offset at which the data from each processor is written
      // into the result
      std::vector<int> displs(comm_size, 0);
      for (auto i = 0; i < comm_size - 1; ++i) {
        displs[i + 1] = displs[i] + arg_sizes[i];
      }

      // initialise the result matrix with zeros
      if ((nb_rows_max == 0) and (nb_entries == 0)) {
        // If there is no data to collect return a 0x0 matrix.
        DynMatrix_t<T> res(0, 0);
        return res;
      }
      DynMatrix_t<T> res(nb_rows_max, nb_entries / nb_rows_max);
      res.setZero();

      message = MPI_Allgatherv(arg.data(), send_buf_size, mpi_type<T>(),
                               res.data(), arg_sizes.data(), displs.data(),
                               mpi_type<T>(), this->comm);
      if (message != 0) {
        std::stringstream error{};
        error << "MPI_Allgatherv failed with " << message << " on rank "
              << this->rank();
        throw RuntimeError(error.str());
      }
      return res;
    }

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
    //! return logical and
    bool logical_and(const bool & arg) const {
      if (this->comm == MPI_COMM_NULL) {
        return arg;
      } else {
        bool res;
        MPI_Allreduce(&arg, &res, 1, MPI_C_BOOL, MPI_LAND, this->comm);
        return res;
      }
    }

    //! return logical and
    bool logical_or(const bool & arg) const {
      if (this->comm == MPI_COMM_NULL) {
        return arg;
      } else {
        bool res;
        MPI_Allreduce(&arg, &res, 1, MPI_C_BOOL, MPI_LOR, this->comm);
        return res;
      }
    }

    MPI_Comm get_mpi_comm() { return this->comm; }

    //! find whether the underlying communicator is mpi
    static bool has_mpi() { return true; }

   private:
    MPI_Comm comm;
  };

#else  // WITH_MPI

  //! stub communicator object that doesn't communicate anything
  class Communicator {
   public:
    Communicator() {}
    ~Communicator() {}

    //! get rank of present process
    int rank() const { return 0; }

    //! get total number of processes
    int size() const { return 1; }

    //! Barrier syncronization, nothing to be done in serial
    void barrier() { return; }

    //! sum reduction on scalar types
    template <typename T>
    T sum(const T & arg) const {
      return arg;
    }

    //! max reduction on scalar types
    template <typename T>
    T max(const T & arg) const {
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
    T gather(const T & arg) const {
      return arg;
    }

    //! broadcast of scalar types
    template <typename T>
    T bcast(T & arg, const Int &) {
      return arg;
    }

    //! return logical and
    bool logical_or(const bool & arg) const { return arg; }

    //! return logical and
    bool logical_and(const bool & arg) const { return arg; }

    //! find whether the underlying communicator is mpi
    static bool has_mpi() { return false; }
  };

#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_COMMUNICATOR_HH_
