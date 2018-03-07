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

#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace µSpectre {

#ifdef WITH_MPI

  template<typename T> decltype(auto) mpi_type() { };
  template<> decltype(auto) mpi_type<char>() { return MPI_CHAR; }
  template<> decltype(auto) mpi_type<short>() { return MPI_SHORT; }
  template<> decltype(auto) mpi_type<int>() { return MPI_INT; }
  template<> decltype(auto) mpi_type<long>() { return MPI_LONG; }
  template<> decltype(auto) mpi_type<unsigned char>() { return MPI_UNSIGNED_CHAR; }
  template<> decltype(auto) mpi_type<unsigned short>() { return MPI_UNSIGNED_SHORT; }
  template<> decltype(auto) mpi_type<unsigned int>() { return MPI_UNSIGNED; }
  template<> decltype(auto) mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
  template<> decltype(auto) mpi_type<float>() { return MPI_FLOAT; }
  template<> decltype(auto) mpi_type<double>() { return MPI_DOUBLE; }

  //! lightweight abstraction for communicator object
  class Communicator {
  public:
    Communicator(MPI_Comm comm=MPI_COMM_NULL): comm{comm} {};
    ~Communicator() {};

    //! sum reduction on scalar types
    template<typename T>
    constexpr T sum(const T &arg) {
      T res;
      MPI_Allreduce(&arg, &res, 1, mpi_type<T>(), MPI_SUM, this->comm);
      return res;
    }

    constexpr MPI_Comm get_mpi_comm() { return this->comm; }

  private:
    MPI_Comm comm;
  };

#else /* WITH_MPI */

  //! stub communicator object that doesn't communicate anything
  class Communicator {
  public:
    Communicator() {};
    ~Communicator() {};

    //! sum reduction on scalar types
    template<typename T>
    constexpr T sum(T &arg) { return arg; }
  };

#endif

}

#endif /* COMMUNICATOR_H */