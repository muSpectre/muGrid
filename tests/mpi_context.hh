/**
 * @file   mpi_initializer.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   07 Mar 2018
 *
 * @brief  Singleton for initialization and tear down of MPI.
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef MPI_CONTEXT_H
#define MPI_CONTEXT_H

#include "common/communicator.hh"

namespace muSpectre {

  /*!
   * MPI context singleton. Initialize MPI once when needed.
   */
  class MPIContext {
  public:
    Communicator comm;
    static MPIContext &get_context() {
      static MPIContext context;
      return context;
    }

  private:
    MPIContext(): comm(Communicator(MPI_COMM_WORLD)) {
      MPI_Init(&boost::unit_test::framework::master_test_suite().argc,
               &boost::unit_test::framework::master_test_suite().argv);
    }
    ~MPIContext() {
      // Wait for all processes to finish before calling finalize.
      MPI_Barrier(comm.get_mpi_comm());
      MPI_Finalize();
    }
  public:
    MPIContext(MPIContext const&) = delete;
    void operator=(MPIContext const&) = delete;
  };

}

#endif /* MPI_CONTEXT_H */
