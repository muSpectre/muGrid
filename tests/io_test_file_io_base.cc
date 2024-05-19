/**
 * @file   io_test_file_io_base.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   04 Aug 2020
 *
 * @brief  description
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

#include <boost/mpl/list.hpp>

#include "mpi_context.hh"

#include "libmugrid/file_io_base.hh"
#include "libmugrid/file_io_netcdf.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(file_io_base);

  BOOST_AUTO_TEST_CASE(FileIOBaseClass) {
    const std::string file_name{"file_io_base_test.nc"};
    remove(file_name.c_str());  // remove test_file if it already exists
    FileIOBase::OpenMode open_mode{FileIOBase::OpenMode::Write};
    auto & comm{MPIContext::get_context().comm};
    FileIONetCDF file_io_object(file_name, open_mode, comm);

    BOOST_CHECK_EQUAL(comm.size(), file_io_object.get_communicator().size());

    const FileFrame frame = file_io_object.append_frame();
    BOOST_CHECK_EQUAL(file_io_object.size(), 1);

    for (volatile FileFrame frame_it : file_io_object) {
      continue;
    }
  };

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
