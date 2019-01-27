# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  Main configuration file
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
# Additional permission under GNU GPL version 3 section 7
#
# If you modify this Program, or any covered work, by linking or combining it
# with proprietary FFT implementations or numerical libraries, containing parts
# covered by the terms of those libraries' licenses, the licensors of this
# Program grant you additional permission to convey the resulting work.
# =============================================================================

#compile executables

set(binaries
  demonstrator1.cc
  demonstrator_dynamic_solve.cc
  demonstrator2.cc
  hyper-elasticity.cc
  small_case.cc
  )

if(${MUSPECTRE_MPI_PARALLEL})
  list(APPEND binaries
    demonstrator_mpi.cc
    )
endif()

foreach(binaryfile ${binaries})
  get_filename_component(binaryname  ${binaryfile} NAME_WE)
  add_executable(${binaryname} ${binaryfile})
  target_link_libraries(${binaryname} PRIVATE muSpectre::muSpectre cxxopts)

  muSpectre_move_to_project(${binaryname})
endforeach()

# or copy them
file(GLOB pybins "${CMAKE_SOURCE_DIR}/bin/*.py")
foreach(pybin ${pybins})
  get_filename_component(binaryname  ${pybin} NAME_WE)
  configure_file(
    ${pybin}
    "${CMAKE_BINARY_DIR}/${binaryname}.py"
    COPYONLY)
endforeach()

# additional files to copy
set(FILES_FOR_COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/odd_image.npz)

foreach(file_for_copy ${FILES_FOR_COPY})
  get_filename_component(binaryname  ${file_for_copy} NAME)
  configure_file(
    ${FILE_FOR_COPY}
    "${CMAKE_BINARY_DIR}/${binaryname}"
    COPYONLY)
endforeach()
