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

find_package(PythonInterp ${MUSPECTRE_PYTHON_MAJOR_VERSION} REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

#compile executables
set(binaries
  demonstrator_dynamic_solve.cc
  hyper-elasticity.cc
  small_case.cc
  small_elasto_plastic_case.cc
  projection_comparison.cc
  )

if(${MUSPECTRE_MPI_PARALLEL})
  list(APPEND binaries
    demonstrator_mpi.cc
    )
endif()


# or copy them
file(GLOB pyexamples "${CMAKE_SOURCE_DIR}/examples/*.py")
foreach(pyexample ${pyexamples})
  get_filename_component(binaryname  ${pyexample} NAME_WE)
  configure_file(
    ${pyexample}
    "${CMAKE_BINARY_DIR}/examples/${binaryname}.py"
    COPYONLY)
endforeach()



######### Test examples ##########
set(muspectre_examples)

macro(muSpectre_add_example)
  muTools_add_test(${ARGN} LINK_LIBRARIES ${MUSPECTRE_NAMESPACE}muSpectre TEST_LIST muspectre_examples)
endmacro()

muSpectre_add_example(small_case_cc_example
  SOURCES small_case.cc TYPE BOOST
  --report_level=detailed)

muSpectre_add_example(hyper-elasticity_cc_example
  SOURCES hyper-elasticity.cc TYPE BOOST
  --report_level=detailed)

muSpectre_add_example(demonstrator_dynamic_solve_cc_example
  SOURCES demonstrator_dynamic_solve.cc
  TYPE BOOST 5)

muSpectre_add_example(projection_comparison_cc_example
  SOURCES  projection_comparison.cc
  TYPE BOOST 5 0)

muSpectre_add_example(small_elasto_plastic_case_cc_example
  SOURCES  small_elasto_plastic_case.cc
  TYPE BOOST --report_level=detailed)

if(${MUSPECTRE_MPI_PARALLEL})
  muSpectre_add_example(demonstrator_mpi_cc_example1
    SOURCES demonstrator_mpi.cc
    TYPE BOOST MPI_NB_PROCS 1  5 )
  muSpectre_add_example(demonstrator_mpi_cc_example2
    SOURCES demonstrator_mpi.cc
    TYPE BOOST MPI_NB_PROCS 2  5 )
endif()

################## Pyhton examples ###################
if(${SPLIT_CELL})
  muSpectre_add_example(ringing_split_laminate_py_example
    TYPE PYTHON  circle_gibbs_ringing_simple_split_laminate.py 1)
endif()

muSpectre_add_example(crack_py_example
  TYPE PYTHON crack.py 1)

muSpectre_add_example(buckling_py_example
  TYPE PYTHON buckling.py 1)

muSpectre_add_example(hyper-elasticity_py_example
  TYPE PYTHON hyper-elasticity.py)

muSpectre_add_example(projection_comparison_py_example
  TYPE PYTHON projection_comparison.py)

muSpectre_add_example(small_case_py_example
  TYPE PYTHON small_case.py)

muSpectre_add_example(small_elasto_plastic_case_py_example
  TYPE PYTHON small_elasto_plastic_case.py)

muSpectre_add_example(tutorial_example_py_example
  TYPE PYTHON tutorial_example.py 1)

muSpectre_add_example(visualisation_example_finite_py_example
  TYPE PYTHON visualisation_example_finite_strain.py 7 7 1)

muSpectre_add_example(comparison_small_strain_py_example
  TYPE PYTHON  comparison_small_strain.py 1)

muSpectre_add_example(visco_py_example
  TYPE PYTHON  visco_example.py 1)

muSpectre_add_example(visco_damage_py_example
  TYPE PYTHON  visco_example_damage.py 1)

muSpectre_add_example(field_extraction_py_example
  TYPE PYTHON  field_extraction_example.py)

muSpectre_add_example(first_conc_py_example
  TYPE PYTHON  first_concrete_example.py 1)

muSpectre_add_example(sec_conc_py_example
  TYPE PYTHON  second_concrete_example.py 1)

# additional file(s) to copy
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/concrete_micro_structure.npy
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/random_field.npy
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)


if(${MUSPECTRE_MPI_PARALLEL})
  # additional files to copy
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/odd_image.npz
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
 muSpectre_add_example(elasto_plastic_comparison_py_example1
   TYPE PYTHON MPI_NB_PROCS 1 elasto_plastic_comparison.py 1)
 muSpectre_add_example(elasto_plastic_comparison_py_example2
   TYPE PYTHON MPI_NB_PROCS 2 elasto_plastic_comparison.py 1)
  muSpectre_add_example(crack_py_example1
          TYPE PYTHON MPI_NB_PROCS 1 crack.py 1)
  muSpectre_add_example(crack_py_example2
          TYPE PYTHON MPI_NB_PROCS 2 crack.py 1)
endif()
