# =============================================================================
# file   CMakeLists.txt
#
# @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
#
# @date   04 May 2021
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

# copy python scripts
file(GLOB pyexamples "${CMAKE_SOURCE_DIR}/examples/paper_examples/*.py")
foreach(pyexample ${pyexamples})
    get_filename_component(binaryname  ${pyexample} NAME_WE)
    configure_file(
            ${pyexample}
            "${CMAKE_BINARY_DIR}/examples/paper_examples/${binaryname}.py"
            COPYONLY)
endforeach()

######### Test examples ##########
set(muspectre_paper_examples)

macro(muSpectre_add_example)
    muTools_add_test(${ARGN} LINK_LIBRARIES ${MUSPECTRE_NAMESPACE}muSpectre TEST_LIST muspectre_paper_examples)
endmacro()

muSpectre_add_example(example_A-single_pixel_inhomogeneity
        TYPE PYTHON example_A-single_pixel_inhomogeneity.py
        --to-file single_pixel_inhomogeneity.pdf)

muSpectre_add_example(example_B-two_pillars_and_vacuum
        TYPE PYTHON example_B-two_pillars_and_vacuum.py
        --to-file two_pillars_and_vacuum.pdf)

# copy eshelby reference files
file(GLOB eshresult "${CMAKE_SOURCE_DIR}/examples/paper_examples/stored_eshelby_results/*.npz")

foreach(eshresult ${eshresult})
    get_filename_component(binaryname  ${eshresult} NAME_WE)
    configure_file(
            ${eshresult}
            "${CMAKE_BINARY_DIR}/examples/paper_examples/stored_eshelby_results/\
${binaryname}.npz"
            COPYONLY)
endforeach()

muSpectre_add_example(example_C-eshelby_inhomogeneity
   TYPE PYTHON example_C-eshelby_inhomogeneity.py
   --to-file eshelby_inhomogeneity.pdf --small_simulation)

muSpectre_add_example(test_analytical_eshelby_solution
  TYPE PYTHON test_analytical_eshelby_solution.py)


# 2D concrete example input files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/concrete_micro_structure_201.npy
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/random_field_201.npy
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)


muSpectre_add_example(example_D-damage_problem
  TYPE PYTHON example_D-damage_problem.py -a 10.0 -n 2 -d 1 -g 0.02 -f 2.0e-6 -s 10 -v 0 -FEM -SNC)
