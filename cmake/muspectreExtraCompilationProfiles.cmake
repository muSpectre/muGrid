#==============================================================================
# file   muspectreExtraCompilationProfiles.cmake
#
# @author Nicolas Richart <nicolas.richart@epfl.ch>
#
# @date   13 Aug 2021
#
# @brief  new compilation profiles
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

function(declare_compilation_profile name)
  include(CMakeParseArguments)

  cmake_parse_arguments(_args
    "" "COMPILER;LINKER;DOC" "" ${ARGN})

  string(TOUPPER "${name}" _u_name)

  if(NOT _args_DOC)
    string(TOLOWER "${name}" _args_DOC)
  endif()

  if(NOT _args_COMPILER)
    message(FATAL_ERROR "declare_compilation_profile: you should at least give COMPILER flags")
  endif()

  if(NOT _args_LINKER)
    set(_args_LINKER ${_args_COMPILER})
  endif()

  foreach(_flag CXX C Fortran SHARED_LINKER EXE_LINKER)
    set(_stage "compiler")
    set(_flags ${_args_COMPILER})
    if(_stage MATCHES ".*LINKER")
      set(_stage "linker")
      set(_flags ${_args_LINKER})
    endif()
    set(CMAKE_${_flag}_FLAGS_${_u_name} ${_flags}
      CACHE STRING "Flags used by the ${_stage} during coverage builds" FORCE)
    mark_as_advanced(CMAKE_${_flag}_FLAGS_${_u_name})
  endforeach()
endfunction()

declare_compilation_profile(coverage
  COMPILER "-g -ggdb3 -DNDEBUG -DAKANTU_NDEBUG -O2 --coverage")
