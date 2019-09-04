# - Find the FFTW3 library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTW3_FOUND               ... true if fftw is found on the system
#   FFTW3_LIBRARIES           ... full path to fftw library
#   FFTW3_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW3_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW3_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW3_LIBRARY            ... fftw library to use
#   FFTW3_INCLUDE_DIR        ... fftw include directory
#

#If environment variable FFTWDIR is specified, it has same effect as FFTW3_ROOT
if(NOT DEFINED FFTW3_ROOT AND DEFINED ENV{FFTWDIR})
  set(FFTW3_ROOT $ENV{FFTWDIR})
endif()

set(_FFTW3_PRECISIONS "double" "float" "quad")
set(_FFTW3_IMPLEMENTATIONS "sequential" "mpi" "openmp" "threads")

if(NOT FFTW3_FIND_COMPONENTS)
  set(FFTW3_FIND_COMPONENTS "double" "float" "quad")
endif()

set(_precisions)
set(_implementations)
foreach(_comp ${FFTW3_FIND_COMPONENTS})
  list(FIND _FFTW3_PRECISIONS ${_comp} _pos)
  if(NOT _pos EQUAL -1)
    list(APPEND _precisions ${_comp})
  else()
    list(APPEND _implementations ${_comp})
  endif()
endforeach()

if("${_implementations}" STREQUAL "")
  set(_implementations "sequential")
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if(PKG_CONFIG_FOUND AND NOT FFTW3_ROOT )
  pkg_check_modules(PKG_FFTW QUIET "fftw3" )
  mark_as_advanced(pkgcfg_lib_PKG_FFTW3_fftw3)
endif()

#Check whether to search static or dynamic libs
set(CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(${FFTW3_USE_STATIC_LIBS})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(_import_type STATIC)
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(_import_type SHARED)
endif()

set(_precision_double)
set(_precision_simple "f")
set(_precision_quad "l")

set(_needed_vars)
foreach(_impl ${_implementations})
  if("${_impl}" STREQUAL "sequential")
    set(_suffix)
    set(_namespace)
  elseif("${_impl}" STREQUAL "openmp")
    set(_suffix "_omp")
    set(_namespace "::${_impl}")
  else()
    set(_suffix "_${_impl}")
    set(_namespace "::${_impl}")
  endif()

  set(_include_suffix)
  if("${_impl}" STREQUAL "mpi")
    set(_include_suffix "-mpi")
  endif()
  find_path(
    FFTW3_${_impl}_INCLUDE_DIRS
    NAMES "fftw3${_include_suffix}.h"
    PATHS ${FFTW3_ROOT} ${LIB_INSTALL_DIR}
    HINTS ${PKG_FFTW3_PREFIX} ${PKG_FFTW3_LIBRARY_DIRS}
    PATH_SUFFIXES "include"
    )

  mark_as_advanced(FFTW3_${_impl}_INCLUDE_DIRS)

  list(APPEND _needed_vars
    FFTW3_${_impl}_INCLUDE_DIRS
    FFTW3_${_impl}_LIBRARIES
    )
  set(FFTW3_${_impl}_LIBRARIES)

  foreach(_prec ${_precisions})
    set(_lib FFTW3_${_prec}_${_impl}_LIBRARY)
   
    find_library(
      ${_lib}
      NAMES "fftw3${_precision_${_prec}}${_suffix}"
      PATHS ${FFTW3_ROOT} ${LIB_INSTALL_DIR}
      HINTS ${PKG_FFTW3_PREFIX} ${PKG_FFTW3_LIBRARY_DIRS}
      PATH_SUFFIXES "lib" "lib64"
      )

    if(${_lib})
      list(APPEND FFTW3_${_impl}_LIBRARIES ${${_lib}})

      add_library(fftw3::${_prec}${_namespace} ${_import_type} IMPORTED)
      set_target_properties(fftw3::${_prec}${_namespace} PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_${_impl}_INCLUDE_DIRS}
       	IMPORTED_LOCATION ${${_lib}})
      
      set(FFTW3_${_prec}_FOUND TRUE)
      set(FFTW3_${_impl}_FOUND TRUE)
      if(NOT DEFINED _fftw_version OR NOT _fftw_version)
	file(WRITE "${PROJECT_BINARY_DIR}/_fftw_version.c"
	  "#include <fftw3${_include_suffix}.h>
           #include <stdio.h>

           int main() {
             printf(\"%s\", fftw_version);
             return 0;
           }")

	try_run(_res _compile
	  ${PROJECT_BINARY_DIR}
	  "${PROJECT_BINARY_DIR}/_fftw_version.c"
	  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${FFTW3_${_impl}_INCLUDE_DIRS}"
	  LINK_LIBRARIES ${${_lib}}
	  RUN_OUTPUT_VARIABLE _fftw_version
	  COMPILE_OUTPUT_VARIABLE _compile_out
	  )

	if(_fftw_version)
	  string(REGEX MATCH "[0-9.]+" _fftw_version ${_fftw_version})
	endif()
      endif()
    endif()
    mark_as_advanced(FFTW3_${_prec}_${_impl}_LIBRARY)
  endforeach()
endforeach()

if(FFTW3_sequential_INCLUDE_DIRS)
  set(FFTW3_INCLUDE_DIRS ${FFTW3_sequential_INCLUDE_DIRS}
    CACHE PATH "Path to the include of fftw" FORCE)
  list(APPEND _needed_vars FFTW3_INCLUDE_DIRS)
  mark_as_advanced(FFTW3_INCLUDE_DIRS)
endif()

if(FFTW3_sequential_LIBRARIES)
  set(FFTW3_LIBRARIES ${FFTW3_sequential_LIBRARIES}
    CACHE PATH "FFTW libraries" FORCE)
  list(APPEND _needed_vars FFTW3_LIBRARIES)
  mark_as_advanced(FFTW3_LIBRARIES)
endif()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
  REQUIRED_VARS ${_needed_vars}
  VERSION_VAR _fftw_version
  HANDLE_COMPONENTS)
