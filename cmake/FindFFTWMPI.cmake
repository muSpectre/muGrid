# - Find the FFTWMPI library
#
# Usage:
#   find_package(FFTWMPI [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTWMPI_FOUND               ... true if fftw is found on the system
#   FFTWMPI_LIBRARIES           ... full path to fftw library
#   FFTWMPI_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTWMPI_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTWMPI_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTWMPI_LIBRARY            ... fftw library to use
#   FFTWMPI_INCLUDE_DIR        ... fftw include directory
#

#If environment variable FFTWMPIDIR is specified, it has same effect as FFTWMPI_ROOT
if( NOT FFTWMPI_ROOT AND ENV{FFTWDIR} )
  set( FFTWMPI_ROOT $ENV{FFTWDIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTWMPI_ROOT )
  pkg_check_modules( PKG_FFTWMPI QUIET "fftw3" )
endif()

#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if( ${FFTWMPI_USE_STATIC_LIBS} )
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
else()
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
endif()

#find libs
find_library(
  FFTWMPI_LIB
  NAMES "fftw3_mpi"
  PATHS ${FFTWMPI_ROOT} ${PKG_FFTWMPI_PREFIX} ${PKG_FFTWMPI_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  PATH_SUFFIXES "lib" "lib64"
)

find_library(
  FFTWMPIF_LIB
  NAMES "fftw3f_mpi"
  PATHS ${FFTWMPI_ROOT} ${PKG_FFTWMPI_PREFIX} ${PKG_FFTWMPI_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  PATH_SUFFIXES "lib" "lib64"
)

find_library(
  FFTWMPIL_LIB
  NAMES "fftw3l_mpi"
  PATHS ${FFTWMPI_ROOT} ${PKG_FFTWMPI_PREFIX} ${PKG_FFTWMPI_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  PATH_SUFFIXES "lib" "lib64"
)

#find includes
find_path(
  FFTWMPI_INCLUDES
  NAMES "fftw3-mpi.h"
  PATHS ${FFTWMPI_ROOT} ${PKG_FFTWMPI_PREFIX} ${PKG_FFTWMPI_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES "include"
)

set(FFTWMPI_LIBRARIES ${FFTWMPI_LIB})

if(FFTWMPIF_LIB)
  set(FFTWMPI_LIBRARIES ${FFTWMPI_LIBRARIES} ${FFTWMPIF_LIB})
endif()

if(FFTWMPIL_LIB)
  set(FFTWMPI_LIBRARIES ${FFTWMPI_LIBRARIES} ${FFTWMPIL_LIB})
endif()

set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWMPI DEFAULT_MSG
                                  FFTWMPI_INCLUDES FFTWMPI_LIBRARIES)

mark_as_advanced(FFTWMPI_INCLUDES FFTWMPI_LIBRARIES FFTWMPI_LIB FFTWMPIF_LIB FFTWMPIL_LIB)

