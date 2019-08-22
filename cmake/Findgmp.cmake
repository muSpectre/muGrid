
if (GMP_INCLUDE_DIR AND GMP_LIBRARIES)
  set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDE_DIR AND GMP_LIBRARIES)

add_library(gmp INTERFACE IMPORTED)

find_path(GMP_INCLUDE_DIR
  NAMES
  gmp.h
  PATHS
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
  )

find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG
  GMP_INCLUDE_DIR GMP_LIBRARIES)
mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARIES)


if( NOT ${GMP_FOUND})
  include(ExternalProject)
  Externalproject_Add(
    gmp_proj
    URL ftp://ftp.gnu.org/gnu/gmp/gmp-6.1.2.tar.bz2
    PREFIX ${PROJECT_BINARY_DIR}/external/gmp
    CONFIGURE_COMMAND sh -c "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./configure --prefix=<INSTALL_DIR> --enable-cxx"
    BUILD_IN_SOURCE TRUE
    LOG_BUILD TRUE
    LOG_CONFIGURE TRUE
    LOG_INSTALL TRUE
    )

  add_dependencies(gmp gmp_proj)
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/external/gmp/include)
  set_target_properties(gmp
    PROPERTIES
    INTERFACE_LINK_LIBRARIES ${PROJECT_BINARY_DIR}/external/gmp/lib/${CMAKE_SHARED_LIBRARY_PREFIX}gmp${CMAKE_SHARED_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_BINARY_DIR}/external/gmp/include
    )

  set(GMP_FOUND TRUE CACHE INTERNAL "To avoid cyclic search" FORCE)

else()
  set_target_properties(gmp
    PROPERTIES
    INTERFACE_LINK_LIBRARIES ${GMP_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES ${GMP_INCLUDE_DIR}
    )
endif()

