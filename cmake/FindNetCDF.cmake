# - Find NetCDF
# Find the native NetCDF includes and library
#
#  NETCDF_INCLUDES    - where to find netcdf.h, etc
#  NETCDF_LIBRARIES   - Link these libraries when using NetCDF
#  NETCDF_FOUND       - True if NetCDF was found
#
# Normal usage would be:
#  find_package (NetCDF REQUIRED)
#  target_link_libraries (uses_netcdf ${NETCDF_LIBRARIES})

if (NETCDF_INCLUDES AND NETCDF_LIBRARIES)
  # Already in cache, be silent
  set (NETCDF_FIND_QUIETLY TRUE)
endif (NETCDF_INCLUDES AND NETCDF_LIBRARIES)

find_path (NETCDF_INCLUDES netcdf.h
  HINTS "${NETCDF_ROOT}/include" "$ENV{NETCDF_ROOT}/include")

string(REGEX REPLACE "/include/?$" "/lib"
  NETCDF_LIB_HINT ${NETCDF_INCLUDES})

find_library (NETCDF_LIBRARIES
  NAMES netcdf
  HINTS ${NETCDF_LIB_HINT})

if ((NOT NETCDF_LIBRARIES) OR (NOT NETCDF_INCLUDES))
  message(STATUS "Trying to find NetCDF using LD_LIBRARY_PATH (we're desperate)...")

  file(TO_CMAKE_PATH "$ENV{LD_LIBRARY_PATH}" LD_LIBRARY_PATH)

  find_library(NETCDF_LIBRARIES
    NAMES netcdf
    HINTS ${LD_LIBRARY_PATH})

  if (NETCDF_LIBRARIES)
    get_filename_component(NETCDF_LIB_DIR ${NETCDF_LIBRARIES} PATH)
    string(REGEX REPLACE "/lib/?$" "/include"
      NETCDF_H_HINT ${NETCDF_LIB_DIR})

    find_path (NETCDF_INCLUDES netcdf.h
      HINTS ${NETCDF_H_HINT}
      DOC "Path to netcdf.h")
  endif()
endif()

# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NetCDF DEFAULT_MSG NETCDF_LIBRARIES NETCDF_INCLUDES)

mark_as_advanced (NETCDF_LIBRARIES NETCDF_INCLUDES)
