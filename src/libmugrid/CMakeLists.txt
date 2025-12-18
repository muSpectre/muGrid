# Configure version.cc from skeleton
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/version.cc.skeleton"
    "${CMAKE_CURRENT_BINARY_DIR}/version.cc"
    @ONLY
)

# Define version variables for the skeleton
set(GIT_IS_DIRTY "${MUGRID_GIT_DIRTY}")
set(GIT_COMMIT_DESCRIBE "${MUGRID_VERSION_STRING}")
set(GIT_HEAD_SHA1 "${MUGRID_GIT_HASH}")

# Re-configure with proper variable substitution
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/version.cc.skeleton"
    "${CMAKE_CURRENT_BINARY_DIR}/version.cc"
    @ONLY
)

# Source files for libmuGrid
set(MUGRID_SOURCES
    kokkos_init.cc
    exception.cc
    grid_common.cc
    ccoord_operations.cc
    convolution_operator.cc
    field_collection.cc
    field_collection_global.cc
    field_collection_local.cc
    file_io_base.cc
    field.cc
    field_typed.cc
    field_map.cc
    communicator.cc
    cartesian_communicator.cc
    cartesian_decomposition.cc
    raw_memory_operations.cc
    state_field.cc
    state_field_map.cc
    units.cc
    "${CMAKE_CURRENT_BINARY_DIR}/version.cc"
)

# Add NetCDF source if available
if(MUGRID_WITH_NETCDF)
    list(APPEND MUGRID_SOURCES file_io_netcdf.cc)
endif()

# Create shared library
add_library(muGrid STATIC ${MUGRID_SOURCES})

# Set library properties
set_target_properties(muGrid PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(muGrid
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

# Link Eigen3
target_link_libraries(muGrid PUBLIC Eigen3::Eigen)

# Link Kokkos (required dependency)
target_link_libraries(muGrid PUBLIC Kokkos::kokkos)

# Compile definitions
if(MUGRID_WITH_MPI)
    target_compile_definitions(muGrid PUBLIC WITH_MPI)
    target_link_libraries(muGrid PUBLIC MPI::MPI_CXX)
endif()

if(MUGRID_WITH_NETCDF)
    target_compile_definitions(muGrid PUBLIC WITH_NETCDF_IO)
    target_include_directories(muGrid PRIVATE ${MUGRID_NETCDF_INCLUDE_DIRS})
    target_link_libraries(muGrid PRIVATE ${MUGRID_NETCDF_LIBRARIES})
endif()

# Optional libraries for some systems
find_library(DL_LIBRARY dl)
if(DL_LIBRARY)
    target_link_libraries(muGrid PRIVATE ${DL_LIBRARY})
endif()

find_library(EXECINFO_LIBRARY execinfo)
if(EXECINFO_LIBRARY)
    target_link_libraries(muGrid PRIVATE ${EXECINFO_LIBRARY})
endif()

# Installation
# Build list of targets to export (muGrid + any FetchContent dependencies)
set(_MUGRID_INSTALL_TARGETS muGrid)
if(MUGRID_EIGEN_FROM_FETCHCONTENT)
    list(APPEND _MUGRID_INSTALL_TARGETS eigen)
endif()

install(TARGETS ${_MUGRID_INSTALL_TARGETS}
    EXPORT muGridTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install header files
file(GLOB MUGRID_HEADERS "*.hh")
install(FILES ${MUGRID_HEADERS}
    DESTINATION include/libmugrid
)
