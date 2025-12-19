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
    # core/
    core/exception.cc
    core/grid_common.cc
    core/units.cc
    # grid/
    grid/ccoord_operations.cc
    # field/
    field/field.cc
    field/field_typed.cc
    field/field_map.cc
    field/state_field.cc
    field/state_field_map.cc
    # collection/
    collection/field_collection.cc
    collection/field_collection_global.cc
    collection/field_collection_local.cc
    # mpi/
    mpi/communicator.cc
    mpi/cartesian_communicator.cc
    mpi/cartesian_decomposition.cc
    # operators/
    operators/convolution_operator.cc
    # io/
    io/file_io_base.cc
    # util/
    util/raw_memory_operations.cc
    # kokkos/
    kokkos/kokkos_init.cc
    kokkos/kokkos_profiling_workaround.cc
    # fft/
    fft/pocketfft_backend.cc
    fft/pencil_transpose.cc
    fft/fft_utils.cc
    fft/fft_engine.cc
    "${CMAKE_CURRENT_BINARY_DIR}/version.cc"
)

# Add NetCDF source if available
if(MUGRID_WITH_NETCDF)
    list(APPEND MUGRID_SOURCES io/file_io_netcdf.cc)
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/grid>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/field>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/collection>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mpi>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/operators>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/io>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/util>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kokkos>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fft>
        $<INSTALL_INTERFACE:include>
)

# Link Eigen3
target_link_libraries(muGrid PUBLIC Eigen3::Eigen)

# Link Kokkos (required dependency)
# Use WHOLE_ARCHIVE to resolve internal circular dependencies in Kokkos static libs
# This is needed because libkokkoscore.a has objects that depend on each other,
# and macOS linker doesn't rescan archives
target_link_libraries(muGrid PUBLIC
    $<LINK_LIBRARY:WHOLE_ARCHIVE,Kokkos::kokkoscore>
    Kokkos::kokkoscontainers
    Kokkos::kokkosalgorithms
    Kokkos::kokkossimd
)

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

# Install header files for each subdirectory
foreach(subdir core grid field collection mpi operators io util kokkos fft)
    file(GLOB MUGRID_${subdir}_HEADERS "${subdir}/*.hh")
    install(FILES ${MUGRID_${subdir}_HEADERS}
        DESTINATION include/libmugrid/${subdir}
    )
endforeach()

# Install pocketfft header
install(FILES fft/pocketfft/pocketfft_hdronly.h
    DESTINATION include/libmugrid/fft/pocketfft
)
