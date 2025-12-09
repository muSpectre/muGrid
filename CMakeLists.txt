cmake_minimum_required(VERSION 3.18)

# Version discovery from git
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --dirty --always
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MUGRID_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MUGRID_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --quiet
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE MUGRID_GIT_DIRTY_RESULT
    )
    if(MUGRID_GIT_DIRTY_RESULT EQUAL 0)
        set(MUGRID_GIT_DIRTY "false")
    else()
        set(MUGRID_GIT_DIRTY "true")
    endif()
else()
    set(MUGRID_VERSION_STRING "unknown")
    set(MUGRID_GIT_HASH "unknown")
    set(MUGRID_GIT_DIRTY "false")
endif()

# Extract version number for CMake project
string(REGEX MATCH "^v?([0-9]+\\.[0-9]+\\.[0-9]+)" _version_match "${MUGRID_VERSION_STRING}")
if(_version_match)
    set(MUGRID_VERSION "${CMAKE_MATCH_1}")
else()
    set(MUGRID_VERSION "0.0.0")
endif()

project(muGrid
    VERSION ${MUGRID_VERSION}
    LANGUAGES C CXX
    DESCRIPTION "MPI-parallel regular grids"
)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(MUGRID_USE_MPI "Enable MPI support" ON)
option(MUGRID_USE_NETCDF "Enable NetCDF I/O" ON)
option(MUGRID_BUILD_PYTHON "Build Python bindings" ON)
option(MUGRID_BUILD_TESTS "Build tests" ON)
option(MUGRID_BUILD_EXAMPLES "Build examples" ON)

# Kokkos backend selection
set(MUGRID_KOKKOS_BACKEND "SERIAL" CACHE STRING "Kokkos execution backend")
set_property(CACHE MUGRID_KOKKOS_BACKEND PROPERTY STRINGS SERIAL OPENMP CUDA HIP)

# Set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find or fetch Eigen3
find_package(Eigen3 3.4.0 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    # Disable Eigen's tests and examples to avoid polluting our test suite
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
    # Restore BUILD_TESTING for muGrid's own tests
    set(BUILD_TESTING ON CACHE BOOL "" FORCE)
endif()

# Find or fetch Kokkos (required dependency)
find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found, fetching from GitHub...")
    # Enable the selected backend
    set(Kokkos_ENABLE_${MUGRID_KOKKOS_BACKEND} ON CACHE BOOL "" FORCE)
    # Disable Kokkos tests
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    # Enable PIC for shared library linking
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG 4.5.00
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(kokkos)
else()
    message(STATUS "Kokkos found: ${Kokkos_DIR}")
endif()

# MPI support
set(MUGRID_WITH_MPI OFF)
if(MUGRID_USE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        set(MUGRID_WITH_MPI ON)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
        message(STATUS "MPI not found, building without MPI support")
    endif()
endif()

# NetCDF support
set(MUGRID_WITH_NETCDF OFF)
if(MUGRID_USE_NETCDF)
    if(MUGRID_WITH_MPI)
        # Look for parallel NetCDF (PnetCDF)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PNETCDF pnetcdf QUIET)
        endif()
        if(PNETCDF_FOUND)
            set(MUGRID_WITH_NETCDF ON)
            set(MUGRID_NETCDF_LIBRARIES ${PNETCDF_LIBRARIES})
            set(MUGRID_NETCDF_INCLUDE_DIRS ${PNETCDF_INCLUDE_DIRS})
            message(STATUS "PnetCDF found")
        else()
            message(STATUS "PnetCDF not found, building without NetCDF support")
        endif()
    else()
        # Look for serial NetCDF
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(NETCDF netcdf QUIET)
        endif()
        if(NETCDF_FOUND)
            set(MUGRID_WITH_NETCDF ON)
            set(MUGRID_NETCDF_LIBRARIES ${NETCDF_LIBRARIES})
            set(MUGRID_NETCDF_INCLUDE_DIRS ${NETCDF_INCLUDE_DIRS})
            message(STATUS "NetCDF found")
        else()
            message(STATUS "NetCDF not found, building without NetCDF support")
        endif()
    endif()
endif()

# Python bindings dependencies
if(MUGRID_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Find or fetch pybind11
    find_package(pybind11 2.11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching from GitHub...")
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Print configuration summary
message(STATUS "--------------------")
message(STATUS "muGrid configuration")
message(STATUS "  Version        : ${MUGRID_VERSION_STRING}")
message(STATUS "  Kokkos backend : ${MUGRID_KOKKOS_BACKEND}")
message(STATUS "  MPI            : ${MUGRID_WITH_MPI}")
if(MUGRID_WITH_MPI)
    message(STATUS "  Parallel NetCDF: ${MUGRID_WITH_NETCDF}")
else()
    message(STATUS "  Unidata NetCDF : ${MUGRID_WITH_NETCDF}")
endif()
message(STATUS "  Python bindings: ${MUGRID_BUILD_PYTHON}")
message(STATUS "  Tests          : ${MUGRID_BUILD_TESTS}")
message(STATUS "  Examples       : ${MUGRID_BUILD_EXAMPLES}")
message(STATUS "--------------------")

# Add subdirectories
add_subdirectory(src)

if(MUGRID_BUILD_PYTHON)
    add_subdirectory(language_bindings)
endif()

if(MUGRID_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MUGRID_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install CMake config files for downstream projects
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/muGridConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/muGridConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/muGridConfig.cmake"
    INSTALL_DESTINATION lib/cmake/muGrid
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/muGridConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/muGridConfigVersion.cmake"
    DESTINATION lib/cmake/muGrid
)

install(EXPORT muGridTargets
    FILE muGridTargets.cmake
    NAMESPACE muGrid::
    DESTINATION lib/cmake/muGrid
)
