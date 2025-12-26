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

# Windows-specific settings
if(WIN32)
    # Prevent windows.h from defining min/max macros that conflict with std::min/max
    add_compile_definitions(NOMINMAX)
    # Reduce windows.h include size
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

# GPU backend options
option(MUGRID_ENABLE_CUDA "Enable CUDA GPU support" OFF)
option(MUGRID_ENABLE_HIP "Enable HIP GPU support (AMD ROCm)" OFF)

# Enable CUDA language if requested
# This must be after project() but before the first CUDA target
if(MUGRID_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Set default CUDA architecture if not specified
    # 70 = Volta (V100), 75 = Turing, 80 = Ampere (A100), 90 = Hopper (H100)
    # Use "native" on CMake 3.24+ to auto-detect, or set a default
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;90" CACHE STRING "CUDA architectures to compile for")
    endif()
endif()

# Enable HIP language if requested
# This must be after project() but before the first HIP target
if(MUGRID_ENABLE_HIP)
    enable_language(HIP)
    # Set default HIP architecture if not specified
    # gfx906 = MI50/MI60, gfx908 = MI100, gfx90a = MI200 series, gfx942 = MI300 series
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a" CACHE STRING "HIP architectures to compile for")
    endif()
endif()

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(MUGRID_ENABLE_MPI "Enable MPI support" ON)
option(MUGRID_ENABLE_NETCDF "Enable NetCDF I/O" ON)
option(MUGRID_ENABLE_PYTHON "Enable Python bindings" ON)
option(MUGRID_ENABLE_TESTS "Enable tests" ON)
option(MUGRID_ENABLE_EXAMPLES "Enable examples" ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find or fetch Eigen3
find_package(Eigen3 5.0.1 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
        GIT_SHALLOW TRUE
    )
    # Disable Eigen's tests and examples to avoid polluting our test suite
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
    # Restore BUILD_TESTING for muGrid's own tests
    set(BUILD_TESTING ON CACHE BOOL "" FORCE)
    set(MUGRID_EIGEN_FROM_FETCHCONTENT TRUE)
else()
    set(MUGRID_EIGEN_FROM_FETCHCONTENT FALSE)
endif()

# MPI support
if(MUGRID_ENABLE_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
        set(MUGRID_ENABLE_MPI OFF)
        message(STATUS "MPI not found, building without MPI support")
    endif()
endif()

# NetCDF support
if(MUGRID_ENABLE_NETCDF)
    if(MUGRID_ENABLE_MPI)
        # Look for parallel NetCDF (PnetCDF)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PNETCDF pnetcdf QUIET)
        endif()
        if(PNETCDF_FOUND)
            set(MUGRID_NETCDF_LIBRARIES ${PNETCDF_LINK_LIBRARIES})
            set(MUGRID_NETCDF_INCLUDE_DIRS ${PNETCDF_INCLUDE_DIRS})
            message(STATUS "PnetCDF found")
        else()
            set(MUGRID_ENABLE_NETCDF OFF)
            message(STATUS "PnetCDF not found, building without NetCDF support")
        endif()
    else()
        # Look for serial NetCDF - try pkg-config first, then CMake find_package
        set(NETCDF_FOUND FALSE)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(NETCDF netcdf QUIET)
        endif()
        if(NETCDF_FOUND)
            set(MUGRID_NETCDF_LIBRARIES ${NETCDF_LINK_LIBRARIES})
            set(MUGRID_NETCDF_INCLUDE_DIRS ${NETCDF_INCLUDE_DIRS})
            message(STATUS "NetCDF found via pkg-config")
        else()
            # Fallback to CMake find_package (works on Windows with CMake config files)
            find_package(netCDF QUIET CONFIG)
            if(netCDF_FOUND)
                set(MUGRID_NETCDF_LIBRARIES netCDF::netcdf)
                # Extract include directories from the imported target for use by other targets
                get_target_property(_netcdf_include_dirs netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
                if(_netcdf_include_dirs)
                    set(MUGRID_NETCDF_INCLUDE_DIRS "${_netcdf_include_dirs}")
                else()
                    set(MUGRID_NETCDF_INCLUDE_DIRS "")
                endif()
                set(NETCDF_FOUND TRUE)
                message(STATUS "NetCDF found via CMake config")
            else()
                set(MUGRID_ENABLE_NETCDF OFF)
                message(STATUS "NetCDF not found, building without NetCDF support")
            endif()
        endif()
    endif()
endif()

# Python bindings dependencies
if(MUGRID_ENABLE_PYTHON)
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
        set(MUGRID_PYBIND11_FROM_FETCHCONTENT TRUE)
    else()
        set(MUGRID_PYBIND11_FROM_FETCHCONTENT FALSE)
    endif()

    # Fetch DLPack header for zero-copy array exchange
    message(STATUS "Fetching DLPack header...")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE INTERNAL "Set minimum policy version for DLPack header")
    # Disable DLPack's mock target (uses posix_memalign which doesn't exist on Windows)
    set(BUILD_MOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG v1.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(dlpack)
endif()

# Print configuration summary
message(STATUS "--------------------")
message(STATUS "muGrid configuration")
message(STATUS "  Version        : ${MUGRID_VERSION_STRING}")

# Eigen version
if(MUGRID_EIGEN_FROM_FETCHCONTENT)
    message(STATUS "  Eigen          : 5.0.1 (fetched)")
elseif(TARGET Eigen3::Eigen)
    get_target_property(_eigen_version Eigen3::Eigen INTERFACE_EIGEN_VERSION)
    if(_eigen_version)
        message(STATUS "  Eigen          : ${_eigen_version}")
    else()
        message(STATUS "  Eigen          : ${Eigen3_VERSION}")
    endif()
else()
    message(STATUS "  Eigen          : ${Eigen3_VERSION}")
endif()

# CUDA version
if(MUGRID_ENABLE_CUDA)
    message(STATUS "  CUDA           : ON - ${CUDAToolkit_VERSION}")
else()
    message(STATUS "  CUDA           : OFF")
endif()

# HIP version
if(MUGRID_ENABLE_HIP)
    message(STATUS "  HIP            : ON - ${hip_VERSION}")
else()
    message(STATUS "  HIP            : OFF")
endif()

# MPI version
if(MUGRID_ENABLE_MPI)
    if(MPI_CXX_VERSION)
        message(STATUS "  MPI            : ON - MPI ${MPI_CXX_VERSION}")
    else()
        message(STATUS "  MPI            : ON")
    endif()
else()
    message(STATUS "  MPI            : OFF")
endif()

# NetCDF version
if(MUGRID_ENABLE_NETCDF)
    if(MUGRID_ENABLE_MPI)
        if(PNETCDF_VERSION)
            message(STATUS "  Parallel NetCDF: ON - PnetCDF ${PNETCDF_VERSION}")
        else()
            message(STATUS "  Parallel NetCDF: ON")
        endif()
    else()
        if(NETCDF_VERSION)
            message(STATUS "  Unidata NetCDF : ON - NetCDF ${NETCDF_VERSION}")
        else()
            message(STATUS "  Unidata NetCDF : ON")
        endif()
    endif()
else()
    if(MUGRID_ENABLE_MPI)
        message(STATUS "  Parallel NetCDF: OFF")
    else()
        message(STATUS "  Unidata NetCDF : OFF")
    endif()
endif()

# Python bindings version
if(MUGRID_ENABLE_PYTHON)
    message(STATUS "  Python bindings: ON - Python ${Python_VERSION}")
    if(MUGRID_PYBIND11_FROM_FETCHCONTENT)
        message(STATUS "  pybind11       : 2.13.6 (fetched)")
    elseif(pybind11_VERSION)
        message(STATUS "  pybind11       : ${pybind11_VERSION}")
    endif()
else()
    message(STATUS "  Python bindings: OFF")
endif()

message(STATUS "  Tests          : ${MUGRID_ENABLE_TESTS}")
message(STATUS "  Examples       : ${MUGRID_ENABLE_EXAMPLES}")
message(STATUS "--------------------")

# Add subdirectories
add_subdirectory(src)

if(MUGRID_ENABLE_PYTHON)
    add_subdirectory(language_bindings)
endif()

if(MUGRID_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MUGRID_ENABLE_EXAMPLES)
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
