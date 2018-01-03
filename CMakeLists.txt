cmake_minimum_required(VERSION 3.7.0)
project(µSpectre_prototype)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)

add_compile_options(-Wall -Wextra -Weffc++)



enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  string( TOLOWER "${CMAKE_BUILD_TYPE}" build_type )
  if ("release" STREQUAL "${build_type}" )
    add_compile_options(-march=native)
  endif()
  if ("debug" STREQUAL "${build_type}" )
    add_compile_options(-O0)
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
  tests
  src
  ${EIGEN3_INCLUDE_DIRS}
  )

#find_package(MPI REQUIRED)
find_package(FFTW REQUIRED)
#find_package(FFTWMPI REQUIRED)


#build tests
file( GLOB TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/test_*.cc")

add_executable(main_test_suite tests/main_test_suite.cc ${TEST_SRCS})
target_link_libraries(main_test_suite ${Boost_LIBRARIES} Eigen3::Eigen materials common systems fft solvers)
add_test(main_test_suite main_test_suite --report_level=detailed --build_info=TRUE)
#add_test(main_test_suite main_test_suite --build_info=TRUE)



# compile the library
add_compile_options( -Werror)
add_subdirectory(
  "${CMAKE_SOURCE_DIR}/src/"
  )


#compile executables
file( GLOB binaries "${CMAKE_SOURCE_DIR}/bin/*.cc")
foreach(binaryfile ${binaries})
  get_filename_component(binaryname  ${binaryfile} NAME_WE)
  add_executable(${binaryname} ${binaryfile})
  target_link_libraries(${binaryname} ${Boost_LIBRARIES} Eigen3::Eigen materials common systems fft solvers)
endforeach(binaryfile ${binaries})


# compile benchmarks

file(GLOB benchmarks "${CMAKE_SOURCE_DIR}/benchmarks/benchmark*cc")
foreach(benchmark ${benchmarks})
  get_filename_component(benchmark_name ${benchmark} NAME_WE)
  add_executable(${benchmark_name} ${benchmark})
  target_link_libraries(${benchmark_name} ${BOOST_LIBRARIES} Eigen3::Eigen common systems solvers)
endforeach(benchmark ${benchmark})
