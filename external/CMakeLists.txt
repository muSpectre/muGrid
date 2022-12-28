if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

add_library(cxxopts INTERFACE)
target_sources(cxxopts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cxxopts.hpp)
set_property(TARGET cxxopts APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

if(${SPLIT_CELL})
  include_directories(
          "./cork/src/accel"
          "./cork/src/file_formats"
          "./cork/src/isct"
          "./cork/src/math"
          "./cork/src/mpz"
          "./cork/src/mesh"
          "./cork/src/rawmesh"
          "./cork/src/util"
          "./cork/src/tetgen")

  add_subdirectory("./cork/src/tetgen")

  file(GLOB cork_SRC
          "./cork/src/*.cpp"
          "./cork/src/*.h"
          "./cork/src/*.cc"
          "./cork/src/*.hh"
          "./cork/src/accel/*.cpp"
          "./cork/src/accel/*.h"
          "./cork/src/file_formats/*.cpp"
          "./cork/src/file_formats/*.h"
          "./cork/src/isct/*.cpp"
          "./cork/src/isct/*.c"
          "./cork/src/isct/*.cc"
          "./cork/src/isct/*.h"
          "./cork/src/isct/*.hh"
          "./cork/src/math/*.cpp"
          "./cork/src/math/*.h"
          "./cork/src/mesh/*.cpp"
          "./cork/src/mesh/*.tpp"
          "./cork/src/mesh/*.h"
          "./cork/src/rawmesh/*.cpp"
          "./cork/src/rawmesh/*.tpp"
          "./cork/src/rawmesh/*.h"
          "./cork/src/util/*.cpp"
          "./cork/src/util/*.h"
          "./cork/src/tetgen/*.h"
          "./cork/src/tetgen/*.c"
          "./cork/src/tetgen/*.hh"
          "./cork/src/tetgen/*.cc")

  file(GLOB public_headers
        "./cork/src/*.h")

  add_definitions(
        -D_USE_MATH_DEFINES
        -DNOMINMAX
        -DNO_TIMER
        -DREDUCED
        -DCDT_ONLY
        -DTRILIBRARY
        -DANSI_DECLARATORS)

  set_source_files_properties(
        ${cork_SRC}
        PROPERTIES
        COMPILE_FLAGS "-Wno-deprecated")

  add_library(corkpp STATIC ${cork_SRC})
  set_property(TARGET corkpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(corkpp PUBLIC Eigen3::Eigen gmp tetgen)
endif()