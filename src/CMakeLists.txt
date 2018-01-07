include_directories(
  common
  materials
  system
  external
  )

set (muSpectre_SRC
  )
add_subdirectory(common)
add_subdirectory(materials)
add_subdirectory(fft)
add_subdirectory(system)
add_subdirectory(solver)

find_package(FFTW REQUIRED)

add_library(muSpectre ${muSpectre_SRC})
target_link_libraries(muSpectre Eigen3::Eigen ${FFTW_LIBRARIES})
