set (systems_SRC
  system_base.cc
  )
find_package(FFTW REQUIRED)

add_library(systems ${systems_SRC})
target_link_libraries(systems Eigen3::Eigen common)
