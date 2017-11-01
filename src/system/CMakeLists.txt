set (systems_SRC
  system_base.cc
  #fftw_engine_c2c.cc
  #fftw_engine_r2c.cc
  )
find_package(FFTW REQUIRED)

add_library(systems ${systems_SRC})
target_link_libraries(systems Eigen3::Eigen common ${FFTW_LIBRARIES})
