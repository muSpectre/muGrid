set (fft_engine_SRC
  fft_engine_base.cc
  fftw_engine.cc
  projection_finite_strain.cc
  )

find_package(FFTW REQUIRED)
add_library(fft ${fft_engine_SRC})
target_link_libraries(fft Eigen3::Eigen common ${FFTW_LIBRARIES})
