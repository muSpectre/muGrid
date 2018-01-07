set (fft_engine_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engine_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fftw_engine.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_finite_strain.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_finite_strain_fast.cc
  )

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${fft_engine_SRC}
  PARENT_SCOPE)
