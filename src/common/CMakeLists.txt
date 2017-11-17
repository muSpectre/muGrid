set (common_SRC
  #units.cc
  common.cc
  voigt_conversion.cc
  )

add_library(common ${common_SRC})
target_link_libraries(common Eigen3::Eigen)

