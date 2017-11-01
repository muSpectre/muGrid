set (common_SRC
  #units.cc
  voigt_conversion.cc
  )

add_library(common ${common_SRC})
target_link_libraries(common Eigen3::Eigen)

