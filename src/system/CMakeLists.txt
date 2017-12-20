set (systems_SRC
  system_base.cc
  )

add_library(systems ${systems_SRC})
target_link_libraries(systems Eigen3::Eigen common)
