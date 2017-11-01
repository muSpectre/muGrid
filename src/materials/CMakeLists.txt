set (materials_SRC
  material.cc
  #material_hyper_elastic.cc
  #material_linear_elastic.cc
  )

add_library(materials ${materials_SRC})
target_link_libraries(materials Eigen3::Eigen common)

