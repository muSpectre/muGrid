set (materials_SRC
  material_base.cc
  #material_hyper_elastic1.cc
  #material_linear_elastic.cc
  )

add_library(materials ${materials_SRC})
target_link_libraries(materials Eigen3::Eigen common)

