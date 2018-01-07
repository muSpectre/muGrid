set (materials_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/material_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_hyper_elastic1.cc
  #${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic.cc
  )

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${materials_SRC}
  PARENT_SCOPE)


