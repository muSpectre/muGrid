set (common_SRC
  #${CMAKE_CURRENT_SOURCE_DIR}/units.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/voigt_conversion.cc
  )

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${common_SRC}
  PARENT_SCOPE)
