set (systems_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/system_base.cc
  )

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${systems_SRC}
  PARENT_SCOPE)
