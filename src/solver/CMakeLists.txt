set (solvers_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_cg.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/solvers.cc
  )

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${solvers_SRC}
  PARENT_SCOPE)
