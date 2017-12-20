set (solvers_SRC
  solver_base.cc
  solver_cg.cc
  )

add_library(solvers ${solvers_SRC})
target_link_libraries(solvers Eigen3::Eigen common systems)
