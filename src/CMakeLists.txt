include_directories(
  common
  materials
  system
  external
  )

add_subdirectory(common)
add_subdirectory(materials)
add_subdirectory(fft)
add_subdirectory(system)
add_subdirectory(solver)
