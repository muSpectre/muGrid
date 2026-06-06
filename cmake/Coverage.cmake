# Coverage.cmake
#
# Code-coverage instrumentation for muGrid.
#
# Enabling the cache option MUGRID_ENABLE_COVERAGE adds the compiler/linker
# flags required to collect line and branch coverage to every target that
# links against the INTERFACE target `mugrid::coverage`.  The muGrid library
# and the C++ test executables pick this up automatically (see the respective
# CMakeLists.txt).
#
# Convenience targets (created only when a coverage report generator is found):
#
#   coverage-xml     Cobertura XML report (build/coverage.xml), for CI upload
#   coverage-html    Browsable HTML report (build/coverage_html/index.html)
#   coverage-summary Plain-text summary printed to the console
#   coverage         Alias that builds all of the above
#
# Typical usage:
#
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMUGRID_ENABLE_COVERAGE=ON
#   cmake --build build
#   ctest --test-dir build              # or run pytest for the Python layer
#   cmake --build build --target coverage
#
# Coverage requires a non-optimised build for accurate line mapping; configuring
# with CMAKE_BUILD_TYPE=Debug is strongly recommended.

option(MUGRID_ENABLE_COVERAGE "Instrument the build for code coverage" OFF)

# Always provide the interface target so that targets can unconditionally link
# against it.  When coverage is disabled it simply carries no flags.
add_library(mugrid_coverage INTERFACE)
add_library(mugrid::coverage ALIAS mugrid_coverage)

if(NOT MUGRID_ENABLE_COVERAGE)
    return()
endif()

if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    message(WARNING
        "MUGRID_ENABLE_COVERAGE is only supported with GCC or Clang; "
        "ignoring for compiler '${CMAKE_CXX_COMPILER_ID}'.")
    return()
endif()

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING
        "Code coverage is being collected with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "Optimised builds produce misleading line/branch coverage; prefer "
        "-DCMAKE_BUILD_TYPE=Debug.")
endif()

message(STATUS "  Coverage       : ON")

# --coverage expands to -fprofile-arcs -ftest-coverage at compile time and
# -lgcov (or the Clang equivalent) at link time.  -O0 -g keep the line mapping
# faithful, and disabling inlining/elision avoids attributing code to the wrong
# lines.
#
# The flags are applied per source language. Plain C++ sources get them
# directly; CUDA sources are compiled by nvcc, which only understands
# host-compiler flags when they are forwarded with -Xcompiler. Device kernels
# themselves are NOT instrumented (gcov is host-only), but the host-side launch
# and dispatch code in the *_gpu translation units is.
set(_cov_cxx_flags --coverage -O0 -g -fno-inline)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC-only inlining knobs; Clang/hipcc reject these.
    list(APPEND _cov_cxx_flags -fno-inline-small-functions -fno-default-inline)
endif()

target_compile_options(mugrid_coverage INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${_cov_cxx_flags}>
)

if(MUGRID_ENABLE_CUDA)
    set(_cov_cuda_flags "")
    foreach(_flag IN LISTS _cov_cxx_flags)
        list(APPEND _cov_cuda_flags "-Xcompiler=${_flag}")
    endforeach()
    target_compile_options(mugrid_coverage INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:${_cov_cuda_flags}>
    )
endif()

if(MUGRID_ENABLE_HIP)
    # hipcc is Clang-based and accepts these flags directly.
    target_compile_options(mugrid_coverage INTERFACE
        $<$<COMPILE_LANGUAGE:HIP>:--coverage;-O0;-g;-fno-inline>
    )
endif()

target_link_options(mugrid_coverage INTERFACE --coverage)

# ---------------------------------------------------------------------------
# Report-generation targets (gcovr).
# ---------------------------------------------------------------------------
find_program(GCOVR_EXECUTABLE gcovr)

if(NOT GCOVR_EXECUTABLE)
    message(STATUS
        "gcovr not found; coverage data will still be produced (*.gcda) but the "
        "'coverage' convenience targets are unavailable. Install with "
        "'pip install gcovr'.")
    return()
endif()

# Pick the gcov tool that matches the compiler so the .gcno/.gcda versions line
# up (important when several GCC/Clang versions coexist).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_gcov_tool "llvm-cov gcov")
else()
    set(_gcov_tool "gcov")
endif()

# Only the library itself is interesting; exclude tests, fetched dependencies
# and generated files from the report.
set(_gcovr_common
    --root "${CMAKE_SOURCE_DIR}"
    --gcov-executable "${_gcov_tool}"
    # Run gcov serially: parallel workers race on the temporary .gcov files they
    # emit for shared system headers, which intermittently aborts the report.
    -j 1
    --filter "${CMAKE_SOURCE_DIR}/src/libmugrid/"
    --exclude ".*\\.skeleton"
    # pocketfft is vendored third-party code (Max-Planck-Society); we only
    # exercise the entry points we call, so its internals would otherwise drag
    # the headline numbers down without reflecting muGrid's own test quality.
    --exclude ".*/fft/pocketfft/.*"
    --exclude-unreachable-branches
    --exclude-throw-branches
    # Under MPI, several ranks merge counters into the same .gcda files; the
    # merged hit counts can trip gcov's "suspicious hit count" heuristic. Treat
    # those as warnings rather than fatal errors (a no-op for serial builds).
    --gcov-ignore-parse-errors=suspicious_hits.warn
    --print-summary
)

add_custom_target(coverage-xml
    COMMAND ${GCOVR_EXECUTABLE} ${_gcovr_common}
            --xml-pretty --output "${CMAKE_BINARY_DIR}/coverage.xml"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating Cobertura XML coverage report (coverage.xml)"
    VERBATIM
)

add_custom_target(coverage-html
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/coverage_html"
    COMMAND ${GCOVR_EXECUTABLE} ${_gcovr_common}
            --html-details "${CMAKE_BINARY_DIR}/coverage_html/index.html"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating HTML coverage report (coverage_html/index.html)"
    VERBATIM
)

add_custom_target(coverage-summary
    COMMAND ${GCOVR_EXECUTABLE} ${_gcovr_common}
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Printing coverage summary"
    VERBATIM
)

# Each gcovr invocation emits temporary .gcov files into its working directory;
# concurrent runs (which Ninja would otherwise launch for the aggregate target)
# race on those shared filenames. Chain the report targets so they run serially.
add_dependencies(coverage-html coverage-xml)
add_dependencies(coverage-summary coverage-html)

add_custom_target(coverage
    DEPENDS coverage-summary
    COMMENT "Generating all coverage reports (XML, HTML, summary)"
)
