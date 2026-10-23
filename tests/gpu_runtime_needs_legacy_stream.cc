/**
 * @file   gpu_runtime_needs_legacy_stream.cc
 *
 * @brief  Must NOT compile: built only by the test of the same name with a
 *         per-thread default stream, which gpu_runtime.hh refuses (#error)
 *         because muGrid's device exchanges rely on the legacy default
 *         stream.
 */
#include "memory/gpu_runtime.hh"

int main() { return 0; }
