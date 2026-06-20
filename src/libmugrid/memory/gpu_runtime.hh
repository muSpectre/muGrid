/**
 * @file   memory/gpu_runtime.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Jun 2026
 *
 * @brief  Single shared CUDA/HIP portability shim
 *
 * All GPU sources include this header instead of defining their own
 * backend macros. The rule: backend-specific blocks may differ only in
 * API spelling; no control flow or arithmetic lives inside the #ifdef
 * branches. Kernels themselves are single-source `__global__` functions.
 *
 * Copyright © 2026 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_LIBMUGRID_MEMORY_GPU_RUNTIME_HH_
#define SRC_LIBMUGRID_MEMORY_GPU_RUNTIME_HH_

#include <string>

#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)

#include <cuda_runtime.h>

using gpuStream_t = cudaStream_t;

// Kernel launches (only valid in CUDA/HIP translation units).
//
// Every launch is immediately followed by a check of the pending runtime
// error: a `<<<>>>` launch reports configuration and "no kernel image for
// device" failures synchronously through cudaGetLastError(). Without this
// check those failures are swallowed — e.g. a library built for the wrong
// GPU architecture launches nothing, runtime-API calls (cudaMemcpy/memset)
// keep working, and reductions silently return 0. Mirrors the manual check
// already used by the stiffness operators; see gpu_assert_last_error below.
#define GPU_LAUNCH_KERNEL(kernel, grid, block, ...)         \
    do {                                                    \
        kernel<<<grid, block>>>(__VA_ARGS__);               \
        ::muGrid::gpu_assert_last_error(#kernel);           \
    } while (0)
#define GPU_LAUNCH_KERNEL_SHMEM(kernel, grid, block, shmem, ...) \
    do {                                                         \
        kernel<<<grid, block, shmem>>>(__VA_ARGS__);             \
        ::muGrid::gpu_assert_last_error(#kernel);                \
    } while (0)
#define GPU_LAUNCH_KERNEL_STREAM(kernel, grid, block, stream, ...) \
    do {                                                           \
        kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);           \
        ::muGrid::gpu_assert_last_error(#kernel);                  \
    } while (0)

// Host-callable runtime API (valid in any translation unit)
#define GPU_DEVICE_SYNCHRONIZE() (void)cudaDeviceSynchronize()
#define GPU_MALLOC(ptr, size) (void)cudaMalloc(ptr, size)
#define GPU_FREE(ptr) (void)cudaFree(ptr)
#define GPU_MEMSET(ptr, value, size) (void)cudaMemset(ptr, value, size)
#define GPU_MEMSET_2D(ptr, pitch, value, width, height) \
    (void)cudaMemset2D(ptr, pitch, value, width, height)
#define GPU_MEMCPY_D2H(dst, src, size) \
    (void)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define GPU_MEMCPY_H2D(dst, src, size) \
    (void)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define GPU_MEMCPY_D2D(dst, src, size) \
    (void)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)
#define GPU_MEMCPY_2D_D2D(dst, dpitch, src, spitch, width, height) \
    (void)cudaMemcpy2D(dst, dpitch, src, spitch, width, height, \
                       cudaMemcpyDeviceToDevice)
#define GPU_MEMCPY_TO_SYMBOL(symbol, src, size) \
    (void)cudaMemcpyToSymbol(symbol, src, size)

namespace muGrid {
    //! nullptr if the last GPU runtime call succeeded, else the error string
    inline const char * gpu_last_error() {
        cudaError_t err{cudaGetLastError()};
        return err == cudaSuccess ? nullptr : cudaGetErrorString(err);
    }
}  // namespace muGrid

#elif defined(MUGRID_ENABLE_HIP)

#include <hip/hip_runtime.h>

using gpuStream_t = hipStream_t;

// See the CUDA branch above: each launch is checked immediately so that a
// wrong-architecture build (or a bad launch configuration) raises instead of
// silently doing nothing.
#define GPU_LAUNCH_KERNEL(kernel, grid, block, ...)                  \
    do {                                                             \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__);  \
        ::muGrid::gpu_assert_last_error(#kernel);                    \
    } while (0)
#define GPU_LAUNCH_KERNEL_SHMEM(kernel, grid, block, shmem, ...)         \
    do {                                                                \
        hipLaunchKernelGGL(kernel, grid, block, shmem, 0, __VA_ARGS__); \
        ::muGrid::gpu_assert_last_error(#kernel);                       \
    } while (0)
#define GPU_LAUNCH_KERNEL_STREAM(kernel, grid, block, stream, ...)        \
    do {                                                                 \
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, __VA_ARGS__); \
        ::muGrid::gpu_assert_last_error(#kernel);                        \
    } while (0)

#define GPU_DEVICE_SYNCHRONIZE() (void)hipDeviceSynchronize()
#define GPU_MALLOC(ptr, size) (void)hipMalloc(ptr, size)
#define GPU_FREE(ptr) (void)hipFree(ptr)
#define GPU_MEMSET(ptr, value, size) (void)hipMemset(ptr, value, size)
#define GPU_MEMSET_2D(ptr, pitch, value, width, height) \
    (void)hipMemset2D(ptr, pitch, value, width, height)
#define GPU_MEMCPY_D2H(dst, src, size) \
    (void)hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)
#define GPU_MEMCPY_H2D(dst, src, size) \
    (void)hipMemcpy(dst, src, size, hipMemcpyHostToDevice)
#define GPU_MEMCPY_D2D(dst, src, size) \
    (void)hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice)
#define GPU_MEMCPY_2D_D2D(dst, dpitch, src, spitch, width, height) \
    (void)hipMemcpy2D(dst, dpitch, src, spitch, width, height, \
                      hipMemcpyDeviceToDevice)
#define GPU_MEMCPY_TO_SYMBOL(symbol, src, size) \
    (void)hipMemcpyToSymbol(symbol, src, size)

namespace muGrid {
    //! nullptr if the last GPU runtime call succeeded, else the error string
    inline const char * gpu_last_error() {
        hipError_t err{hipGetLastError()};
        return err == hipSuccess ? nullptr : hipGetErrorString(err);
    }
}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA / MUGRID_ENABLE_HIP

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
namespace muGrid {
    /**
     * Throw a RuntimeError if a GPU kernel launch left a pending runtime error.
     *
     * `context` is typically the launched kernel's name (the launch macros pass
     * it automatically). This converts an otherwise silent launch failure —
     * most commonly "no kernel image for device" / "the provided PTX was
     * compiled with an unsupported toolchain" when the library is built for a
     * GPU architecture the device does not support — into an actionable error.
     */
    inline void gpu_assert_last_error(const char * context) {
        const char * err{gpu_last_error()};
        if (err != nullptr) {
            throw RuntimeError(std::string("GPU kernel launch failed (") +
                               context + "): " + err);
        }
    }
}  // namespace muGrid
#endif  // MUGRID_ENABLE_CUDA / MUGRID_ENABLE_HIP

#endif  // SRC_LIBMUGRID_MEMORY_GPU_RUNTIME_HH_
