/**
 * @file   fft/gpu_fft_backend.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Backend-agnostic CRTP base for the GPU FFT backends (cuFFT, rocFFT)
 *
 * cuFFT and rocFFT differ only in API spelling for everything outside the
 * actual library call: the `batch == 0` empty-subdomain guard, the
 * complex-alignment check, the strided-real limitation, the plan cache (1D and
 * N-D) with its hash, the grow-only scratch buffer used to preserve the c2r_nd
 * input, the per-call device synchronisation, and the destructor cleanup. This
 * CRTP base owns all of that once. A concrete backend supplies only the
 * library calls through the hooks `make_plan`, `destroy_plan`, and
 * `exec_{r2c,c2r,c2c_forward,c2c_backward}`, plus the N-D entry points (whose
 * advanced-data-layout derivation is genuinely backend-specific).
 *
 * The base is templated on the concrete backend (`Derived`, CRTP) and on the
 * cached-plan type (`PlanT` — `cufftHandle` for cuFFT, a small struct for
 * rocFFT). It routes GPU runtime calls through `memory/gpu_runtime.hh` (the
 * shared CUDA/HIP shim), so no `#ifdef` lives here.
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
 *
 */

#ifndef SRC_LIBMUGRID_FFT_GPU_FFT_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_GPU_FFT_BACKEND_HH_

#include "fft_backend.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"
#include "memory/device_alloc.hh"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace muGrid {

/**
 * CRTP base for GPU FFT backends. `Derived` is the concrete backend; `PlanT`
 * is the type cached per plan (a `cufftHandle`, or a struct bundling the
 * rocFFT plan with its execution info and work buffer).
 *
 * `Derived` must provide (protected, and befriend this base):
 *   PlanT make_plan(const PlanKey & key);
 *   void  destroy_plan(PlanT & plan);
 *   void  exec_r2c(PlanT & plan, const Real * in, Complex * out);
 *   void  exec_c2r(PlanT & plan, const Complex * in, Real * out);
 *   void  exec_c2c_forward(PlanT & plan, const Complex * in, Complex * out);
 *   void  exec_c2c_backward(PlanT & plan, const Complex * in, Complex * out);
 *   const char * name() const;           // (FFT1DBackend pure virtual)
 *   r2c_nd / c2r_nd                       // (FFT1DBackend, backend-specific)
 * and, in its destructor, call `this->destroy_all_plans()` (see below).
 */
template <class Derived, class PlanT>
class GpuFFTBackend : public FFT1DBackend {
 public:
  bool supports_device_memory() const final { return true; }

  //! Both GPU backends transform whole multidimensional r2c/c2r natively, so
  //! the serial engine hands the entire transform over in one planned call.
  bool supports_nd() const final { return true; }

  void r2c(Index_t n, Index_t batch, const Real * input, Index_t in_stride,
           Index_t in_dist, Complex * output, Index_t out_stride,
           Index_t out_dist) final {
    // Ranks with an empty subdomain (possible when a grid dimension is smaller
    // than the process grid) have nothing to transform; the libraries reject
    // batch == 0 at plan creation.
    if (batch == 0) {
      return;
    }
    // The real array of an r2c transform is the input. cuFFT cannot stride it
    // (it overrides supports_strided_r2c() to false); rocFFT can.
    if (!this->supports_strided_r2c() && in_stride != 1) {
      throw RuntimeError(this->strided_error("real-to-complex (r2c)", in_stride));
    }
    this->check_complex_aligned(input, "r2c");
    PlanT & plan = this->plan_for(
        PlanKey{Kind::R2C, Direction::FORWARD, Precision::Double, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_r2c(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2r(Index_t n, Index_t batch, const Complex * input, Index_t in_stride,
           Index_t in_dist, Real * output, Index_t out_stride,
           Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    // The real array of a c2r transform is the output.
    if (!this->supports_strided_c2r() && out_stride != 1) {
      throw RuntimeError(
          this->strided_error("complex-to-real (c2r)", out_stride));
    }
    this->check_complex_aligned(output, "c2r");
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2R, Direction::BACKWARD, Precision::Double, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2r(plan, input, output);
    // Synchronize so the result is complete before the caller hands the buffer
    // to GPU-aware MPI (which is not ordered against the FFT stream).
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2c_forward(Index_t n, Index_t batch, const Complex * input,
                   Index_t in_stride, Index_t in_dist, Complex * output,
                   Index_t out_stride, Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2C, Direction::FORWARD, Precision::Double, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2c_forward(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2c_backward(Index_t n, Index_t batch, const Complex * input,
                    Index_t in_stride, Index_t in_dist, Complex * output,
                    Index_t out_stride, Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2C, Direction::BACKWARD, Precision::Double, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2c_backward(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  // --- Single-precision (Real32/Complex32) 1D primitives -------------------
  // Mirror the double overloads above but tag the plan key Precision::Single
  // (so float and double plans never share a cache slot) and dispatch to the
  // backend's fp32 exec hooks. The real-side alignment guard uses the fp32
  // complex size (8 bytes), since the engine pads fp32 real fields to that.

  void r2c(Index_t n, Index_t batch, const Real32 * input, Index_t in_stride,
           Index_t in_dist, Complex32 * output, Index_t out_stride,
           Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    if (!this->supports_strided_r2c() && in_stride != 1) {
      throw RuntimeError(
          this->strided_error("real-to-complex (r2c)", in_stride));
    }
    this->check_complex_aligned(input, "r2c", 2 * sizeof(Real32));
    PlanT & plan = this->plan_for(
        PlanKey{Kind::R2C, Direction::FORWARD, Precision::Single, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_r2c(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2r(Index_t n, Index_t batch, const Complex32 * input, Index_t in_stride,
           Index_t in_dist, Real32 * output, Index_t out_stride,
           Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    if (!this->supports_strided_c2r() && out_stride != 1) {
      throw RuntimeError(
          this->strided_error("complex-to-real (c2r)", out_stride));
    }
    this->check_complex_aligned(output, "c2r", 2 * sizeof(Real32));
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2R, Direction::BACKWARD, Precision::Single, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2r(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2c_forward(Index_t n, Index_t batch, const Complex32 * input,
                   Index_t in_stride, Index_t in_dist, Complex32 * output,
                   Index_t out_stride, Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2C, Direction::FORWARD, Precision::Single, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2c_forward(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

  void c2c_backward(Index_t n, Index_t batch, const Complex32 * input,
                    Index_t in_stride, Index_t in_dist, Complex32 * output,
                    Index_t out_stride, Index_t out_dist) final {
    if (batch == 0) {
      return;
    }
    PlanT & plan = this->plan_for(
        PlanKey{Kind::C2C, Direction::BACKWARD, Precision::Single, n, batch,
                in_stride, in_dist, out_stride, out_dist});
    this->derived().exec_c2c_backward(plan, input, output);
    GPU_STREAM_SYNCHRONIZE_DEFAULT();
  }

 protected:
  /** Transform kind for plan caching. */
  enum class Kind : int { R2C = 0, C2R = 1, C2C = 2 };

  /**
   * Direction for C2C transforms (R2C/C2R carry a fixed direction). cuFFT bakes
   * the direction into the *execution* call and shares one Z2Z plan across
   * directions, whereas rocFFT bakes it into the plan. Keying on the direction
   * is correct for both: rocFFT gets the two distinct plans it needs, and cuFFT
   * caches one direction-agnostic Z2Z handle per direction (a negligible extra
   * plan, created once).
   */
  enum class Direction : int { FORWARD = 0, BACKWARD = 1 };

  /**
   * Floating-point precision of a transform. Double and single plans never
   * share a cache slot, and the backend selects the matching library plan
   * type (rocfft_precision_*, CUFFT_{R2C,D2Z}…) from it.
   */
  enum class Precision : int { Double = 0, Single = 1 };

  /** Plan-cache key: every parameter that defines a 1D batched plan. */
  struct PlanKey {
    Kind kind;
    Direction direction;
    Precision precision;
    Index_t n, batch, in_stride, in_dist, out_stride, out_dist;
    bool operator==(const PlanKey & o) const {
      return kind == o.kind && direction == o.direction &&
             precision == o.precision && n == o.n && batch == o.batch &&
             in_stride == o.in_stride && in_dist == o.in_dist &&
             out_stride == o.out_stride && out_dist == o.out_dist;
    }
  };

  /** Hash for PlanKey (the boost-style 0x9e3779b9 combiner, written once). */
  struct PlanKeyHash {
    std::size_t operator()(const PlanKey & k) const {
      std::size_t result = std::hash<int>{}(static_cast<int>(k.kind));
      auto mix = [&result](std::size_t h) {
        result ^= h + 0x9e3779b9 + (result << 6) + (result >> 2);
      };
      mix(std::hash<int>{}(static_cast<int>(k.direction)));
      mix(std::hash<int>{}(static_cast<int>(k.precision)));
      mix(std::hash<Index_t>{}(k.n));
      mix(std::hash<Index_t>{}(k.batch));
      mix(std::hash<Index_t>{}(k.in_stride));
      mix(std::hash<Index_t>{}(k.in_dist));
      mix(std::hash<Index_t>{}(k.out_stride));
      mix(std::hash<Index_t>{}(k.out_dist));
      return result;
    }
  };

  /** Get or create the cached 1D plan for `key` (via Derived::make_plan). */
  PlanT & plan_for(const PlanKey & key) {
    auto it = this->plan_cache.find(key);
    if (it == this->plan_cache.end()) {
      it = this->plan_cache.emplace(key, this->derived().make_plan(key)).first;
    }
    return it->second;
  }

  /**
   * Get or create the cached N-D plan for the string signature `key`, building
   * it with `make` (a callable returning `PlanT`) on a miss. The N-D plan
   * parameters are backend-specific, so the caller forms both the key and the
   * builder; the cache and its lifetime are shared here.
   */
  template <class MakeFn>
  PlanT & nd_plan_for(const std::string & key, MakeFn && make) {
    auto it = this->nd_plan_cache.find(key);
    if (it == this->nd_plan_cache.end()) {
      it = this->nd_plan_cache.emplace(key, make()).first;
    }
    return it->second;
  }

  /**
   * Grow-only device scratch buffer of at least `bytes`. Used to stage the
   * Fourier input of c2r_nd, since the multidimensional real-inverse transform
   * overwrites its input but the engine requires it preserved. Never shrinks;
   * freed in the destructor.
   */
  void * ensure_scratch(std::size_t bytes) {
    if (bytes > this->scratch_bytes) {
      if (this->scratch != nullptr) {
        device_deallocate(this->scratch);
        this->scratch = nullptr;
        this->scratch_bytes = 0;
      }
      // Route through the device allocator chokepoint (single owner + visible
      // to the allocation profiler).
      this->scratch = device_allocate(bytes, "fft-nd-scratch");
      this->scratch_bytes = bytes;
    }
    return this->scratch;
  }

  /**
   * Throw unless `ptr` is aligned like a complex value (16 bytes). muGrid pads
   * FFT-engine real-space fields so the real array of every real<->complex
   * transform satisfies this by construction (see the FFTEngineBase
   * constructor); a violation indicates a layout-padding bug or a field not
   * allocated through a muGrid FFT engine.
   */
  void check_complex_aligned(const void * ptr, const char * operation,
                             std::size_t complex_align = 2 * sizeof(Real))
      const {
    // The error branch is unreachable through the public API (the layout
    // invariant guarantees alignment), so it is excluded from coverage.
    // GCOVR_EXCL_START
    if (reinterpret_cast<std::uintptr_t>(ptr) % complex_align != 0) {
      throw RuntimeError(
          std::string("The real array passed to the ") + this->name() + " " +
          operation + " transform is not aligned to the complex type (" +
          std::to_string(complex_align) +
          " bytes). muGrid "
          "pads the layout of FFT engine real-space fields so that this cannot "
          "happen; this error indicates a bug in muGrid's layout padding, or a "
          "field whose memory was not allocated through a muGrid FFT engine. "
          "External arrays must be copied into an engine field before "
          "transforming.");
    }
    // GCOVR_EXCL_STOP
  }

  /**
   * Destroy every cached plan via Derived::destroy_plan and clear the caches.
   * MUST be called from the *derived* destructor: CRTP forbids dispatching to
   * a derived hook once the derived subobject has been destroyed, so the base
   * destructor cannot do this itself (it only frees the backend-agnostic
   * scratch buffer).
   */
  void destroy_all_plans() {
    for (auto & entry : this->plan_cache) {
      this->derived().destroy_plan(entry.second);
    }
    for (auto & entry : this->nd_plan_cache) {
      this->derived().destroy_plan(entry.second);
    }
    this->plan_cache.clear();
    this->nd_plan_cache.clear();
  }

  ~GpuFFTBackend() override {
    // Plans are released by the derived destructor (destroy_all_plans); only
    // the backend-agnostic scratch remains to free here.
    if (this->scratch != nullptr) {
      device_deallocate(this->scratch);
    }
  }

  //! Plan cache (1D batched transforms)
  std::unordered_map<PlanKey, PlanT, PlanKeyHash> plan_cache;

  //! Plan cache for N-dimensional transforms, keyed by a string signature
  std::unordered_map<std::string, PlanT> nd_plan_cache;

  //! Grow-only scratch for c2r_nd input preservation
  void * scratch{nullptr};
  std::size_t scratch_bytes{0};

 private:
  Derived & derived() { return static_cast<Derived &>(*this); }
  const Derived & derived() const {
    return static_cast<const Derived &>(*this);
  }

  std::string strided_error(const char * kind, Index_t stride) const {
    return std::string(this->name()) + " does not support strided " + kind +
           " transforms (the real-side stride is " + std::to_string(stride) +
           " but must be 1). This limitation affects 3D MPI-parallel FFTs on "
           "this GPU backend; use the CPU FFT backend for that case, or a 2D "
           "grid that transforms with unit stride.";
  }
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_GPU_FFT_BACKEND_HH_
