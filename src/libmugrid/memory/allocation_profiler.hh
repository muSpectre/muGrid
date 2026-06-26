/**
 * @file   memory/allocation_profiler.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Optional profiling of Field buffer allocations
 *
 * A process-wide singleton that, when enabled, records every Field buffer
 * allocation and deallocation and reports — on demand, typically at the end
 * of a run — the average and peak memory footprint per memory pool, together
 * with the names of the allocated buffers and how much memory is physically
 * available in each pool.
 *
 * Buffers are grouped by memory pool. Host and a given device are normally
 * separate pools, but on a physically unified-memory architecture (e.g. an
 * MI300A APU) they share one pool and are reported jointly; this is detected
 * via Device::is_host_accessible() (see memory/device.hh).
 *
 * The instrumentation in Array (which calls record_alloc/record_free) is
 * always compiled in. Recording is off by default and gated by an atomic
 * flag, so when disabled the per-allocation cost is a single relaxed atomic
 * load. Enable it with enable(), or set the environment variable
 * MUGRID_PROFILE_ALLOCATIONS=1 to start recording automatically.
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

#ifndef SRC_LIBMUGRID_MEMORY_ALLOCATION_PROFILER_HH_
#define SRC_LIBMUGRID_MEMORY_ALLOCATION_PROFILER_HH_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory/memory_info.hh"

namespace muGrid {

    /**
     * Per-buffer line in a profiling report: a named buffer, where it was
     * requested ("cpu"/"cuda:0"/...), and its current and peak footprint.
     */
    struct BufferReport {
        std::string label;          //!< buffer name (Field name or "<unnamed>")
        std::string space;          //!< requesting space: "cpu"/"cuda:N"/...
        std::size_t current_bytes;  //!< bytes currently live
        std::size_t peak_bytes;     //!< high-water mark for this label
    };

    /**
     * Profiling report for one memory pool.
     */
    struct PoolReport {
        std::string name;        //!< "host", "cuda:0", "unified", ...
        bool unified;            //!< host and device share this physical pool
        std::size_t current;     //!< bytes currently live in the pool
        std::size_t peak;        //!< peak bytes ever simultaneously live
        double average;          //!< time-weighted average live bytes
        MemoryCapacity capacity; //!< physical capacity of the pool
        std::vector<BufferReport> buffers;  //!< by-label, sorted by peak desc
    };

    //! Full profiling report: one entry per memory pool.
    struct AllocationReport {
        std::vector<PoolReport> pools;
        double elapsed_seconds;  //!< length of the measurement window
    };

    /**
     * Process-wide allocation profiler. Access via instance(). Thread-safe.
     */
    class AllocationProfiler {
       public:
        static AllocationProfiler & instance();

        //! Start recording. Resets the measurement window if not already on.
        void enable();
        //! Stop recording (retains accumulated statistics).
        void disable();
        //! Discard all statistics and restart the measurement window.
        void reset();
        //! Whether recording is currently active.
        bool is_enabled() const;

        /**
         * Record an allocation. @p handle is the unique buffer pointer (used
         * to match the matching free); @p label is the buffer name; @p space
         * is the requesting memory space ("cpu", "cuda:0", ...).
         */
        void record_alloc(const void * handle, const std::string & label,
                           const std::string & space, std::size_t bytes);

        //! Record the deallocation of a previously recorded @p handle.
        void record_free(const void * handle);

        //! Snapshot the current statistics as a structured report.
        AllocationReport report() const;

        //! Render report() as a human-readable multi-line string.
        std::string format_report() const;

       private:
        AllocationProfiler() = default;

        using Clock = std::chrono::steady_clock;

        //! Live record for one allocated buffer.
        struct Live {
            std::string pool;   //!< normalized pool key
            std::string label;  //!< buffer name
            std::string space;  //!< requesting space
            std::size_t bytes;
        };

        //! Aggregated footprint of all buffers sharing one label in a pool.
        struct LabelStat {
            std::string space;
            std::size_t current{0};
            std::size_t peak{0};
        };

        //! Time-integrated statistics for one memory pool.
        struct PoolStat {
            bool unified{false};
            std::size_t current{0};
            std::size_t peak{0};
            double byte_seconds{0.0};
            Clock::time_point last_update{};
            std::map<std::string, LabelStat> labels{};
        };

        //! Map a requesting space string to a normalized pool key, applying
        //! the host+device merge on unified-memory systems.
        static std::string pool_key_for(const std::string & space,
                                         bool & unified_out);

        //! Advance a pool's time integral up to @p now.
        void accrue(PoolStat & pool, Clock::time_point now) const;

        mutable std::mutex mutex{};
        //! Recording gate. Atomic so the per-allocation fast path can skip
        //! the mutex entirely when recording is off.
        std::atomic<bool> enabled{false};
        bool started{false};
        Clock::time_point start{};
        std::unordered_map<const void *, Live> live{};
        std::map<std::string, PoolStat> pools{};
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_ALLOCATION_PROFILER_HH_
