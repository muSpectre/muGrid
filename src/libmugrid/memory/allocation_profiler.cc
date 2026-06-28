/**
 * @file   memory/allocation_profiler.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Optional profiling of Field buffer allocations
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

#include "memory/allocation_profiler.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "memory/device.hh"

namespace muGrid {

    namespace {
        //! Parse the device id from a space string such as "cuda:2"; returns
        //! 0 if there is no ":N" suffix.
        int device_id_from_space(const std::string & space) {
            const auto colon{space.find(':')};
            if (colon == std::string::npos) {
                return 0;
            }
            try {
                return std::stoi(space.substr(colon + 1));
            } catch (...) {
                return 0;
            }
        }

        bool is_host_space_string(const std::string & space) {
            return space.rfind("cpu", 0) == 0 || space.rfind("host", 0) == 0 ||
                   space.rfind("Host", 0) == 0;
        }

        //! True if device @p device_id shares one physical memory pool with
        //! the host (integrated GPU / APU) — i.e. it is a device whose memory
        //! is host-accessible. On a host-only build Device::gpu() is the CPU,
        //! which is not a device, so this is false.
        bool device_is_unified(int device_id) {
            const Device dev{Device::gpu(device_id)};
            return dev.is_device() && dev.is_host_accessible();
        }

        //! Format a byte count with binary units.
        std::string human_bytes(double bytes) {
            const char * units[]{"B", "KiB", "MiB", "GiB", "TiB"};
            int u{0};
            while (bytes >= 1024.0 && u < 4) {
                bytes /= 1024.0;
                ++u;
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.2f %s", bytes, units[u]);
            return std::string{buf};
        }
    }  // namespace

    AllocationProfiler & AllocationProfiler::instance() {
        // Intentionally leaked (never destroyed): Array buffers belonging to
        // globally/statically-scoped field collections may be freed during
        // program teardown and call record_free(); a function-local static
        // would already be destroyed by then. The leak is a single small
        // object and is the same shutdown-safety pattern used elsewhere
        // (see bind_py_device.cc).
        static AllocationProfiler * profiler{[] {
            auto * p{new AllocationProfiler{}};
            // Opt-in via the environment, mirroring MUGRID_GPU_AWARE_MPI.
            const char * env{std::getenv("MUGRID_PROFILE_ALLOCATIONS")};
            if (env != nullptr && std::strlen(env) > 0 &&
                (env[0] == '1' || env[0] == 'y' || env[0] == 'Y' ||
                 env[0] == 't' || env[0] == 'T')) {
                p->enable();
            }
            return p;
        }()};
        return *profiler;
    }

    void AllocationProfiler::enable() {
        std::lock_guard<std::mutex> lock{this->mutex};
        if (!this->enabled) {
            this->enabled = true;
            if (!this->started) {
                this->start = Clock::now();
                this->started = true;
            }
        }
    }

    void AllocationProfiler::disable() {
        std::lock_guard<std::mutex> lock{this->mutex};
        this->enabled = false;
    }

    void AllocationProfiler::reset() {
        std::lock_guard<std::mutex> lock{this->mutex};
        this->live.clear();
        this->pools.clear();
        this->start = Clock::now();
        this->started = true;
    }

    bool AllocationProfiler::is_enabled() const {
        return this->enabled.load(std::memory_order_relaxed);
    }

    std::string AllocationProfiler::pool_key_for(const std::string & space,
                                                 bool & unified_out) {
        unified_out = false;
        if (is_host_space_string(space)) {
            // A host allocation shares the device's pool only when that device
            // is physically unified with the host. We probe device 0, the
            // common single-APU case. (Multiple distinct unified devices would
            // each be their own physical pool; that rare case is not split.)
            if (device_is_unified(0)) {
                unified_out = true;
                return "unified";
            }
            return "host";
        }
        const int id{device_id_from_space(space)};
        if (device_is_unified(id)) {
            unified_out = true;
            return "unified";
        }
        return space;  // e.g. "cuda:0"
    }

    void AllocationProfiler::accrue(PoolStat & pool, Clock::time_point now) const {
        const std::chrono::duration<double> dt{now - pool.last_update};
        pool.byte_seconds += static_cast<double>(pool.current) * dt.count();
        pool.last_update = now;
    }

    void AllocationProfiler::record_alloc(const void * handle,
                                          const std::string & label,
                                          const std::string & space,
                                          std::size_t bytes) {
        if (handle == nullptr || bytes == 0) {
            return;
        }
        // Fast path: when recording is off, a single relaxed atomic load and
        // no lock. (A benign race with a concurrent disable() at worst records
        // one extra allocation.)
        if (!this->enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> lock{this->mutex};
        const auto now{Clock::now()};
        // Idempotent re-label: if this handle is already recorded (e.g. it was
        // recorded generically at the device_allocate chokepoint and is now
        // being re-recorded by the owning Field under its real name), undo the
        // previous accounting first, so the bytes move to the new label rather
        // than being counted twice.
        if (this->live.find(handle) != this->live.end()) {
            this->remove_locked(handle, now);
        }
        bool unified{false};
        const std::string pk{pool_key_for(space, unified)};

        auto [it, inserted] = this->pools.try_emplace(pk);
        PoolStat & pool{it->second};
        if (inserted) {
            pool.last_update = this->start;
            pool.unified = unified;
        }
        this->accrue(pool, now);
        pool.current += bytes;
        pool.peak = std::max(pool.peak, pool.current);

        LabelStat & ls{pool.labels[label]};
        ls.space = space;
        ls.current += bytes;
        ls.peak = std::max(ls.peak, ls.current);

        this->live[handle] = Live{pk, label, space, bytes};
    }

    void AllocationProfiler::record_free(const void * handle) {
        if (handle == nullptr) {
            return;
        }
        // Same fast path as record_alloc. disable() therefore freezes the
        // snapshot; use reset() to start a clean measurement window.
        if (!this->enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> lock{this->mutex};
        this->remove_locked(handle, Clock::now());
    }

    void AllocationProfiler::remove_locked(const void * handle,
                                           Clock::time_point now) {
        auto it{this->live.find(handle)};
        if (it == this->live.end()) {
            return;
        }
        const Live & rec{it->second};
        auto pit{this->pools.find(rec.pool)};
        if (pit != this->pools.end()) {
            PoolStat & pool{pit->second};
            this->accrue(pool, now);
            pool.current =
                rec.bytes <= pool.current ? pool.current - rec.bytes : 0;
            auto lsit{pool.labels.find(rec.label)};
            if (lsit != pool.labels.end()) {
                LabelStat & ls{lsit->second};
                ls.current = rec.bytes <= ls.current ? ls.current - rec.bytes : 0;
            }
        }
        this->live.erase(it);
    }

    AllocationReport AllocationProfiler::report() const {
        std::lock_guard<std::mutex> lock{this->mutex};
        const auto now{Clock::now()};
        const std::chrono::duration<double> elapsed{now - this->start};

        AllocationReport out{};
        out.elapsed_seconds = this->started ? elapsed.count() : 0.0;

        for (const auto & [pk, pool] : this->pools) {
            PoolReport pr{};
            pr.name = pk;
            pr.unified = pool.unified;
            pr.current = pool.current;
            pr.peak = pool.peak;

            // Finalize the time integral up to now.
            const std::chrono::duration<double> dt{now - pool.last_update};
            const double byte_seconds{
                pool.byte_seconds +
                static_cast<double>(pool.current) * dt.count()};
            pr.average = out.elapsed_seconds > 0.0
                             ? byte_seconds / out.elapsed_seconds
                             : static_cast<double>(pool.current);

            // Physical capacity of the pool.
            if (pk == "unified") {
                pr.capacity = device_memory_capacity(0);
                if (!pr.capacity.valid) {
                    pr.capacity = host_memory_capacity();
                }
            } else if (is_host_space_string(pk)) {
                pr.capacity = host_memory_capacity();
            } else {
                pr.capacity = device_memory_capacity(device_id_from_space(pk));
            }

            for (const auto & [label, ls] : pool.labels) {
                pr.buffers.push_back(
                    BufferReport{label, ls.space, ls.current, ls.peak});
            }
            std::sort(pr.buffers.begin(), pr.buffers.end(),
                      [](const BufferReport & a, const BufferReport & b) {
                          return a.peak_bytes > b.peak_bytes;
                      });
            out.pools.push_back(std::move(pr));
        }
        return out;
    }

    std::string AllocationProfiler::format_report() const {
        const AllocationReport rep{this->report()};
        std::ostringstream os;
        os << "=== muGrid Field allocation profile ===\n";
        char win[64];
        std::snprintf(win, sizeof(win), "  measurement window: %.3f s\n",
                      rep.elapsed_seconds);
        os << win;
        if (rep.pools.empty()) {
            os << "  no allocations recorded\n";
            return os.str();
        }
        for (const auto & pool : rep.pools) {
            os << "\n[" << pool.name << "]";
            if (pool.unified) {
                os << " (host+device share one physical pool)";
            }
            os << "\n";
            os << "  peak    " << human_bytes(static_cast<double>(pool.peak))
               << "   avg " << human_bytes(pool.average) << "   live "
               << human_bytes(static_cast<double>(pool.current));
            if (pool.capacity.valid) {
                os << "   |  of " << human_bytes(static_cast<double>(
                                         pool.capacity.total))
                   << " (avail "
                   << human_bytes(static_cast<double>(pool.capacity.available))
                   << ")";
                if (pool.capacity.total > 0) {
                    char pct[32];
                    std::snprintf(
                        pct, sizeof(pct), "  %.1f%% peak",
                        100.0 * static_cast<double>(pool.peak) /
                            static_cast<double>(pool.capacity.total));
                    os << pct;
                }
            }
            os << "\n";
            for (const auto & buf : pool.buffers) {
                os << "     " << human_bytes(static_cast<double>(buf.peak_bytes))
                   << "  " << buf.label;
                if (pool.unified) {
                    os << "  (" << buf.space << ")";
                }
                os << "\n";
            }
        }
        return os.str();
    }

}  // namespace muGrid
