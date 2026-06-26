/**
 * @file   test_allocation_profiler.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Tests for unified-memory detection and the allocation profiler
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

#include "tests.hh"

#include "collection/field_collection_global.hh"
#include "memory/allocation_profiler.hh"
#include "memory/device.hh"
#include "memory/memory_info.hh"
#include "memory/unified_memory.hh"

namespace muGrid {

    BOOST_AUTO_TEST_SUITE(allocation_profiler_test);

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(host_is_never_unified) {
        // A host device shares nothing to detect; it is never "unified".
        BOOST_CHECK(!Device::cpu().is_unified());
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(host_capacity_is_queryable) {
        // Best-effort, but Linux and macOS (the CI/dev platforms) report it.
#if defined(__linux__) || defined(__APPLE__)
        const auto cap{host_memory_capacity()};
        BOOST_CHECK(cap.valid);
        BOOST_CHECK_GT(cap.total, 0u);
#endif
    }

    /* ---------------------------------------------------------------------- */
    // Find the by-label record for `name` across all pools; null if absent.
    static const BufferReport * find_buffer(const AllocationReport & rep,
                                            const std::string & name) {
        for (const auto & pool : rep.pools) {
            for (const auto & buf : pool.buffers) {
                if (buf.label == name) {
                    return &buf;
                }
            }
        }
        return nullptr;
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(named_host_field_is_profiled) {
        auto & prof{AllocationProfiler::instance()};
        prof.reset();
        prof.enable();

        constexpr Index_t SDim{twoD};
        constexpr Index_t len{4};
        GlobalFieldCollection fc{SDim};
        fc.set_nb_sub_pts("quad", OneQuadPt);
        fc.register_real_field("profiled-scalar", 1, "quad");
        fc.register_real_field("profiled-vector", SDim, "quad");
        fc.initialise(CcoordOps::get_cube<SDim>(len),
                      CcoordOps::get_cube<SDim>(len), {});

        const auto rep{prof.report()};

        // The format helper must always be callable without throwing.
        BOOST_CHECK_NO_THROW((void)prof.format_report());

        const auto * scalar{find_buffer(rep, "profiled-scalar")};
        const auto * vector{find_buffer(rep, "profiled-vector")};
        BOOST_REQUIRE(scalar != nullptr);
        BOOST_REQUIRE(vector != nullptr);

        const std::size_t nb_pixels{len * len};
        const std::size_t scalar_bytes{nb_pixels * 1 * sizeof(Real)};
        const std::size_t vector_bytes{nb_pixels * SDim * sizeof(Real)};
        BOOST_CHECK_EQUAL(scalar->peak_bytes, scalar_bytes);
        BOOST_CHECK_EQUAL(vector->peak_bytes, vector_bytes);

        // Both host buffers live in one pool whose peak is at least their
        // combined footprint.
        BOOST_REQUIRE_EQUAL(rep.pools.size(), 1u);
        BOOST_CHECK_GE(rep.pools.front().peak, scalar_bytes + vector_bytes);
        BOOST_CHECK(!rep.pools.front().unified);

        prof.disable();
        prof.reset();
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(freed_buffers_drop_live_but_keep_peak) {
        auto & prof{AllocationProfiler::instance()};
        prof.reset();
        prof.enable();
        {
            constexpr Index_t SDim{twoD};
            constexpr Index_t len{8};
            GlobalFieldCollection fc{SDim};
            fc.set_nb_sub_pts("quad", OneQuadPt);
            fc.register_real_field("transient", 1, "quad");
            fc.initialise(CcoordOps::get_cube<SDim>(len),
                          CcoordOps::get_cube<SDim>(len), {});
            const auto live{prof.report()};
            BOOST_REQUIRE_EQUAL(live.pools.size(), 1u);
            BOOST_CHECK_GT(live.pools.front().current, 0u);
        }
        // Collection destroyed: live drops to zero, peak is retained.
        const auto after{prof.report()};
        BOOST_REQUIRE_EQUAL(after.pools.size(), 1u);
        BOOST_CHECK_EQUAL(after.pools.front().current, 0u);
        BOOST_CHECK_GT(after.pools.front().peak, 0u);
        prof.disable();
        prof.reset();
    }

    BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
