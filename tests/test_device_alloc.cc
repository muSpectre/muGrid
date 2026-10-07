/**
 * @file   test_device_alloc.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   07 Oct 2026
 *
 * @brief  Tests for the external device allocator registration
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

#include <cstdlib>
#include <vector>

#include "core/exception.hh"
#include "memory/device_alloc.hh"

namespace muGrid {

    BOOST_AUTO_TEST_SUITE(device_alloc_test);

    namespace {
        //! Host-backed fake allocator recording what was called on its ctx.
        //! device_allocate() only touches the GPU runtime on the default
        //! path, so this runs in CPU-only builds too.
        struct FakeAllocator {
            int nb_allocs{0};
            int nb_frees{0};
            int nb_releases{0};

            DeviceAllocator registration() {
                DeviceAllocator a{};
                a.allocate = [](void * ctx, std::size_t bytes) -> void * {
                    ++static_cast<FakeAllocator *>(ctx)->nb_allocs;
                    return std::malloc(bytes);
                };
                a.deallocate = [](void * ctx, void * ptr) {
                    ++static_cast<FakeAllocator *>(ctx)->nb_frees;
                    std::free(ptr);
                };
                a.release = [](void * ctx) {
                    ++static_cast<FakeAllocator *>(ctx)->nb_releases;
                };
                a.ctx = this;
                return a;
            }
        };

        //! Restore the default allocator even if a check throws.
        struct ClearAllocatorGuard {
            ~ClearAllocatorGuard() { clear_device_allocator(); }
        };
    }  // namespace

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(rejects_incomplete_allocator) {
        DeviceAllocator a{};
        BOOST_CHECK_THROW(set_device_allocator(a), RuntimeError);
        BOOST_CHECK(!device_allocator_is_external());
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(frees_go_to_the_producing_allocator) {
        ClearAllocatorGuard guard{};
        FakeAllocator first{}, second{};

        set_device_allocator(first.registration());
        BOOST_CHECK(device_allocator_is_external());
        void * p1{device_allocate(64)};
        BOOST_CHECK_EQUAL(first.nb_allocs, 1);

        // Replacing the allocator must neither free nor release `first`
        // while p1 is alive.
        set_device_allocator(second.registration());
        void * p2{device_allocate(64)};
        BOOST_CHECK_EQUAL(second.nb_allocs, 1);
        BOOST_CHECK_EQUAL(first.nb_releases, 0);

        device_deallocate(p1);
        BOOST_CHECK_EQUAL(first.nb_frees, 1);
        BOOST_CHECK_EQUAL(second.nb_frees, 0);
        // Replaced and its last pointer gone: released exactly once.
        BOOST_CHECK_EQUAL(first.nb_releases, 1);

        device_deallocate(p2);
        BOOST_CHECK_EQUAL(second.nb_frees, 1);
        // Still current, so not released.
        BOOST_CHECK_EQUAL(second.nb_releases, 0);
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(clear_keeps_live_allocations_valid) {
        FakeAllocator fake{};
        set_device_allocator(fake.registration());
        std::vector<void *> ptrs{};
        for (int i{0}; i < 3; ++i) {
            ptrs.push_back(device_allocate(32));
        }

        clear_device_allocator();
        BOOST_CHECK(!device_allocator_is_external());
        BOOST_CHECK_EQUAL(fake.nb_releases, 0);

        for (auto * p : ptrs) {
            device_deallocate(p);
        }
        BOOST_CHECK_EQUAL(fake.nb_frees, 3);
        BOOST_CHECK_EQUAL(fake.nb_releases, 1);
    }

    /* ---------------------------------------------------------------------- */
    BOOST_AUTO_TEST_CASE(unused_allocator_released_on_clear) {
        FakeAllocator fake{};
        set_device_allocator(fake.registration());
        clear_device_allocator();
        BOOST_CHECK_EQUAL(fake.nb_releases, 1);
    }

    BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
