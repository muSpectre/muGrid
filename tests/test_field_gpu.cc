/**
 * @file   test_field_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   09 Dec 2024
 *
 * @brief  Testing GPU (device) field functionality
 *
 * Copyright © 2024 Lars Pastewka
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

#include "tests.hh"
#include "field/field_typed.hh"
#include "collection/field_collection_global.hh"
#include "memory/memory_space.hh"
#include "util/math.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(field_gpu_test);

  /* ---------------------------------------------------------------------- */
  /* Test device type traits */
  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(device_type_traits) {
    // Test is_host_space trait
    BOOST_CHECK(is_host_space_v<HostSpace>);
    BOOST_CHECK(!is_device_space_v<HostSpace>);

    // Test DLPack device type for host
    BOOST_CHECK_EQUAL(dlpack_device_type_v<HostSpace>, DLPackDeviceType::CPU);

#if defined(MUGRID_ENABLE_CUDA)
    // Test is_device_space trait for CUDA
    BOOST_CHECK(!is_host_space_v<CudaSpace>);
    BOOST_CHECK(is_device_space_v<CudaSpace>);

    // Test DLPack device type for CUDA
    BOOST_CHECK_EQUAL(dlpack_device_type_v<CudaSpace>, DLPackDeviceType::CUDA);
#endif

#if defined(MUGRID_ENABLE_HIP)
    // Test is_device_space trait for HIP/ROCm
    BOOST_CHECK(!is_host_space_v<HIPSpace>);
    BOOST_CHECK(is_device_space_v<HIPSpace>);

    // Test DLPack device type for HIP/ROCm
    BOOST_CHECK_EQUAL(dlpack_device_type_v<HIPSpace>, DLPackDeviceType::ROCm);
#endif

    // Test device_name function
    BOOST_CHECK_EQUAL(std::string(device_name<HostSpace>()), "cpu");

#if defined(MUGRID_ENABLE_CUDA)
    BOOST_CHECK_EQUAL(std::string(device_name<CudaSpace>()), "cuda");
#endif

#if defined(MUGRID_ENABLE_HIP)
    BOOST_CHECK_EQUAL(std::string(device_name<HIPSpace>()), "rocm");
#endif
  }

  /* ---------------------------------------------------------------------- */
  /* Test host field device introspection */
  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(host_field_device_info) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    GlobalFieldCollection fc{SDim};
    fc.initialise(CcoordOps::get_cube<SDim>(len),
                  CcoordOps::get_cube<SDim>(len), {});

    auto & field{fc.register_real_field("test_field", 3)};

    // Host field should report as NOT on device
    BOOST_CHECK(!field.is_on_device());

    // Host field should report CPU device type
    BOOST_CHECK_EQUAL(field.get_dlpack_device_type(), DLPackDeviceType::CPU);

    // Host field should have device ID 0
    BOOST_CHECK_EQUAL(field.get_device_id(), 0);

    // Host field should report "cpu" device string
    BOOST_CHECK_EQUAL(field.get_device_string(), "cpu");
  }

  /* ---------------------------------------------------------------------- */
  /* Test device field creation (only when GPU backend is enabled) */
  /* ---------------------------------------------------------------------- */
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
  BOOST_AUTO_TEST_CASE(device_field_creation) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    // Create a device collection
    GlobalFieldCollection fc{SDim, {}, StorageOrder::ArrayOfStructures,
                             FieldCollection::MemoryLocation::Device};
    fc.initialise(CcoordOps::get_cube<SDim>(len),
                  CcoordOps::get_cube<SDim>(len), {});

    // Collection should report as on device
    BOOST_CHECK(fc.is_on_device());

    // Create a field on the device collection - it should be a device field
    auto & device_field{fc.register_real_field("device_test", 3)};

    // Device field should report as on device
    BOOST_CHECK(device_field.is_on_device());

    // Device field should have correct device ID
    BOOST_CHECK_EQUAL(device_field.get_device_id(), 0);

    // Check correct DLPack device type
#if defined(MUGRID_ENABLE_CUDA)
    BOOST_CHECK_EQUAL(device_field.get_dlpack_device_type(),
                      DLPackDeviceType::CUDA);
    BOOST_CHECK_EQUAL(device_field.get_device_string(), "cuda:0");
#elif defined(MUGRID_ENABLE_HIP)
    BOOST_CHECK_EQUAL(device_field.get_dlpack_device_type(),
                      DLPackDeviceType::ROCm);
    BOOST_CHECK_EQUAL(device_field.get_device_string(), "rocm:0");
#endif

    // Check field has correct size
    BOOST_CHECK_EQUAL(device_field.get_nb_entries(), ipow(len, SDim));
    BOOST_CHECK_EQUAL(device_field.get_nb_components(), 3);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(device_field_set_zero) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    GlobalFieldCollection fc{SDim, {}, StorageOrder::ArrayOfStructures,
                             FieldCollection::MemoryLocation::Device};
    fc.initialise(CcoordOps::get_cube<SDim>(len),
                  CcoordOps::get_cube<SDim>(len), {});

    auto & device_field{fc.register_real_field("device_zero_test", 2)};

    // set_zero should work on device fields
    BOOST_CHECK_NO_THROW(device_field.set_zero());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(host_device_deep_copy) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    // Create host and device collections
    GlobalFieldCollection fc_host{SDim};
    fc_host.initialise(CcoordOps::get_cube<SDim>(len),
                       CcoordOps::get_cube<SDim>(len), {});

    GlobalFieldCollection fc_device{SDim, {}, StorageOrder::ArrayOfStructures,
                                    FieldCollection::MemoryLocation::Device};
    fc_device.initialise(CcoordOps::get_cube<SDim>(len),
                         CcoordOps::get_cube<SDim>(len), {});

    // Cast to typed fields to access deep_copy_from() methods
    auto & host_field{dynamic_cast<RealField &>(
        fc_host.register_real_field("host_src", 2))};
    auto & device_field{dynamic_cast<RealFieldDevice &>(
        fc_device.register_real_field("device_dst", 2))};

    // Initialize host field with some values
    host_field.set_zero();
    auto * host_data = host_field.data();
    for (size_t i = 0; i < host_field.get_buffer_size(); ++i) {
      host_data[i] = static_cast<Real>(i);
    }

    // Deep copy from host to device
    device_field.deep_copy_from(host_field);

    // Create another host field and copy back
    auto & host_check{dynamic_cast<RealField &>(
        fc_host.register_real_field("host_check", 2))};
    host_check.deep_copy_from(device_field);

    // Check that values match
    auto * check_data = host_check.data();
    bool all_match = true;
    for (size_t i = 0; i < host_field.get_buffer_size(); ++i) {
      if (host_data[i] != check_data[i]) {
        all_match = false;
        break;
      }
    }
    BOOST_CHECK(all_match);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(device_field_complex) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    GlobalFieldCollection fc{SDim, {}, StorageOrder::ArrayOfStructures,
                             FieldCollection::MemoryLocation::Device};
    fc.initialise(CcoordOps::get_cube<SDim>(len),
                  CcoordOps::get_cube<SDim>(len), {});

    auto & complex_device{fc.register_complex_field("complex_device", 2)};

    BOOST_CHECK(complex_device.is_on_device());
    BOOST_CHECK_EQUAL(complex_device.get_nb_components(), 2);
    BOOST_CHECK_NO_THROW(complex_device.set_zero());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(device_field_integer_types) {
    constexpr Index_t SDim{twoD};
    constexpr Index_t len{4};

    GlobalFieldCollection fc{SDim, {}, StorageOrder::ArrayOfStructures,
                             FieldCollection::MemoryLocation::Device};
    fc.initialise(CcoordOps::get_cube<SDim>(len),
                  CcoordOps::get_cube<SDim>(len), {});

    auto & int_device{fc.register_int_field("int_device", 1)};
    auto & uint_device{fc.register_uint_field("uint_device", 1)};

    BOOST_CHECK(int_device.is_on_device());
    BOOST_CHECK(uint_device.is_on_device());
    BOOST_CHECK_NO_THROW(int_device.set_zero());
    BOOST_CHECK_NO_THROW(uint_device.set_zero());
  }
#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

  /* ---------------------------------------------------------------------- */
  /* Test that DefaultDeviceSpace resolves correctly */
  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(default_device_space_resolution) {
#if defined(MUGRID_ENABLE_CUDA)
    // When CUDA is enabled, DefaultDeviceSpace should be CudaSpace
    bool is_cuda = std::is_same_v<DefaultDeviceSpace, CudaSpace>;
    BOOST_CHECK(is_cuda);
#elif defined(MUGRID_ENABLE_HIP)
    // When HIP is enabled, DefaultDeviceSpace should be HIPSpace
    bool is_hip = std::is_same_v<DefaultDeviceSpace, HIPSpace>;
    BOOST_CHECK(is_hip);
#else
    // Without GPU backend, DefaultDeviceSpace is HostSpace
    bool is_host = std::is_same_v<DefaultDeviceSpace, HostSpace>;
    BOOST_CHECK(is_host);
#endif
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
