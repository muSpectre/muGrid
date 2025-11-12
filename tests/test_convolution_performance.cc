/**
 * @file   test_convolution_performance.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Nov 2025
 *
 * @brief  Performance tests for convolution operators using PAPI
 *
 * Copyright © 2025 Lars Pastewka
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
#include "libmugrid/convolution_operator.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field.hh"

#include <papi.h>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace muGrid {

  /**
   * @brief Helper class to manage PAPI initialization and cleanup
   */
  class PAPIManager {
  public:
    PAPIManager() {
      int retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT && retval > 0) {
        throw std::runtime_error("PAPI library version mismatch!");
      }
      if (retval < 0) {
        throw std::runtime_error("PAPI library initialization failed!");
      }

      // Check if thread support is available
      retval = PAPI_thread_init((unsigned long (*)(void))(pthread_self));
      if (retval != PAPI_OK && retval != PAPI_ECMP) {
        std::cerr << "Warning: PAPI thread support not available" << std::endl;
      }
    }

    ~PAPIManager() {
      PAPI_shutdown();
    }
  };

  /**
   * @brief Structure to hold performance metrics
   */
  struct PerformanceMetrics {
    long long total_cycles;
    long long total_instructions;
    long long fp_operations;
    long long l1_dcache_misses;
    long long l2_dcache_misses;
    long long l3_cache_misses;
    double wall_time_ms;

    // Derived metrics
    double ipc;  // Instructions per cycle
    double gflops;
    double l1_miss_rate;
    double l2_miss_rate;
    double l3_miss_rate;

    void compute_derived_metrics() {
      ipc = static_cast<double>(total_instructions) / total_cycles;
      gflops = (fp_operations / 1e9) / (wall_time_ms / 1000.0);

      // Compute cache miss rates relative to total memory accesses
      // (approximated by instructions, as PAPI may not have a direct counter)
      l1_miss_rate = static_cast<double>(l1_dcache_misses) / total_instructions * 100.0;
      l2_miss_rate = static_cast<double>(l2_dcache_misses) / total_instructions * 100.0;
      l3_miss_rate = static_cast<double>(l3_cache_misses) / total_instructions * 100.0;
    }

    void print(std::ostream& out) const {
      out << std::fixed << std::setprecision(2);
      out << "\n=== Performance Metrics ===\n";
      out << "Wall time:          " << wall_time_ms << " ms\n";
      out << "Total cycles:       " << total_cycles << "\n";
      out << "Total instructions: " << total_instructions << "\n";
      out << "IPC:                " << ipc << "\n";
      out << "FP operations:      " << fp_operations << "\n";
      out << "GFLOPS:             " << gflops << "\n";
      out << "L1 cache misses:    " << l1_dcache_misses
          << " (" << l1_miss_rate << "% of instructions)\n";
      out << "L2 cache misses:    " << l2_dcache_misses
          << " (" << l2_miss_rate << "% of instructions)\n";
      out << "L3 cache misses:    " << l3_cache_misses
          << " (" << l3_miss_rate << "% of instructions)\n";
      out << "===========================\n";
    }
  };

  /**
   * @brief RAII wrapper for PAPI event counting
   */
  class PAPIEventCounter {
  public:
    PAPIEventCounter(int* events, int num_events)
        : num_events_(num_events), event_set_(PAPI_NULL) {
      int retval = PAPI_create_eventset(&event_set_);
      if (retval != PAPI_OK) {
        throw std::runtime_error("Failed to create PAPI event set");
      }

      retval = PAPI_add_events(event_set_, events, num_events);
      if (retval != PAPI_OK) {
        std::cerr << "Warning: Could not add all PAPI events: "
                  << PAPI_strerror(retval) << std::endl;
        // Try adding events one by one to see which ones work
        PAPI_cleanup_eventset(event_set_);
        PAPI_destroy_eventset(&event_set_);
        PAPI_create_eventset(&event_set_);

        for (int i = 0; i < num_events; ++i) {
          retval = PAPI_add_event(event_set_, events[i]);
          if (retval != PAPI_OK) {
            char event_name[PAPI_MAX_STR_LEN];
            PAPI_event_code_to_name(events[i], event_name);
            std::cerr << "  Could not add event: " << event_name << std::endl;
          }
        }
      }

      values_.resize(num_events, 0);
    }

    ~PAPIEventCounter() {
      if (event_set_ != PAPI_NULL) {
        PAPI_cleanup_eventset(event_set_);
        PAPI_destroy_eventset(&event_set_);
      }
    }

    void start() {
      int retval = PAPI_start(event_set_);
      if (retval != PAPI_OK) {
        throw std::runtime_error("Failed to start PAPI counters");
      }
    }

    void stop() {
      int retval = PAPI_stop(event_set_, values_.data());
      if (retval != PAPI_OK) {
        throw std::runtime_error("Failed to stop PAPI counters");
      }
    }

    long long get_value(int index) const {
      return values_[index];
    }

  private:
    int num_events_;
    int event_set_;
    std::vector<long long> values_;
  };

  /**
   * @brief Measure performance of convolution operator
   */
  PerformanceMetrics measure_convolution_performance(
      const ConvolutionOperator& conv_op,
      const TypedFieldBase<Real>& nodal_field,
      TypedFieldBase<Real>& quad_field,
      int num_iterations = 10) {

    PerformanceMetrics metrics{};

    // Define PAPI events to monitor
    int events[] = {
      PAPI_TOT_CYC,   // Total cycles
      PAPI_TOT_INS,   // Total instructions
      PAPI_FP_OPS,    // Floating point operations
      PAPI_L1_DCM,    // L1 data cache misses
      PAPI_L2_DCM,    // L2 data cache misses
      PAPI_L3_TCM     // L3 total cache misses
    };
    int num_events = 6;

    try {
      PAPIEventCounter counter(events, num_events);

      // Warm-up iteration
      conv_op.apply(nodal_field, quad_field);

      // Start timing and counting
      auto start_time = std::chrono::high_resolution_clock::now();
      counter.start();

      // Perform multiple iterations
      for (int i = 0; i < num_iterations; ++i) {
        conv_op.apply(nodal_field, quad_field);
      }

      // Stop timing and counting
      counter.stop();
      auto end_time = std::chrono::high_resolution_clock::now();

      // Extract metrics
      metrics.total_cycles = counter.get_value(0);
      metrics.total_instructions = counter.get_value(1);
      metrics.fp_operations = counter.get_value(2);
      metrics.l1_dcache_misses = counter.get_value(3);
      metrics.l2_dcache_misses = counter.get_value(4);
      metrics.l3_cache_misses = counter.get_value(5);

      // Compute wall time in milliseconds
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);
      metrics.wall_time_ms = duration.count() / 1000.0;

      // Average over iterations
      metrics.total_cycles /= num_iterations;
      metrics.total_instructions /= num_iterations;
      metrics.fp_operations /= num_iterations;
      metrics.l1_dcache_misses /= num_iterations;
      metrics.l2_dcache_misses /= num_iterations;
      metrics.l3_cache_misses /= num_iterations;
      metrics.wall_time_ms /= num_iterations;

      metrics.compute_derived_metrics();

    } catch (const std::exception& e) {
      std::cerr << "Error during PAPI measurement: " << e.what() << std::endl;
      throw;
    }

    return metrics;
  }

  BOOST_AUTO_TEST_SUITE(convolution_performance);

  BOOST_AUTO_TEST_CASE(performance_2d_small_grid) {
    std::cout << "\n=== Testing 2D Convolution Performance (Small Grid) ===\n";

    // Initialize PAPI
    PAPIManager papi_manager;

    // Test parameters
    const Index_t nb_x_pts = 32;
    const Index_t nb_y_pts = 32;
    const Index_t nb_stencil_x = 3;
    const Index_t nb_stencil_y = 3;
    const Index_t nb_operators = 2;
    const Index_t nb_quad_pts = 4;
    const Index_t nb_field_components = 3;

    // Create a simple 3x3 stencil operator
    Eigen::MatrixXd stencil_mat(nb_operators * nb_quad_pts,
                                 nb_stencil_x * nb_stencil_y);
    stencil_mat.setRandom();

    Shape_t conv_pts_shape = {nb_stencil_x, nb_stencil_y};
    Shape_t pixel_offset = {-1, -1};  // Center stencil

    ConvolutionOperator conv_op(pixel_offset, stencil_mat, conv_pts_shape,
                                1, nb_quad_pts, nb_operators);

    // Create field collection
    GlobalFieldCollection fc({nb_x_pts, nb_y_pts},
                            std::map<std::string, Index_t>{{"quad", nb_quad_pts}});

    // Create fields
    auto& nodal_field = fc.register_real_field("nodal", nb_field_components);
    auto& quad_field = fc.register_real_field("quad",
                                               nb_field_components * nb_operators,
                                               "quad");

    // Initialize with random data
    nodal_field.eigen().setRandom();

    // Measure performance
    auto metrics = measure_convolution_performance(conv_op, nodal_field,
                                                   quad_field, 100);

    // Print results
    std::cout << "Grid size: " << nb_x_pts << " x " << nb_y_pts << "\n";
    std::cout << "Stencil size: " << nb_stencil_x << " x " << nb_stencil_y << "\n";
    std::cout << "Field components: " << nb_field_components << "\n";
    std::cout << "Operators: " << nb_operators << "\n";
    std::cout << "Quadrature points: " << nb_quad_pts << "\n";
    metrics.print(std::cout);

    // Basic sanity checks
    BOOST_CHECK_GT(metrics.fp_operations, 0);
    BOOST_CHECK_GT(metrics.total_cycles, 0);
    BOOST_CHECK_GT(metrics.total_instructions, 0);
  }

  BOOST_AUTO_TEST_CASE(performance_2d_large_grid) {
    std::cout << "\n=== Testing 2D Convolution Performance (Large Grid) ===\n";

    // Initialize PAPI
    PAPIManager papi_manager;

    // Test parameters - larger grid
    const Index_t nb_x_pts = 256;
    const Index_t nb_y_pts = 256;
    const Index_t nb_stencil_x = 3;
    const Index_t nb_stencil_y = 3;
    const Index_t nb_operators = 2;
    const Index_t nb_quad_pts = 4;
    const Index_t nb_field_components = 3;

    // Create a simple 3x3 stencil operator
    Eigen::MatrixXd stencil_mat(nb_operators * nb_quad_pts,
                                 nb_stencil_x * nb_stencil_y);
    stencil_mat.setRandom();

    Shape_t conv_pts_shape = {nb_stencil_x, nb_stencil_y};
    Shape_t pixel_offset = {-1, -1};

    ConvolutionOperator conv_op(pixel_offset, stencil_mat, conv_pts_shape,
                                1, nb_quad_pts, nb_operators);

    // Create field collection
    GlobalFieldCollection fc({nb_x_pts, nb_y_pts},
                            std::map<std::string, Index_t>{{"quad", nb_quad_pts}});

    // Create fields
    auto& nodal_field = fc.register_real_field("nodal", nb_field_components);
    auto& quad_field = fc.register_real_field("quad",
                                               nb_field_components * nb_operators,
                                               "quad");

    // Initialize with random data
    nodal_field.eigen().setRandom();

    // Measure performance with fewer iterations for large grid
    auto metrics = measure_convolution_performance(conv_op, nodal_field,
                                                   quad_field, 10);

    // Print results
    std::cout << "Grid size: " << nb_x_pts << " x " << nb_y_pts << "\n";
    std::cout << "Stencil size: " << nb_stencil_x << " x " << nb_stencil_y << "\n";
    std::cout << "Field components: " << nb_field_components << "\n";
    std::cout << "Operators: " << nb_operators << "\n";
    std::cout << "Quadrature points: " << nb_quad_pts << "\n";
    metrics.print(std::cout);

    // Basic sanity checks
    BOOST_CHECK_GT(metrics.fp_operations, 0);
    BOOST_CHECK_GT(metrics.gflops, 0.0);

    // Cache miss rates should be reasonable (< 50% for L1)
    BOOST_CHECK_LT(metrics.l1_miss_rate, 50.0);
  }

  BOOST_AUTO_TEST_CASE(performance_comparison_grid_sizes) {
    std::cout << "\n=== Comparing Performance Across Grid Sizes ===\n";

    // Initialize PAPI
    PAPIManager papi_manager;

    std::vector<Index_t> grid_sizes = {16, 32, 64, 128, 256};

    std::cout << std::setw(10) << "Grid Size"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "L1 Miss %"
              << std::setw(15) << "L2 Miss %"
              << std::setw(15) << "L3 Miss %"
              << std::setw(15) << "IPC" << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (auto grid_size : grid_sizes) {
      const Index_t nb_stencil = 3;
      const Index_t nb_operators = 2;
      const Index_t nb_quad_pts = 4;
      const Index_t nb_components = 3;

      Eigen::MatrixXd stencil_mat(nb_operators * nb_quad_pts,
                                   nb_stencil * nb_stencil);
      stencil_mat.setRandom();

      Shape_t conv_pts_shape = {nb_stencil, nb_stencil};
      Shape_t pixel_offset = {-1, -1};

      ConvolutionOperator conv_op(pixel_offset, stencil_mat, conv_pts_shape,
                                  1, nb_quad_pts, nb_operators);

      GlobalFieldCollection fc({grid_size, grid_size},
                              std::map<std::string, Index_t>{{"quad", nb_quad_pts}});

      auto& nodal_field = fc.register_real_field("nodal", nb_components);
      auto& quad_field = fc.register_real_field("quad",
                                                 nb_components * nb_operators,
                                                 "quad");

      nodal_field.eigen().setRandom();

      int iterations = (grid_size <= 64) ? 100 : 10;
      auto metrics = measure_convolution_performance(conv_op, nodal_field,
                                                     quad_field, iterations);

      std::cout << std::setw(10) << (grid_size * grid_size)
                << std::setw(15) << std::fixed << std::setprecision(3)
                << metrics.gflops
                << std::setw(15) << metrics.l1_miss_rate
                << std::setw(15) << metrics.l2_miss_rate
                << std::setw(15) << metrics.l3_miss_rate
                << std::setw(15) << metrics.ipc << "\n";
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
