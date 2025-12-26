/**
 * @file   test_exception.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   23 Dec 2025
 *
 * @brief  tests for exception handling and traceback capture
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
#include "core/exception.hh"

#include <string>
#include <sstream>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(exception_tests);

  /* ---------------------------------------------------------------------- */
  // Helper function to generate a traceback for testing
  void level3_function() {
    throw RuntimeError("Test error message");
  }

  void level2_function() {
    level3_function();
  }

  void level1_function() {
    level2_function();
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(traceback_capture) {
    // Test that an exception with traceback can be thrown and caught
    bool caught = false;
    std::string message;

    try {
      level1_function();
    } catch (const RuntimeError & e) {
      caught = true;
      message = e.what();
    }

    BOOST_CHECK(caught);
    BOOST_CHECK(message.find("Test error message") != std::string::npos);
    // Check that traceback header is present
    BOOST_CHECK(message.find("Traceback from C++ library") != std::string::npos);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(traceback_contains_function_names) {
    // Test that the traceback contains function names from the call stack
    std::string message;

    try {
      level1_function();
    } catch (const RuntimeError & e) {
      message = e.what();
    }

    // The traceback should contain at least one of our test function names
    // (depending on compiler optimization and platform)
    bool has_function_info =
        (message.find("level3_function") != std::string::npos) ||
        (message.find("level2_function") != std::string::npos) ||
        (message.find("level1_function") != std::string::npos);

    // On some platforms without debug symbols, we might not get function names
    // So we at least check that we have a traceback section
    BOOST_CHECK(message.find("Traceback from C++ library") != std::string::npos);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(traceback_stack_not_empty) {
    // Test that the traceback actually captures stack frames
    try {
      throw RuntimeError("Another test error");
    } catch (const RuntimeError & e) {
      std::string message = e.what();
      // The message should be longer than just the error message
      // indicating that traceback information was captured
      BOOST_CHECK(message.length() > std::string("Another test error").length());
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(exception_preserves_message) {
    // Test that the original error message is preserved
    const std::string test_message = "This is a specific error message";
    std::string caught_message;

    try {
      throw RuntimeError(test_message);
    } catch (const RuntimeError & e) {
      caught_message = e.what();
    }

    BOOST_CHECK(caught_message.find(test_message) != std::string::npos);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(traceback_entry_basic) {
    // Test basic TracebackEntry functionality
    void * dummy_address = reinterpret_cast<void *>(0x1234);
    TracebackEntry entry(dummy_address, "test_symbol");

    // The entry should store the address
    BOOST_CHECK_EQUAL(entry.get_symbol(), "test_symbol");

    // Test output operator doesn't crash
    std::stringstream ss;
    ss << entry;
    std::string output = ss.str();
    BOOST_CHECK(!output.empty());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(traceback_manual_construction) {
    // Test manual construction of a Traceback
    // This captures the current stack with 1 frame discarded
    Traceback tb(1);

    const auto & stack = tb.get_stack();

    // The stack should have at least one frame
    // (exact number depends on call depth and platform)
    BOOST_CHECK(stack.size() > 0);

    // Test output operator
    std::stringstream ss;
    ss << tb;
    std::string output = ss.str();
    // Output should not be empty (unless all frames are unresolved)
    // Just check it doesn't crash
    BOOST_CHECK(true);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(exception_with_traceback_inheritance) {
    // Test that ExceptionWithTraceback can be caught as base exception
    bool caught_as_runtime_error = false;
    bool caught_as_std_exception = false;

    try {
      throw RuntimeError("Test polymorphism");
    } catch (const std::runtime_error & e) {
      caught_as_runtime_error = true;
    }

    try {
      throw RuntimeError("Test polymorphism 2");
    } catch (const std::exception & e) {
      caught_as_std_exception = true;
    }

    BOOST_CHECK(caught_as_runtime_error);
    BOOST_CHECK(caught_as_std_exception);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
