/**
 * @file   exception.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   04 Feb 2020
 *
 * @brief  exception class for libmuGrid that collect a stack trace
 *
 * Copyright © 2017 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_LIBMUGRID_EXCEPTION_HH_
#define SRC_LIBMUGRID_EXCEPTION_HH_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace muGrid {

  /**
   * @class TracebackEntry
   * @brief A class that represents an entry from the stack traceback.
   *
   * This class is used to manage individual entries in a stack traceback. It
   * provides methods to get the symbol, name, and file associated with the
   * entry, and to check if the entry has been successfully resolved to a
   * function/method name.
   */
  class TracebackEntry {
   public:
    /**
     * @brief Construct a new TracebackEntry object with a given address and
     * symbol.
     *
     * @param address The address of the stack frame.
     * @param symbol The symbol associated with the stack frame.
     */
    TracebackEntry(void * address, const std::string & symbol);

    /**
     * @brief Construct a new TracebackEntry object with a given address and
     * symbol.
     *
     * @param address The address of the stack frame.
     * @param symbol The symbol associated with the stack frame.
     */
    TracebackEntry(void * address, const char * symbol);

    /**
     * @brief Copy constructor for the TracebackEntry class.
     *
     * @param other The TracebackEntry object to copy from.
     */
    TracebackEntry(const TracebackEntry & other);

    /**
     * @brief Destroy the TracebackEntry object.
     */
    ~TracebackEntry();

    /**
     * @brief Assignment operator for the TracebackEntry class.
     *
     * @param other The TracebackEntry object to assign from.
     * @return TracebackEntry& A reference to the assigned object.
     */
    TracebackEntry & operator=(const TracebackEntry & other);

    /**
     * @brief Get the symbol associated with the stack frame.
     *
     * @return const std::string& The symbol associated with the stack frame.
     */
    const std::string & get_symbol() const { return this->symbol; }

    /**
     * @brief Get the name associated with the stack frame.
     *
     * @return const std::string& The name associated with the stack frame.
     */
    const std::string & get_name() const { return this->name; }

    /**
     * @brief Get the file associated with the stack frame.
     *
     * @return const std::string& The file associated with the stack frame.
     */
    const std::string & get_file() const { return this->file; }

    /**
     * @brief Check if the stack frame has been successfully resolved to a
     * function/method name.
     *
     * @return bool True if the stack frame has been successfully resolved,
     * false otherwise.
     */
    bool is_resolved() const { return this->resolved; }

    /**
     * @brief Output the TracebackEntry object.
     *
     * This function outputs the TracebackEntry object to the provided output
     * stream. If the stack frame has been successfully resolved, it outputs the
     * file and name associated with the stack frame. Otherwise, it outputs a
     * message indicating that the stack frame could not be resolved to a
     * function/method name.
     *
     * The output is formatted like Python stack tracebacks, because its
     * primary purpose is to be displayed in conjunction with a Python
     * exception.
     *
     * @param os The output stream to output the TracebackEntry object to.
     * @param self The TracebackEntry object to output.
     * @return std::ostream& The output stream.
     */
    friend std::ostream & operator<<(std::ostream & os,
                                     const TracebackEntry & self) {
      if (self.resolved) {
        os << "  File \"" << self.file << "\"" << std::endl;
        os << "    " << self.name;
      } else {
        os << "  Stack frame [" << self.address << "] could not be resolved to "
           << "a function/method name.";
      }
      return os;
    }

   protected:
    /**
     * @brief Discover the name and file associated with the stack frame.
     *
     * This function attempts to resolve the stack frame to a function/method
     * name and file. It demangles the function name if necessary.
     */
    void discover_name_and_file();

    void * address;      ///< The address of the stack frame.
    std::string symbol;  ///< The symbol associated with the stack frame.
    std::string name;    ///< The name associated with the stack frame.
    std::string file;    ///< The file associated with the stack frame.
    bool resolved;  ///< True if the stack frame has been successfully resolved
                    ///< to a function/method name, false otherwise.
  };

  /**
   * @class Traceback
   * @brief A class that captures and manages a stack traceback.
   *
   * This class is used to capture the stack traceback at the point of exception
   * creation. It provides methods to get the stack traceback and to print it.
   */
  class Traceback {
   public:
    /**
     * @brief Construct a new Traceback object.
     *
     * This constructor captures the current stack traceback, discarding the
     * topmost entries as specified.
     *
     * @param discard_entries The number of topmost entries to discard from the
     * captured stack traceback.
     */
    explicit Traceback(int discard_entries);

    /**
     * @brief Destroy the Traceback object.
     *
     * This is a virtual destructor, allowing this class to be used as a base
     * class.
     */
    virtual ~Traceback();

    /**
     * @brief Get the stack traceback.
     *
     * This function returns a reference to the vector of traceback entries.
     *
     * @return const std::vector<TracebackEntry>& The stack traceback.
     */
    const std::vector<TracebackEntry> & get_stack() const {
      return this->stack;
    }

    /**
     * @brief Output the stack traceback.
     *
     * This function outputs the stack traceback to the provided output stream.
     * The traceback is output in reverse order (most recent entry last), and
     * stops at the first entry that could not be resolved to a function name.
     *
     * @param os The output stream to output the traceback to.
     * @param self The Traceback object to output the traceback of.
     * @return std::ostream& The output stream.
     */
    friend std::ostream & operator<<(std::ostream & os,
                                     const Traceback & self) {
      size_t i = 0;
      for (; i < self.stack.size(); ++i) {
        /* We stop dumping the stack trace at the first entry that could not be
         * resolved to a function name. This is typically the entry point to
         * the library from Python or from the Boost test environment. Since
         * Python provides traceback, we actually do not want the full stack
         * trace that also contains calls within the Python library.
         */
        if (!self.stack[i].is_resolved())
          break;
      }
      // Print stack trace in reverse or (most recent entry last)
      for (ssize_t j = i - 1; j >= 0; --j) {
        os << self.stack[j];
        if (j != 0)
          os << std::endl;
      }
      return os;
    }

   protected:
    /**
     * @brief The captured stack traceback.
     *
     * This vector contains the entries of the captured stack traceback.
     */
    std::vector<TracebackEntry> stack;
  };

  /**
   * @class ExceptionWithTraceback
   * @brief A template class that extends the exception class provided as a
   * template parameter.
   *
   * This class is used to add traceback information to exceptions. It captures
   * the stack trace at the point of exception creation. The traceback
   * information is then included in the exception message.
   *
   * @tparam T The exception class to extend. This should be a type derived from
   * std::exception.
   */
  template <class T>
  class ExceptionWithTraceback : public T {
   public:
    /**
     * @brief Construct a new ExceptionWithTraceback object.
     *
     * This constructor initializes the base exception with the provided
     * message, captures the current stack trace, and prepares the full
     * exception message including the traceback information.
     *
     * @param message The message for the base exception.
     */
    explicit ExceptionWithTraceback(const std::string & message)
        : T{message}, traceback{3}, buffer{} {
      std::stringstream os;
      os << T::what() << std::endl;
      os << "Traceback from C++ library (most recent call last):" << std::endl;
      os << this->traceback;
      buffer = os.str();
    }

    /**
     * @brief Destroy the ExceptionWithTraceback object.
     *
     * This is a no-throw destructor, as required for exceptions.
     */
    virtual ~ExceptionWithTraceback() noexcept {}

    /**
     * @brief Get the exception message.
     *
     * This function returns the full exception message, including the traceback
     * information.
     *
     * @return const char* The exception message.
     */
    virtual const char * what() const noexcept { return buffer.c_str(); }

   protected:
    Traceback traceback;  ///< The captured stack trace.
    std::string buffer;   ///< The full exception message, including the
                          ///< traceback information.
  };

  /**
   * @typedef RuntimeError
   * @brief A type alias for ExceptionWithTraceback specialized with
   * std::runtime_error.
   *
   * This type is used to throw exceptions that include a stack trace, for
   * runtime errors.
   */
  using RuntimeError = ExceptionWithTraceback<std::runtime_error>;
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_EXCEPTION_HH_
