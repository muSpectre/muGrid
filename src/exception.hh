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

  //! An entry from the stack traceback
  class TracebackEntry {
   public:
    TracebackEntry(void * address, const std::string & symbol);
    TracebackEntry(void * address, const char * symbol);
    TracebackEntry(const TracebackEntry & other);

    ~TracebackEntry();

    TracebackEntry & operator=(const TracebackEntry & other);

    const std::string & get_symbol() const { return this->symbol; }
    const std::string & get_name() const { return this->name; }
    const std::string & get_file() const { return this->file; }

    bool is_resolved() const { return this->resolved; }

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
    void discover_name_and_file();

    void * address;
    std::string symbol;
    std::string name;
    std::string file;
    bool resolved;  // has name and file been successfully resolved?
  };

  class Traceback {
   public:
    explicit Traceback(int discard_entries);
    virtual ~Traceback();

    const std::vector<TracebackEntry> & get_stack() const {
      return this->stack;
    }

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
    std::vector<TracebackEntry> stack;
  };

  template <class T>
  class ExceptionWithTraceback : public T {
   public:
    explicit ExceptionWithTraceback(const std::string & message)
        : T{message}, traceback{3}, buffer{} {
      std::stringstream os;
      os << T::what() << std::endl;
      os << "Traceback from C++ library (most recent call last):" << std::endl;
      os << this->traceback;
      buffer = os.str();
    }
    virtual ~ExceptionWithTraceback() noexcept {}

    virtual const char * what() const noexcept { return buffer.c_str(); }

   protected:
    Traceback traceback;
    std::string buffer;
  };

  using RuntimeError = ExceptionWithTraceback<std::runtime_error>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_EXCEPTION_HH_
