/**
 * @file   exception.cc
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

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include "exception.hh"

using muGrid::Traceback;
using muGrid::TracebackEntry;

const int MAX_DEPTH = 256;

TracebackEntry::TracebackEntry(void * address, const std::string & symbol)
    : address(address), symbol(symbol), name{}, file{}, resolved{false} {
  discover_name_and_file();
}

TracebackEntry::TracebackEntry(void * address, const char * symbol)
    : address(address), symbol(symbol), name{}, file{}, resolved{false} {
  discover_name_and_file();
}

TracebackEntry::TracebackEntry(const TracebackEntry & other)
    : address(other.address), symbol(other.symbol),
      name(other.name), file{other.file}, resolved{other.resolved} {}

TracebackEntry::~TracebackEntry() {}

TracebackEntry & TracebackEntry::operator=(const TracebackEntry & other) {
  this->address = other.address;
  this->symbol = other.symbol;
  this->name = other.name;
  this->file = other.file;
  this->resolved = other.resolved;
  return *this;
}

void TracebackEntry::discover_name_and_file() {
  Dl_info info;
  if (!dladdr(this->address, &info))
    return;

  if (info.dli_sname) {
    this->name = info.dli_sname;

    int status;
    char * demangled =
        abi::__cxa_demangle(this->name.c_str(), NULL, NULL, &status);
    if (status == 0 && demangled)
      this->name = demangled;
    if (demangled)
      free(demangled);

    this->resolved = true;
  }

  if (info.dli_fname)
    this->file = info.dli_fname;
}

Traceback::Traceback(int discard_entries) : stack{} {
  void * buffer[MAX_DEPTH];
  int size = backtrace(buffer, MAX_DEPTH);
  char ** symbols = backtrace_symbols(buffer, size);

  for (int i = discard_entries; i < size; ++i) {
    this->stack.push_back(TracebackEntry{buffer[i], symbols[i]});
  }

  free(symbols);
}

Traceback::~Traceback() {}
