/**
 * @file   file_io_base.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   17 May 2020
 *
 * @brief  Interface for parallel I/O of grid data
 *
 * Copyright © 2020 Richard Leute, Lars Pastewka, Till Junge
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

#ifndef SRC_LIBMUGRID_FILE_IO_BASE_HH_
#define SRC_LIBMUGRID_FILE_IO_BASE_HH_

#include "communicator.hh"
#include "field_collection.hh"
#include "field_collection_global.hh"
#include "field_collection_local.hh"

#include <vector>
#include <string>
#include <iostream>

#ifdef WITH_MPI
using Size_t = MPI_Offset;
#else  // WITH_MPI
using Size_t = size_t;
#endif  // WITH_MPI

namespace muGrid {

  constexpr static char REGISTER_ALL_FIELDS[] =
      "REGISTER_ALL_FIELDS";  // default value for parameter field_names in
                              // register_field_collection() which means all
                              // fields of the field collection are
                              // registered
  constexpr static char REGISTER_ALL_STATE_FIELDS[] =
      "REGISTER_ALL_STATE_FIELDS";  // default value for parameter
                                    // state_field_unique_prefixes in
                                    // register_field_collection() which
                                    // means all state fields of the field
                                    // collection are registered

  /**
   * base class for FileIO related exceptions
   */
  class FileIOError : public muGrid::RuntimeError {
   public:
    //! constructor
    explicit FileIOError(const std::string & what)
        : muGrid::RuntimeError(what) {}
    //! constructor
    explicit FileIOError(const char * what) : muGrid::RuntimeError(what) {}
  };

  //! forward declaration of the `muSpectre::Field`
  class Field;

  //! forward declaration of the `muGrid::FileFrame`
  class FileFrame;

  /**
   * Virtual base class for FileIO classes.
   */
  class FileIOBase {
   public:
    enum class OpenMode { Read, Write, Append };

    //! Default constructor
    FileIOBase() = delete;

    /**
     * Constructor with the domain's number of grid points in each direciton,
     * the number of components to transform, and the communicator
     */

    FileIOBase(const std::string & file_name, const OpenMode & open_mode,
               Communicator comm = Communicator());

    //! Copy constructor
    FileIOBase(const FileIOBase & other) = delete;

    //! Move constructor
    FileIOBase(FileIOBase && other) = delete;

    //! Destructor
    virtual ~FileIOBase() = default;

    //! Copy assignment operator
    FileIOBase & operator=(const FileIOBase & other) = delete;

    //! Move assignment operator
    FileIOBase & operator=(FileIOBase && other) = delete;

    //! Random access operator, the frame must exists for this operation to
    //! succeed
    FileFrame operator[](const Index_t & frame_index);

    //! tell the I/O object about the field collections we want to dump to this
    //! file before the file is opened. If no field names are given all fields
    //! of the given field collection are registered (default)
    virtual void register_field_collection(
        muGrid::FieldCollection & fc,
        std::vector<std::string> field_names = {REGISTER_ALL_FIELDS},
        std::vector<std::string> state_field_unique_prefixes = {
            REGISTER_ALL_STATE_FIELDS}) = 0;

    //! close file
    virtual void close() = 0;

    //! read the fields identified by `field_names` frame from file
    virtual void read(const Index_t & frame,
                      const std::vector<std::string> & field_names) = 0;

    //! read all registered fields in frame from file
    virtual void read(const Index_t & frame) = 0;

    //! write contents of all fields identified by `field_names` within the
    //! field collection to the file
    virtual void write(const Index_t & frame,
                       const std::vector<std::string> & field_names) = 0;

    //! write contents of all fields within the field collection to the file
    virtual void write(const Index_t & frame) = 0;

    //! yield an empty file frame at the end of the file
    FileFrame append_frame();

    //! return the communicator object
    Communicator & get_communicator();

    class iterator;
    //! stl conformance
    iterator begin();
    //! stl conformance
    iterator end();
    //! stl conformance
    size_t size() const;

   protected:
    //! open file for read/write
    //! This function should be called by the constructor at instantiation.
    virtual void open() = 0;

    //! register global field collection
    virtual void register_field_collection_global(
        muGrid::GlobalFieldCollection & fc_global,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) = 0;

    //! register local field collection
    virtual void register_field_collection_local(
        muGrid::LocalFieldCollection & fc_local,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) = 0;

    const std::string file_name;
    const OpenMode open_mode;
    Communicator comm;

    Index_t nb_frames{0};
  };

  /**
   * Virtual base class for Frame classes.
   */
  class FileFrame {
   public:
    //! Default constructor
    FileFrame() = delete;

    /**
     * Constructor with the FileIOBase object and the required frame number
     */
    explicit FileFrame(FileIOBase & parent, Index_t frame);

    //! Copy constructor
    FileFrame(const FileFrame & other) = default;

    //! Move constructor
    FileFrame(FileFrame && other) = default;

    //! Destructor
    virtual ~FileFrame() = default;

    //! Copy assignment operator
    FileFrame & operator=(const FileFrame & other) = delete;

    //! Move assignment operator
    FileFrame & operator=(FileFrame && other) = delete;

    //! read the fields identified by `field_names` from the current frame
    void read(const std::vector<std::string> & field_names) const;

    //! read all fields of the registered field collection(s) from the current
    //! frame
    void read() const;

    //! write contents of all fields within the field collection with the names
    //! 'field_names' to the file
    void write(const std::vector<std::string> & field_names) const;

    //! write contents of all fields within the field collection with the names
    //! 'field_names' to the file
    void write() const;

   protected:
    FileIOBase & parent;
    Index_t frame;
  };

  class FileIOBase::iterator {
   public:
    //! stl
    using value_type = FileFrame;
    using const_value_type = const value_type;            //!< stl conformance
    using pointer = value_type *;                         //!< stl conformance
    using difference_type = std::ptrdiff_t;               //!< stl conformance
    using iterator_category = std::forward_iterator_tag;  //!< stl conformance

    //! Default constructor
    iterator() = delete;

    //! constructor
    explicit iterator(FileIOBase & parent, Index_t frame_index = 0)
        : parent{parent}, frame_index{frame_index} {}

    //! Copy constructor
    iterator(const iterator & other) = default;

    //! Move constructor
    iterator(iterator && other) = default;

    //! Destructor
    ~iterator() = default;

    //! Copy assignment operator
    iterator & operator=(const iterator & other) = delete;

    //! Move assignment operator
    iterator & operator=(iterator && other) = delete;

    //! dereferencing
    inline const_value_type operator*() const {
      return this->parent[this->frame_index];
    }

    //! pre-increment
    inline iterator & operator++() {
      ++this->frame_index;
      return *this;
    }

    bool operator==(const iterator & other) const {
      return this->frame_index == other.frame_index;
    }

    bool operator!=(const iterator & other) const {
      return this->frame_index != other.frame_index;
    }

   protected:
    FileIOBase & parent;
    Index_t frame_index{};
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FILE_IO_BASE_HH_
