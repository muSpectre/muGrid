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
#else   // WITH_MPI
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
   * @class FileIOBase
   * @brief A virtual base class for FileIO classes.
   *
   * This class provides a common interface for file input/output operations.
   * It defines an enumeration for open modes (Read, Write, Append) and provides
   * a set of constructors and assignment operators (both copy and move are
   * deleted). It also provides a virtual destructor and a random access
   * operator.
   *
   * @note This class cannot be instantiated directly (default constructor is
   * deleted).
   */
  class FileIOBase {
   public:
    /**
     * @enum OpenMode
     * @brief Enumeration for file open modes.
     *
     * This enumeration defines the possible modes for opening a file:
     * - Read: File is opened for reading only. This mode is used when the data
     * in the file is only needed for input and will not be modified.
     * - Write: File is opened for writing only. This mode is used when new data
     * is to be written to a file. If the file already exists, this mode will
     * fail to prevent accidental data loss.
     * - Overwrite: File is opened for writing only. This mode is used when new
     * data is to be written to a file. If the file already exists, it will be
     * overwritten. Use this mode with caution to prevent accidental data loss.
     * - Append: File is opened for writing only. This mode is used when new
     * data is to be added to the end of a file. If the file already exists, the
     * new data will be added at the end, preserving the existing data.
     */
    enum class OpenMode { Read, Write, Overwrite, Append };

    //! Default constructor is deleted to prevent direct instantiation of this
    //! class without parameters.
    FileIOBase() = delete;

    /**
     * @brief Constructor with file name, open mode, and communicator.
     *
     * This constructor initializes a FileIOBase object with the given file
     * name, open mode, and communicator.
     *
     * @param file_name The name of the file to be opened.
     * @param open_mode The mode to open the file in (Read, Write, or Append).
     * @param comm The communicator to be used for parallel I/O operations
     * (default is a default-constructed Communicator object).
     */
    FileIOBase(const std::string & file_name, const OpenMode & open_mode,
               Communicator comm = Communicator());

    //! Copy constructor is deleted to prevent copying of FileIOBase objects.
    FileIOBase(const FileIOBase & other) = delete;

    //! Move constructor is deleted to prevent moving of FileIOBase objects.
    FileIOBase(FileIOBase && other) = delete;

    //! Virtual destructor.
    virtual ~FileIOBase() = default;

    //! Copy assignment operator is deleted to prevent copying of FileIOBase
    //! objects.
    FileIOBase & operator=(const FileIOBase & other) = delete;

    //! Move assignment operator is deleted to prevent moving of FileIOBase
    //! objects.
    FileIOBase & operator=(FileIOBase && other) = delete;

    /**
     * @brief Random access operator.
     *
     * This operator provides access to a specific frame in the file.
     * The frame must exist for this operation to succeed.
     *
     * @param frame_index The index of the frame to access.
     * @return FileFrame The requested frame.
     */
    FileFrame operator[](const Index_t & frame_index);

    /**
     * @brief Register a field collection to be dumped to the file.
     *
     * This function should be called before the file is opened. If no field
     * names are given, all fields of the given field collection are registered
     * by default.
     *
     * @param fc The field collection to be registered.
     * @param field_names The names of the fields to be registered. Default is
     * all fields.
     * @param state_field_unique_prefixes The unique prefixes of the state
     * fields to be registered. Default is all state fields.
     */
    virtual void register_field_collection(
        muGrid::FieldCollection & fc,
        std::vector<std::string> field_names = {REGISTER_ALL_FIELDS},
        std::vector<std::string> state_field_unique_prefixes = {
            REGISTER_ALL_STATE_FIELDS}) = 0;

    /**
     * @brief Close the file.
     */
    virtual void close() = 0;

    /**
     * @brief Read the fields identified by `field_names` from a specific frame
     * in the file.
     *
     * @param frame The frame to read from.
     * @param field_names The names of the fields to read.
     */
    virtual void read(const Index_t & frame,
                      const std::vector<std::string> & field_names) = 0;

    /**
     * @brief Read all registered fields from a specific frame in the file.
     *
     * @param frame The frame to read from.
     */
    virtual void read(const Index_t & frame) = 0;

    /**
     * @brief Write the contents of all fields identified by `field_names`
     * within the field collection to a specific frame in the file.
     *
     * @param frame The frame to write to.
     * @param field_names The names of the fields to write.
     */
    virtual void write(const Index_t & frame,
                       const std::vector<std::string> & field_names) = 0;

    /**
     * @brief Write the contents of all fields within the field collection to a
     * specific frame in the file.
     *
     * @param frame The frame to write to.
     */
    virtual void write(const Index_t & frame) = 0;

    /**
     * @brief Yield an empty file frame at the end of the file.
     *
     * @return FileFrame The empty file frame.
     */
    FileFrame append_frame();

    /**
     * @brief Get the communicator object.
     *
     * @return Communicator& The communicator object.
     */
    Communicator & get_communicator();

    /**
     * @class iterator
     * @brief A class for iterating over the frames in the file.
     */
    class iterator;

    /**
     * @brief Get an iterator pointing to the first frame in the file.
     *
     * @return iterator An iterator pointing to the first frame.
     */
    iterator begin();

    /**
     * @brief Get an iterator pointing one past the last frame in the file.
     *
     * @return iterator An iterator pointing one past the last frame.
     */
    iterator end();

    /**
     * @brief Get the number of frames in the file.
     *
     * @return size_t The number of frames.
     */
    size_t size() const;

   protected:
    /**
     * @brief Open the file for read/write operations.
     *
     * This function should be called by the constructor at instantiation.
     * It is a pure virtual function and must be implemented by derived classes.
     */
    virtual void open() = 0;

    /**
     * @brief Register a global field collection to be dumped to the file.
     *
     * This function should be called before the file is opened. If no field
     * names are given, all fields of the given field collection are registered
     * by default.
     *
     * @param fc_global The global field collection to be registered.
     * @param field_names The names of the fields to be registered. Default is
     * all fields.
     * @param state_field_unique_prefixes The unique prefixes of the state
     * fields to be registered. Default is all state fields.
     */
    virtual void register_field_collection_global(
        muGrid::GlobalFieldCollection & fc_global,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) = 0;

    /**
     * @brief Register a local field collection to be dumped to the file.
     *
     * This function should be called before the file is opened. If no field
     * names are given, all fields of the given field collection are registered
     * by default.
     *
     * @param fc_local The local field collection to be registered.
     * @param field_names The names of the fields to be registered. Default is
     * all fields.
     * @param state_field_unique_prefixes The unique prefixes of the state
     * fields to be registered. Default is all state fields.
     */
    virtual void register_field_collection_local(
        muGrid::LocalFieldCollection & fc_local,
        const std::vector<std::string> & field_names,
        const std::vector<std::string> & state_field_unique_prefixes) = 0;

    /**
     * @brief The name of the file to be opened.
     */
    const std::string file_name;

    /**
     * @brief The mode to open the file in (Read, Write, or Append).
     */
    const OpenMode open_mode;

    /**
     * @brief The communicator to be used for parallel I/O operations.
     */
    Communicator comm;

    /**
     * @brief The number of frames in the file.
     */
    Index_t nb_frames{0};
  };

  /**
   * @class FileFrame
   * @brief A virtual base class for Frame classes.
   *
   * This class provides a common interface for file frame operations.
   * It provides a set of constructors and assignment operators (both copy and
   * move are deleted). It also provides a virtual destructor and methods for
   * reading and writing frames.
   *
   * @note This class cannot be instantiated directly (default constructor is
   * deleted).
   */
  class FileFrame {
   public:
    /**
     * @brief Default constructor is deleted to prevent direct instantiation of
     * this class without parameters.
     */
    FileFrame() = delete;

    /**
     * @brief Constructor with the FileIOBase object and the required frame
     * number.
     *
     * This constructor initializes a FileFrame object with the given FileIOBase
     * object and frame number.
     *
     * @param parent The FileIOBase object.
     * @param frame The frame number.
     */
    explicit FileFrame(FileIOBase & parent, Index_t frame);

    /**
     * @brief Copy constructor is deleted to prevent copying of FileFrame
     * objects.
     */
    FileFrame(const FileFrame & other) = default;

    /**
     * @brief Move constructor is deleted to prevent moving of FileFrame
     * objects.
     */
    FileFrame(FileFrame && other) = default;

    /**
     * @brief Virtual destructor.
     */
    virtual ~FileFrame() = default;

    /**
     * @brief Copy assignment operator is deleted to prevent copying of
     * FileFrame objects.
     */
    FileFrame & operator=(const FileFrame & other) = delete;

    /**
     * @brief Move assignment operator is deleted to prevent moving of FileFrame
     * objects.
     */
    FileFrame & operator=(FileFrame && other) = delete;

    /**
     * @brief Read the fields identified by `field_names` from the current
     * frame.
     *
     * @param field_names The names of the fields to read.
     */
    void read(const std::vector<std::string> & field_names) const;

    /**
     * @brief Read all fields of the registered field collection(s) from the
     * current frame.
     */
    void read() const;

    /**
     * @brief Write the contents of all fields within the field collection with
     * the names 'field_names' to the file.
     *
     * @param field_names The names of the fields to write.
     */
    void write(const std::vector<std::string> & field_names) const;

    /**
     * @brief Write the contents of all fields within the field collection to
     * the file.
     */
    void write() const;

   protected:
    /**
     * @brief The FileIOBase object.
     */
    FileIOBase & parent;

    /**
     * @brief The frame number.
     */
    Index_t frame;
  };

  /**
   * @class FileIOBase::iterator
   * @brief A class for iterating over the frames in the file.
   *
   * This class provides a common interface for file frame iteration operations.
   * It provides a set of constructors and assignment operators (both copy and
   * move are deleted). It also provides a destructor and methods for
   * dereferencing and incrementing.
   *
   * @note This class cannot be instantiated directly (default constructor is
   * deleted).
   */
  class FileIOBase::iterator {
   public:
    //! STL type aliases for iterator conformance
    using value_type = FileFrame;
    using const_value_type = const value_type;            //!< STL conformance
    using pointer = value_type *;                         //!< STL conformance
    using difference_type = std::ptrdiff_t;               //!< STL conformance
    using iterator_category = std::forward_iterator_tag;  //!< STL conformance

    /**
     * @brief Default constructor is deleted to prevent direct instantiation of
     * this class without parameters.
     */
    iterator() = delete;

    /**
     * @brief Constructor with the FileIOBase object and the required frame
     * index.
     *
     * This constructor initializes an iterator object with the given FileIOBase
     * object and frame index.
     *
     * @param parent The FileIOBase object.
     * @param frame_index The frame index.
     */
    explicit iterator(FileIOBase & parent, Index_t frame_index = 0)
        : parent{parent}, frame_index{frame_index} {}

    /**
     * @brief Copy constructor.
     */
    iterator(const iterator & other) = default;

    /**
     * @brief Move constructor.
     */
    iterator(iterator && other) = default;

    /**
     * @brief Destructor.
     */
    ~iterator() = default;

    /**
     * @brief Copy assignment operator is deleted to prevent copying of iterator
     * objects.
     */
    iterator & operator=(const iterator & other) = delete;

    /**
     * @brief Move assignment operator is deleted to prevent moving of iterator
     * objects.
     */
    iterator & operator=(iterator && other) = delete;

    /**
     * @brief Dereferencing operator.
     *
     * This operator provides access to a specific frame in the file.
     * The frame must exist for this operation to succeed.
     *
     * @return const_value_type The requested frame.
     */
    inline const_value_type operator*() const {
      return this->parent[this->frame_index];
    }

    /**
     * @brief Pre-increment operator.
     *
     * This operator increments the frame index by one.
     *
     * @return iterator& A reference to the incremented iterator.
     */
    inline iterator & operator++() {
      ++this->frame_index;
      return *this;
    }

    /**
     * @brief Equality comparison operator.
     *
     * This operator checks if the frame index of the current iterator is equal
     * to that of the other iterator.
     *
     * @param other The other iterator to compare with.
     * @return bool True if the frame indices are equal, false otherwise.
     */
    bool operator==(const iterator & other) const {
      return this->frame_index == other.frame_index;
    }

    /**
     * @brief Inequality comparison operator.
     *
     * This operator checks if the frame index of the current iterator is not
     * equal to that of the other iterator.
     *
     * @param other The other iterator to compare with.
     * @return bool True if the frame indices are not equal, false otherwise.
     */
    bool operator!=(const iterator & other) const {
      return this->frame_index != other.frame_index;
    }

   protected:
    /**
     * @brief The FileIOBase object.
     */
    FileIOBase & parent;

    /**
     * @brief The frame index.
     */
    Index_t frame_index{};
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FILE_IO_BASE_HH_
