/**
 * @file   file_io_base.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   20 Mai 2020
 *
 * @brief  Interface for parallel I/O of grid data
 *
 * Copyright © 2020 Till Junge
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

#include "file_io_base.hh"

namespace muGrid {

  FileIOBase::FileIOBase(const std::string & file_name,
                         const OpenMode & open_mode, Communicator comm)
      : file_name{file_name}, open_mode{open_mode}, comm{comm} {};

  /* ---------------------------------------------------------------------- */
  FileFrame FileIOBase::operator[](const Index_t & frame_index) {
    Index_t tmp_frame_index{frame_index};
    if (tmp_frame_index < 0) {
      tmp_frame_index = this->nb_frames + tmp_frame_index;
      if (tmp_frame_index < 0) {
        throw FileIOError{
            "You request the frame '" + std::to_string(frame_index) +
            "' but you have in total only '" + std::to_string(this->nb_frames) +
            "' frames in your FileIO object. Thus this is not possible."};
      }
    }

    if (tmp_frame_index > this->nb_frames) {
      throw FileIOError{
          "The frame " + std::to_string(tmp_frame_index) +
          " exceeds the total number of frames (" +
          std::to_string(this->nb_frames) +
            "). Only existing frames are accessible by operator[]."};
    }

    FileFrame file_frame{*this, tmp_frame_index};
    return file_frame;
  }

  /* ---------------------------------------------------------------------- */
  FileFrame FileIOBase::append_frame() {
    FileFrame file_frame{*this, this->nb_frames};
    this->nb_frames++;
    return file_frame;
  }

  /* ---------------------------------------------------------------------- */
  Communicator & FileIOBase::get_communicator() { return this->comm; }

  /* ---------------------------------------------------------------------- */
  FileIOBase::iterator FileIOBase::begin() {
    return FileIOBase::iterator{*this, 0};
  }

  /* ---------------------------------------------------------------------- */
  FileIOBase::iterator FileIOBase::end() {
    return FileIOBase::iterator{*this, static_cast<Index_t>(this->size())};
  }

  /* ---------------------------------------------------------------------- */
  size_t FileIOBase::size() const { return this->nb_frames; }

  /* ---------------------------------------------------------------------- */
  FileFrame::FileFrame(FileIOBase & parent, Index_t frame)
      : parent{parent}, frame{frame} {}

  /* ---------------------------------------------------------------------- */
  void FileFrame::read(const std::vector<std::string> & field_names) const {
    this->parent.read(this->frame, field_names);
  }

  /* ---------------------------------------------------------------------- */
  void FileFrame::read() const { this->parent.read(this->frame); }

  /* ---------------------------------------------------------------------- */
  void FileFrame::write(const std::vector<std::string> & field_names) const {
    this->parent.write(this->frame, field_names);
  }

  /* ---------------------------------------------------------------------- */
  void FileFrame::write() const { this->parent.write(this->frame); }

}  // namespace muGrid
