/**
 * @file   field_collection_local.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  FieldCollection base-class for local fields
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef FIELD_COLLECTION_LOCAL_H
#define FIELD_COLLECTION_LOCAL_H

#include "common/field_collection_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  /** `LocalFieldCollection` derives from `FieldCollectionBase` and stores
    * local fields, i.e. fields that are only defined for a subset of all pixels
    * in the computational domain. The coordinates of these active pixels are
    * explicitly stored by this field collection.
    * `LocalFieldCollection::add_pixel` allows to add individual pixels to the
    * field collection.
    */
  template<Dim_t DimS, Dim_t DimM>
  class LocalFieldCollection:
    public FieldCollectionBase<DimS, DimM, LocalFieldCollection<DimS, DimM>>
  {
  public:
    //! base class
    using Parent = FieldCollectionBase<DimS, DimM,
                                       LocalFieldCollection<DimS, DimM>>;
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    using Field_p = typename Parent::Field_p; //!< field pointer
    using ccoords_container = std::vector<Ccoord>; //!< list of pixels
    //! iterator over managed pixels
    using iterator = typename ccoords_container::iterator;

    //! Default constructor
    LocalFieldCollection();

    //! Copy constructor
    LocalFieldCollection(const LocalFieldCollection &other) = delete;

    //! Move constructor
    LocalFieldCollection(LocalFieldCollection &&other) = delete;

    //! Destructor
    virtual ~LocalFieldCollection()  = default;

    //! Copy assignment operator
    LocalFieldCollection& operator=(const LocalFieldCollection &other) = delete;

    //! Move assignment operator
    LocalFieldCollection& operator=(LocalFieldCollection &&other) = delete;

    //! add a pixel/voxel to the field collection
    inline void add_pixel(const Ccoord & local_ccoord);

    /** allocate memory, etc. at this point, the field collection
        knows how many entries it should have from the size of the
        coords containes (which grows by one every time add_pixel is
        called. The job of initialise is to make sure that all fields
        are either of size 0, in which case they need to be allocated,
        or are of the same size as the product of 'sizes' any field of
        a different size is wrong TODO: check whether it makes sense
        to put a runtime check here
     **/
    inline void initialise();


    //! returns the linear index corresponding to cell coordinates
    template <class CcoordRef>
    inline size_t get_index(CcoordRef && ccoord) const;
    //! returns the cell coordinates corresponding to a linear index
    inline Ccoord get_ccoord(size_t index) const;

    //! iterator to first pixel
    inline iterator begin() {return this->ccoords.begin();}
    //! iterator past last pixel
    inline iterator end() {return this->ccoords.end();}

  protected:
    //! container of pixel coords for non-global collections
    ccoords_container ccoords{};
    //! container of indices for non-global collections (slow!)
    std::map<Ccoord, size_t> indices{};
  private:
  };


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  LocalFieldCollection<DimS, DimM>::LocalFieldCollection()
    :Parent(){}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void LocalFieldCollection<DimS, DimM>::
  add_pixel(const Ccoord & local_ccoord) {
    if (this->is_initialised) {
      throw FieldCollectionError
        ("once a field collection has been initialised, you can't add new "
         "pixels.");
    }
    this->indices[local_ccoord] = this->ccoords.size();
    this->ccoords.push_back(local_ccoord);
    this->size_++;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void LocalFieldCollection<DimS, DimM>::
  initialise() {
    if (this->is_initialised) {
      throw std::runtime_error("double initialisation");
    }
    std::for_each(std::begin(this->fields), std::end(this->fields),
                  [this](auto && item) {
                    auto && field = *item.second;
                    const auto field_size = field.size();
                    if (field_size == 0) {
                      field.resize(this->size());
                    } else if (field_size != this->size()) {
                      std::stringstream err_stream;
                      err_stream << "Field '" << field.get_name()
                                 << "' contains " << field_size
                                 << " entries, but the field collection "
                                 << "has " << this->size() << " pixels";
                      throw FieldCollectionError(err_stream.str());
                    }
                  });
    this->is_initialised = true;
  }


  //----------------------------------------------------------------------------//
  //! returns the linear index corresponding to cell coordinates
  template <Dim_t DimS, Dim_t DimM>
  template <class CcoordRef>
  size_t
  LocalFieldCollection<DimS, DimM>::get_index(CcoordRef && ccoord) const {
    static_assert(std::is_same<
                    Ccoord,
                    std::remove_const_t<
                      std::remove_reference_t<CcoordRef>>>::value,
                  "can only be called with values or references of Ccoord");
    return this->indices.at(std::forward<CcoordRef>(ccoord));
  }


  //----------------------------------------------------------------------------//
  //! returns the cell coordinates corresponding to a linear index
  template <Dim_t DimS, Dim_t DimM>
  typename LocalFieldCollection<DimS, DimM>::Ccoord
  LocalFieldCollection<DimS, DimM>::get_ccoord(size_t index) const {
    return this->ccoords[std::move(index)];
  }


}  // muSpectre

#endif /* FIELD_COLLECTION_LOCAL_H */
