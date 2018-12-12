/**
 * @file   field_collection_global.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  FieldCollection base-class for global fields
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_COMMON_FIELD_COLLECTION_GLOBAL_HH_
#define SRC_COMMON_FIELD_COLLECTION_GLOBAL_HH_

#include "common/ccoord_operations.hh"
#include "common/field_collection_base.hh"

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Dim_t DimS> class LocalFieldCollection;

  /** `GlobalFieldCollection` derives from `FieldCollectionBase` and stores
   * global fields that live throughout the whole computational domain, i.e.
   * are defined for each pixel.
   */
  template <Dim_t DimS>
  class GlobalFieldCollection
      : public FieldCollectionBase<DimS, GlobalFieldCollection<DimS>> {
   public:
    //! for compile time check
    constexpr static bool Global{true};

    using Parent =
        FieldCollectionBase<DimS, GlobalFieldCollection<DimS>>;  //!< base class
    //! helpful for functions that fill global fields from local fields
    using LocalFieldCollection_t = LocalFieldCollection<DimS>;
    //! helpful for functions that fill global fields from local fields
    using GlobalFieldCollection_t = GlobalFieldCollection<DimS>;
    using Ccoord = typename Parent::Ccoord;    //!< cell coordinates type
    using Field_p = typename Parent::Field_p;  //!< spatial coordinates type
    //! iterator over all pixels contained it the collection
    using iterator = typename CcoordOps::Pixels<DimS>::iterator;
    //! Default constructor
    GlobalFieldCollection();

    //! Copy constructor
    GlobalFieldCollection(const GlobalFieldCollection &other) = delete;

    //! Move constructor
    GlobalFieldCollection(GlobalFieldCollection &&other) = default;

    //! Destructor
    virtual ~GlobalFieldCollection() = default;

    //! Copy assignment operator
    GlobalFieldCollection &
    operator=(const GlobalFieldCollection &other) = delete;

    //! Move assignment operator
    GlobalFieldCollection &operator=(GlobalFieldCollection &&other) = default;

    /** allocate memory, etc. At this point, the collection is
        informed aboud the size and shape of the domain (through the
        sizes parameter). The job of initialise is to make sure that
        all fields are either of size 0, in which case they need to be
        allocated, or are of the same size as the product of 'sizes'
        (if standard strides apply) any field of a different size is
        wrong.

        TODO: check whether it makes sense to put a runtime check here
     **/
    inline void initialise(Ccoord sizes, Ccoord locations);

    //! return subdomain resolutions
    inline const Ccoord &get_sizes() const;
    //! return subdomain locations
    inline const Ccoord &get_locations() const;

    //! returns the linear index corresponding to cell coordinates
    template <class CcoordRef>
    inline size_t get_index(CcoordRef &&ccoord) const;
    //! returns the cell coordinates corresponding to a linear index
    inline Ccoord get_ccoord(size_t index) const;

    inline iterator begin() const;  //!< returns iterator to first pixel
    inline iterator end() const;    //!< returns iterator past the last pixel

    //! return spatial dimension (template parameter)
    static constexpr inline Dim_t spatial_dim() { return DimS; }

    //! return globalness at compile time
    static constexpr inline bool is_global() { return Global; }

   protected:
    //! number of discretisation cells in each of the DimS spatial directions
    Ccoord sizes{};
    Ccoord locations{};
    CcoordOps::Pixels<DimS> pixels{};  //!< helper to iterate over the grid
   private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  GlobalFieldCollection<DimS>::GlobalFieldCollection() : Parent() {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void GlobalFieldCollection<DimS>::initialise(Ccoord sizes, Ccoord locations) {
    if (this->is_initialised) {
      throw std::runtime_error("double initialisation");
    }
    this->pixels = CcoordOps::Pixels<DimS>(sizes, locations);
    this->size_ = CcoordOps::get_size(sizes);
    this->sizes = sizes;
    this->locations = locations;

    std::for_each(
        std::begin(this->fields), std::end(this->fields), [this](auto &&item) {
          auto &&field = *item.second;
          const auto field_size = field.size();
          if (field_size == 0) {
            field.resize(this->size());
          } else if (field_size != this->size()) {
            std::stringstream err_stream;
            err_stream << "Field '" << field.get_name() << "' contains "
                       << field_size << " entries, but the field collection "
                       << "has " << this->size() << " pixels";
            throw FieldCollectionError(err_stream.str());
          }
        });
    this->is_initialised = true;
  }

  //----------------------------------------------------------------------------//
  //! return subdomain resolutions
  template <Dim_t DimS>
  const typename GlobalFieldCollection<DimS>::Ccoord &
  GlobalFieldCollection<DimS>::get_sizes() const {
    return this->sizes;
  }

  //----------------------------------------------------------------------------//
  //! return subdomain locations
  template <Dim_t DimS>
  const typename GlobalFieldCollection<DimS>::Ccoord &
  GlobalFieldCollection<DimS>::get_locations() const {
    return this->locations;
  }

  //----------------------------------------------------------------------------//
  //! returns the cell coordinates corresponding to a linear index
  template <Dim_t DimS>
  typename GlobalFieldCollection<DimS>::Ccoord
  GlobalFieldCollection<DimS>::get_ccoord(size_t index) const {
    return CcoordOps::get_ccoord(this->get_sizes(), this->get_locations(),
                                 std::move(index));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  typename GlobalFieldCollection<DimS>::iterator
  GlobalFieldCollection<DimS>::begin() const {
    return this->pixels.begin();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  typename GlobalFieldCollection<DimS>::iterator
  GlobalFieldCollection<DimS>::end() const {
    return this->pixels.end();
  }
  //-------------------------------------------------------------------------//
  //! returns the linear index corresponding to cell coordinates
  template <Dim_t DimS>
  template <class CcoordRef>
  size_t GlobalFieldCollection<DimS>::get_index(CcoordRef &&ccoord) const {
    static_assert(
        std::is_same<Ccoord, std::remove_const_t<
                                 std::remove_reference_t<CcoordRef>>>::value,
        "can only be called with values or references of Ccoord");
    return CcoordOps::get_index(this->get_sizes(), this->get_locations(),
                                std::forward<CcoordRef>(ccoord));
  }

}  // namespace muSpectre

#endif  // SRC_COMMON_FIELD_COLLECTION_GLOBAL_HH_
