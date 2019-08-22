#ifndef SRC_CELL_CELL_SPLIT_FACTORY_HH_
#define SRC_CELL_CELL_SPLIT_FACTORY_HH_

#include "common/muSpectre_common.hh"
#include "libmugrid/ccoord_operations.hh"
#include "cell/cell_split.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "projection/projection_small_strain.hh"
#include "libmufft/fftw_engine.hh"
#include "cell/cell_factory.hh"

#ifdef WITH_MPI
#include <libmufft/communicator.hh>
#include <libmufft/fftwmpi_engine.hh>
#endif

#include <memory>
namespace muSpectre {
  template <size_t DimS, size_t DimM = DimS,
            typename Cell = CellSplit<DimS, DimM>,
            typename FFTEngine = muFFT::FFTWEngine<DimS>>
  inline Cell
  make_cell_split(Ccoord_t<DimS> resolutions, Rcoord_t<DimS> lengths,
                  Formulation form,
                  Gradient_t<DimS> gradient = make_fourier_gradient<DimS>(),
                  const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto && input = cell_input<DimS, DimM, FFTEngine>(resolutions, lengths,
                                                      form, gradient, comm);
    auto cell{Cell(std::move(input))};
    return cell;
  }
  /* ---------------------------------------------------------------------- */
  // this function returns a unique pointer to the CellSplit class of the cell
  // all members of cell and its descending cell class such as CellSplit are
  // available
  template <size_t DimS, size_t DimM = DimS,
            typename Cell = CellSplit<DimS, DimM>,
            typename FFTEngine = muFFT::FFTWEngine<DimS>>
  std::unique_ptr<Cell>
  make_cell_ptr(Ccoord_t<DimS> resolutions, Rcoord_t<DimS> lengths,
                Formulation form,
                Gradient_t<DimS> gradient = make_fourier_gradient<DimS>(),
                const muFFT::Communicator & comm = muFFT::Communicator()) {
    auto && input = cell_input<DimS, DimM, FFTEngine>(resolutions, lengths,
                                                      form, gradient, comm);
    return std::make_unique<Cell>(std::move(input));
  }
}  // namespace muSpectre

#endif  // SRC_CELL_CELL_SPLIT_FACTORY_HH_
