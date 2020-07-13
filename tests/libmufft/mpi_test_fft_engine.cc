/**
 * @file   mpi_test_fft_engine.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   06 Mar 2017
 *
 * @brief  tests for MPI-parallel fft engine implementations
 *
 * Copyright © 2017 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 50

#include <boost/mpl/list.hpp>

#include "tests.hh"
#include "mpi_context.hh"
#include <libmufft/fftw_engine.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/field_collection.hh>
#include <libmugrid/field_map_static.hh>
#include <libmugrid/iterators.hh>

namespace muFFT {
  using muFFT::tol;

  BOOST_AUTO_TEST_SUITE(mpi_fft_engine);
  /* ---------------------------------------------------------------------- */
  template <typename Engine, Dim_t dim, Dim_t NbGridPts, bool serial = false>
  struct FFTW_fixture {
    constexpr static Dim_t BoxNbGridPts{NbGridPts};
    constexpr static Dim_t serial_engine{serial};
    constexpr static Real BoxLength{4.5};
    constexpr static Dim_t sdim{dim};
    constexpr static Dim_t NbComponents{sdim * sdim};
    static DynCcoord_t res() {
      return muGrid::CcoordOps::get_cube(sdim, BoxNbGridPts);
    }
    FFTW_fixture() : engine(res(), MPIContext::get_context().comm) {}
    Engine engine;
  };

  template <typename Engine, Dim_t dim, Dim_t NbGridPts, bool serial>
  constexpr Dim_t FFTW_fixture<Engine, dim, NbGridPts, serial>::BoxNbGridPts;
  template <typename Engine, Dim_t dim, Dim_t NbGridPts, bool serial>
  constexpr Dim_t FFTW_fixture<Engine, dim, NbGridPts, serial>::sdim;
  template <typename Engine, Dim_t dim, Dim_t NbGridPts, bool serial>
  constexpr Dim_t FFTW_fixture<Engine, dim, NbGridPts, serial>::NbComponents;

  template <typename Engine>
  struct FFTW_fixture_python_segfault {
    constexpr static Dim_t serial_engine{false};
    constexpr static Dim_t dim{twoD};
    constexpr static Dim_t sdim{twoD};
    constexpr static Dim_t mdim{twoD};
    constexpr static Dim_t NbComponents{sdim * sdim};
    static DynCcoord_t res() { return {6, 4}; }
    FFTW_fixture_python_segfault()
        : engine{res(), MPIContext::get_context().comm} {}
    Engine engine;
  };

  template <typename Engine>
  constexpr Dim_t FFTW_fixture_python_segfault<Engine>::sdim;
  template <typename Engine>
  constexpr Dim_t FFTW_fixture_python_segfault<Engine>::NbComponents;

  using fixlist = boost::mpl::list<
#ifdef WITH_FFTWMPI
      FFTW_fixture<FFTWMPIEngine, twoD, 3>,
      FFTW_fixture<FFTWMPIEngine, threeD, 3>,
      FFTW_fixture<FFTWMPIEngine, twoD, 4>,
      FFTW_fixture<FFTWMPIEngine, threeD, 4>,
      FFTW_fixture_python_segfault<FFTWMPIEngine>,
#endif
#ifdef WITH_PFFT
      FFTW_fixture<PFFTEngine, twoD, 3>, FFTW_fixture<PFFTEngine, threeD, 3>,
      FFTW_fixture<PFFTEngine, twoD, 4>, FFTW_fixture<PFFTEngine, threeD, 4>,
      FFTW_fixture_python_segfault<PFFTEngine>,
#endif
      FFTW_fixture<FFTWEngine, twoD, 3, true>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Constructor_test, Fix, fixlist, Fix) {
    Communicator & comm = MPIContext::get_context().comm;
    if (Fix::serial_engine && comm.size() > 1) {
      return;
    } else {
      BOOST_CHECK_NO_THROW(
          Fix::engine.create_plan(Fix::NbComponents));
    }
    BOOST_CHECK_EQUAL(comm.sum(Fix::engine.size()),
                      muGrid::CcoordOps::get_size(Fix::res()));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fft_test, Fix, fixlist, Fix) {
    if (Fix::serial_engine && Fix::engine.get_communicator().size() > 1) {
      // dont test serial engies in parallel
      return;
    } else {
      Fix::engine.create_plan(Fix::NbComponents);
    }
    using FC_t = muGrid::GlobalFieldCollection;
    FC_t fc{Fix::sdim};
    auto & input{fc.register_real_field("input", Fix::sdim * Fix::sdim)};
    auto & ref{fc.register_real_field("reference", Fix::sdim * Fix::sdim)};
    auto & result{fc.register_real_field("result", Fix::sdim * Fix::sdim)};

    fc.initialise(Fix::engine.get_nb_subdomain_grid_pts(),
                  Fix::engine.get_subdomain_locations());

    using map_t = muGrid::MatrixFieldMap<Real, Mapping::Mut, Fix::sdim,
                                         Fix::sdim, IterUnit::Pixel>;
    map_t inmap{input};
    auto refmap{map_t{ref}};
    auto resultmap{map_t{result}};
    size_t cntr{0};
    for (auto tup : akantu::zip(inmap, refmap)) {
      cntr++;
      auto & in_{std::get<0>(tup)};
      auto & ref_{std::get<1>(tup)};
      in_.setRandom();
      ref_ = in_;
    }
    auto & complex_field{Fix::engine.register_fourier_space_field(
        "complex field", Fix::NbComponents)};
    BOOST_TEST_CHECKPOINT("reached");

    auto && rank{Fix::engine.get_communicator().rank()};
    std::cout << "Rank " << rank
              << ", Fourier_field_collection.nb_subdomain_grid_pts: "
              << Fix::engine.get_nb_fourier_grid_pts() << std::endl;
    std::cout << "engine.nb_domain_grid_pts: "
              << Fix::engine.get_nb_domain_grid_pts() << std::endl;
    std::cout << "engine.nb_subdomain_grid_pts: "
              << Fix::engine.get_nb_subdomain_grid_pts() << std::endl;
    BOOST_TEST_CHECKPOINT("reached1");
    Fix::engine.fft(input, complex_field);
    BOOST_TEST_CHECKPOINT("reached2");
    using cmap_t = muGrid::MatrixFieldMap<Complex, Mapping::Mut, Fix::sdim,
                                          Fix::sdim, IterUnit::Pixel>;
    cmap_t complex_map(complex_field);
    BOOST_TEST_CHECKPOINT("reached3");
    if (Fix::engine.get_subdomain_locations() ==
        muGrid::CcoordOps::get_cube<Fix::sdim>(Index_t{0})) {
      // Check that 0,0 location has no imaginary part.
      Real error = complex_map[0].imag().norm();
      BOOST_CHECK_LT(error, tol);
    }

    /* make sure, the engine has not modified input (which is
       unfortunately const-casted internally, hence this test) */
    for (auto && tup : akantu::zip(inmap, refmap)) {
      Real error{(std::get<0>(tup) - std::get<1>(tup)).norm()};
      BOOST_CHECK_LT(error, tol);
    }

    /* make sure that the ifft of fft returns the original*/
    Fix::engine.ifft(complex_field, result);
    for (auto && tup : akantu::zip(resultmap, refmap)) {
      auto && result{std::get<0>(tup)};
      auto && reference{std::get<1>(tup)};
      auto && normalisation{Fix::engine.normalisation()};
      Real error{(result * normalisation - reference).norm()};
      BOOST_CHECK_LT(error, tol);
      if (error > tol) {
        std::cout << result.array() / reference.array() << std::endl
                  << std::endl;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(gather_test) {
    auto & comm{MPIContext::get_context().comm};
    auto rank{comm.rank()};
    const Dim_t nb_cols = rank + 1;
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat(2, nb_cols);

    for (int col{0}; col < nb_cols; ++col) {
      send_mat.col(col) = send_mat.col(col).Ones(2, 1) * (rank + col);
    }

    auto res{comm.template gather<Real>(send_mat)};
    int counter{0};
    for (int lrank{0}; lrank < comm.size(); ++lrank) {
      for (int col{0}; col < lrank + 1; ++col) {
        BOOST_CHECK_EQUAL(res(0, counter), lrank + col);
        counter++;
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(sum_mat_test) {
    auto & comm{MPIContext::get_context().comm};
    auto nb_cores{comm.size()};
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat(2, 3);
    send_mat(0, 0) = 1.;
    send_mat(0, 1) = 2.;
    send_mat(0, 2) = 3.;
    send_mat(1, 0) = 4.;
    send_mat(1, 1) = 5.;
    send_mat(1, 2) = 6.;
    auto res{comm.template sum_mat<Real>(send_mat)};
    const auto nb_cols{send_mat.cols()};
    const auto nb_rows{send_mat.rows()};
    for (int row{0}; row < nb_rows; row++) {
      for (int col{0}; col < nb_cols; col++) {
        BOOST_CHECK_EQUAL(res(row, col), (row * nb_cols + col + 1) * nb_cores);
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muFFT
