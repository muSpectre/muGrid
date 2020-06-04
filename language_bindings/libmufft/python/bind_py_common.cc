/**
 * @file   bind_py_common.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   08 Jan 2018
 *
 * @brief  Python bindings for the common part of µFFT
 *
 * Copyright © 2018 Till Junge
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

#include "libmufft/fft_utils.hh"
#include "libmufft/mufft_common.hh"

#include <libmugrid/ccoord_operations.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


namespace py = pybind11;
using pybind11::literals::operator""_a;

void add_version(py::module & mod) {
  auto version{mod.def_submodule("version")};

  version.doc() = "version information";

  version.def("info", &muFFT::version::info)
      .def("hash", &muFFT::version::hash)
      .def("description", &muFFT::version::description)
      .def("is_dirty", &muFFT::version::is_dirty);
}

template <muGrid::Index_t dim>
void add_get_nb_hermitian_grid_pts_helper(py::module & mod) {
  mod.def(
      "get_nb_hermitian_grid_pts",
      [](const muGrid::Ccoord_t<dim> & nb_grid_pts) {
        return muFFT::get_nb_hermitian_grid_pts(nb_grid_pts);
      },
      "full_sizes"_a,
      "return the hermitian sizes corresponding to the true sizes");
}

template <muGrid::Index_t dim>
void add_fft_freqs_helper(py::module & mod) {
  using Ccoord = muGrid::Ccoord_t<dim>;
  using FFTFreqs = muFFT::FFT_freqs<dim>;
  using ArrayXDd =
      Eigen::Array<muGrid::Real, dim, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayXDi =
      Eigen::Array<muGrid::Index_t, dim, Eigen::Dynamic, Eigen::RowMajor>;
  std::stringstream name{};
  name << "FFTFreqs_" << dim << "d";
  py::class_<FFTFreqs>(mod, name.str().c_str())
      .def(py::init<Ccoord, std::array<muGrid::Real, dim>>(), "nb_grid_pts"_a,
           "lengths"_a)
      .def(
          "get_xi",
          [](FFTFreqs & fft_freqs, const Eigen::Ref<ArrayXDi> & grid_pts) {
            ArrayXDd xi(dim, grid_pts.cols());
            Ccoord nb_grid_pts;
            for (int j = 0; j < dim; ++j) {
              nb_grid_pts[j] = fft_freqs.get_nb_grid_pts(j);
            }
            for (int i = 0; i < grid_pts.cols(); ++i) {
              auto && grid_coords = grid_pts.col(i);
              Ccoord c;
              /* Wrap back to grid point */
              for (int j = 0; j < dim; ++j) {
                c[j] = grid_coords(j) % nb_grid_pts[j];
                if (c[j] < 0)
                  c[j] += nb_grid_pts[j];
              }
              xi.col(i) = fft_freqs.get_xi(c);
            }
            return xi;
          },
          "grid_pts"_a,
          "return wavevectors corresponding to the given grid indices");
}

void add_get_nb_hermitian(py::module & mod) {
  add_get_nb_hermitian_grid_pts_helper<muGrid::oneD>(mod);
  add_get_nb_hermitian_grid_pts_helper<muGrid::twoD>(mod);
  add_get_nb_hermitian_grid_pts_helper<muGrid::threeD>(mod);
}

template <muGrid::Index_t dim>
void add_get_index_helper(py::module & mod) {
  using Ccoord = muGrid::Ccoord_t<dim>;
  mod.def(
      "get_domain_index",
      [](Ccoord sizes, Ccoord ccoord) {
        return muGrid::CcoordOps::get_index<dim>(sizes, Ccoord{}, ccoord);
      },
      "sizes"_a, "ccoord"_a,
      "return the linear index corresponding to grid point 'ccoord' in a "
      "grid of size 'sizes'");
}

void add_fft_freqs(py::module & mod) {
  add_fft_freqs_helper<muGrid::oneD>(mod);
  add_fft_freqs_helper<muGrid::twoD>(mod);
  add_fft_freqs_helper<muGrid::threeD>(mod);
}

void add_common(py::module & mod) {
  add_version(mod);
  py::enum_<muFFT::FFT_PlanFlags>(mod, "FFT_PlanFlags")
      .value("estimate", muFFT::FFT_PlanFlags::estimate)
      .value("measure", muFFT::FFT_PlanFlags::measure)
      .value("patient", muFFT::FFT_PlanFlags::patient);

  add_get_nb_hermitian(mod);

  add_fft_freqs(mod);
}
