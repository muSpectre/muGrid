#include "mpi_context.hh"
#include "tests.hh"

#include "libmugrid/field_typed.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/communicator.hh"
#include "libmugrid/cartesian_decomposition.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(mpi_decomposition_test);

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(domain_decomposition_test) {
    auto & comm{MPIContext::get_context().comm};

    // A large enough domain
    DynCcoord_t nb_domain_grid_pts{1, 10};

    // Decomposition only along one dimension
    DynCcoord_t nb_subdivisions{1, comm.size()};
    DynCcoord_t nb_ghosts_left{1, 0};
    DynCcoord_t nb_ghosts_right{1, 0};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Check the number of subdomain grid points adds up
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    int dim{1};
    auto res{comm.template sum<Index_t>(nb_subdomain_grid_pts[dim] -
                                        nb_ghosts_left[dim] -
                                        nb_ghosts_right[dim])};
    BOOST_CHECK_EQUAL(res, nb_domain_grid_pts[dim]);
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(inter_subdomain_communicate_test) {
    auto & comm{MPIContext::get_context().comm};

    // Decide the number of subdivisions according to number of processes.
    int nb_process{comm.size()};
    DynCcoord_t nb_subdivisions{};
    if (nb_process == 1)
      nb_subdivisions = DynCcoord_t{1};
    else if (nb_process == 2)
      nb_subdivisions = DynCcoord_t{2};
    else if (nb_process == 4)
      nb_subdivisions = DynCcoord_t{2, 2};
    else if (nb_process == 8)
      nb_subdivisions = DynCcoord_t{2, 2, 2};
    else
      throw RuntimeError("Not planned for this number of processes.");

    // Decide the size of the whole domain
    int spatial_dims{nb_subdivisions.size()};
    int nb_grid_pts_per_dim{5};
    const DynCcoord_t & nb_domain_grid_pts{
        DynCcoord_t(spatial_dims, nb_grid_pts_per_dim)};

    // A function to get referrence values
    auto && get_ref_value{
        [nb_grid_pts_per_dim](const DynCcoord_t & global_coords) {
          Index_t val{0};
          Index_t coeff{1};
          for (int dim{0}; dim < global_coords.size(); ++dim) {
            val += coeff * global_coords[dim];
            coeff *= nb_grid_pts_per_dim;
          }
          return val;
        }};

    // Create a Cartesian decomposition with ghost buffers
    const DynCcoord_t & nb_ghosts_left{DynCcoord_t(spatial_dims, 1)};
    const DynCcoord_t & nb_ghosts_right{DynCcoord_t(spatial_dims, 2)};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Create a field inside the collection for test
    auto & collection{cart_decomp.get_collection()};
    const Index_t nb_components{1};
    const std::string field_name{"test_field"};
    auto & field{collection.real_field(field_name, nb_components)};

    // Fill the field with some values
    auto subdomain_locations{cart_decomp.get_subdomain_locations_with_ghosts()};
    auto nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts_with_ghosts()};
    auto && field_map{field.get_sub_pt_map(Unknown)};
    CcoordOps::Pixels pixels{nb_subdomain_grid_pts};
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id{std::get<0>(pixel_id_coords)};
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_not_ghost_left{
          std::all_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_not_ghost_right{
          std::all_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      if (is_not_ghost_left && is_not_ghost_right) {
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        field_map[id] << get_ref_value(global_coords);
      } else {
        field_map[id] << -1;
      }
    }

    // Communicate the ghost buffer
    cart_decomp.communicate_ghosts(field_name);

    // Check the values at the ghost cells are correct
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id(std::get<0>(pixel_id_coords));
      auto && local_coords{std::get<1>(pixel_id_coords)};
      auto && global_coords{(subdomain_locations + local_coords) %
                            nb_domain_grid_pts};
      BOOST_CHECK_EQUAL(field_map[id].coeffRef(0, 0),
                        get_ref_value(global_coords));
    }
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muGrid
