muspectre_python_sources = [
    'bind_py_module.cc',
    'bind_py_cell.cc',
    'bind_py_cell_data.cc',
    'bind_py_class_solvers.cc',
    'bind_py_common_muspectre.cc',
    'bind_py_fem_discretisation.cc',
    'bind_py_material.cc',
    'bind_py_material_linear_anisotropic.cc',
    'bind_py_material_linear_elastic1.cc',
    'bind_py_material_linear_elastic2.cc',
    'bind_py_material_linear_elastic3.cc',
    'bind_py_material_linear_elastic4.cc',
    'bind_py_material_hyper_elastic1.cc',
    'bind_py_material_hyper_elastic2.cc',
    'bind_py_material_linear_elastic_damage1.cc',
    'bind_py_material_linear_elastic_damage2.cc',
    'bind_py_material_linear_diffusion.cc',
    'bind_py_material_linear_orthotropic.cc',
    'bind_py_material_visco_elastic_ss.cc',
    'bind_py_material_visco_elastic_damage_ss1.cc',
    'bind_py_material_visco_elastic_damage_ss2.cc',
    'bind_py_material_neo_hookean_elastic.cc',
    'bind_py_material_hyper_elasto_plastic1.cc',
    'bind_py_material_hyper_elasto_plastic2.cc',
    'bind_py_material_linear_elastic_generic.cc',
    'bind_py_material_stochastic_plasticity.cc',
    'bind_py_material_phase_field_fracture.cc',
    'bind_py_material_phase_field_fracture2.cc',
    'bind_py_material_dunant.cc',
    'bind_py_krylov_solvers.cc',
    'bind_py_projections.cc',
    'bind_py_solvers.cc'
]

muspectre_python = python.extension_module(
    '_muSpectre', # Name of the module
    muspectre_python_sources,
    install: true,  # Install it
    include_directories: [eigen3_incdir, pybind11_incdir, include_directories('../../src')],
    link_with: [mugrid, mufft, muspectre],
    dependencies: [muspectre_dependencies, python3_dependency]
)

subdir('muSpectre')
