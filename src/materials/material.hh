/**
 * file   material.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   25 Oct 2017
 *
 * @brief  Base class for materials (constitutive models)
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <string>

#include "common/common.hh"
#include "common/field_map_tensor.hh"
#include "common/field_collection.hh"


#ifndef MATERIAL_H
#define MATERIAL_H

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class MaterialBase
  {
  public:
    //! typedefs for data handled by this interface
    //! global field collection for system-wide fields, like stress, strain, etc
    using GFieldCollection_t = FieldCollection<DimS, DimM, true>;
    //! field collection for internal variables, such as eigen-strains,
    //! plastic strains, damage variables, etc, but also for managing which
    //! pixels the material is responsible for
    using MFieldCollection_t = FieldCollection<DimS, DimM, false>;
    using StressMap_t = TensorFieldMap<GFieldCollection_t, Real, 2, DimM>;
    using StrainMap_t = StressMap_t;
    using StiffnessMap_t = TensorFieldMap<GFieldCollection_t, Real, 4, DimM>;
    using Ccoord = Ccoord_t<DimS>;
    //! Default constructor
    MaterialBase() = delete;

    //! Construct by name
    MaterialBase(std::string name);

    //! Copy constructor
    MaterialBase(const MaterialBase &other) = delete;

    //! Move constructor
    MaterialBase(MaterialBase &&other) noexcept = delete;

    //! Destructor
    virtual ~MaterialBase() noexcept = default;

    //! Copy assignment operator
    MaterialBase& operator=(const MaterialBase &other) = delete;

    //! Move assignment operator
    MaterialBase& operator=(MaterialBase &&other) noexcept = delete;


    //! take responsibility for a pixel identified by its cell coordinates
    //! TODO: this won't work. for materials with additional info per pixel (as, e.g. for eigenstrain), we need to pass more parameters, so this needs to be a variadic function.
    void add_pixel(const Ccoord & ccord);

    //! allocate memory, etc
    //! TODO: this won't work. for materials with additional info per pixel (see above TODO), we neet to allocate before we know for sure how many pixels the material is responsible for.
    virtual void initialize() = 0;


    //! computes the first Piola-Kirchhoff stress for finite strain problems
    virtual void compute_stress(const StrainMap_t & F,
                                StressMap_t & P);
    //! computes the first Piola-Kirchhoff stress and the tangent stiffness
    //! for finite strain problems
    virtual void compute_stress_stiffness(const StrainMap_t & F,
                                          StressMap_t & P,
                                          StiffnessMap_t & K);


    //! computes Cauchy stress for small strain problems
    virtual void compute_cauchy(const StrainMap_t & eps,
                                StressMap_t &sig);

    //! computes Cauchy stress and stiffness for small strain problems
    virtual void compute_cauchy_stiffness(const StrainMap_t & eps,
                                          StressMap_t &sig,
                                          StiffnessMap_t & K);
    //! return the materil's name
    const std::string & get_name() const;

    //! for static inheritance stuff
    constexpr static Dim_t sdim() {return DimS;}
    constexpr static Dim_t mdim() {return DimM;}

  protected:
    //! computes stress
    virtual void compute_stress(const StrainMap_t & F,
                                StressMap_t & P,
                                Formulation form);
    //! computes stress and tangent stiffness
    //! for finite strain problems
    virtual void compute_stress_stiffness(const StrainMap_t & F,
                                          StressMap_t & P,
                                          StiffnessMap_t & K,
                                          Formulation form);

    //! members
    const std::string name;
    MFieldCollection_t internal_fields;

  private:
  };
}  // muSpectre

#endif /* MATERIAL_H */
