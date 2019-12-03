/*
  Copyright (C) 2018 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include "common.h"
#include <aspect/particle/property/lpo.h>
#include <deal.II/base/parameter_handler.h>
//#include <aspect/utilities.h>

// A test that verifies the LPO plugin

/*#include <aspect/simulator.h>
#include <deal.II/grid/tria.h>
#include <aspect/material_model/simple.h>
#include <aspect/simulator_access.h>

#include <iostream>*/

/*namespace
{
  using namespace dealii;
  using namespace aspect;

  template <int dim>
  class AdditionalOutputs1 : public MaterialModel::AdditionalMaterialOutputs<dim>
  {
    public:
      AdditionalOutputs1 (const unsigned int n_points,
                          const unsigned int / *n_comp* /)
      {
        additional_material_output1.resize(n_points);
      }

      std::vector<double> additional_material_output1;
  };


  template <int dim>
  class Material1 : public MaterialModel::Simple<dim>
  {
    public:

      virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &/ *in* /,
                            MaterialModel::MaterialModelOutputs<dim> &out) const
      {
        AdditionalOutputs1<dim> *additional;

        additional = out.template get_additional_output<AdditionalOutputs1<dim> >();
        additional->additional_material_output1[0] = 42.0;

      }


  };
}*/


TEST_CASE("LPO")
{
  using namespace dealii;
  using namespace aspect;

  std::cout << "test compute derivatives 1" << std::endl;
  {
    std::cout << "flag 1" << std::endl;
    const int dim2=2;

    // first test initialization 2d.
    Particle::Property::LPO<dim2> lpo_2d;
    ParameterHandler prm;
    lpo_2d.declare_parameters(prm);

    prm.enter_subsection("Postprocess");
    {
      prm.enter_subsection("Particles");
      {
        //prm.set("Number of particles","1"); // 2
        prm.enter_subsection("LPO");
        {
          prm.set("Random number seed","1"); // 2
          prm.set("Number of grains per praticle","5"); //10000;
          /*mobility = prm.get_double("Mobility"); //50;
          x_olivine = prm.get_double("Volume fraction olivine"); // 0.5;
          stress_exponent = prm.get_double("Stress exponents"); //3.5;
          exponent_p = prm.get_double("Exponents p"); //1.5;
          nucliation_efficientcy = prm.get_double("Nucliation efficientcy"); //5;
          threshold_GBS = prm.get_double("Threshold GBS"); //0.0;*/
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    lpo_2d.parse_parameters(prm);
    lpo_2d.initialize();


    Point<dim2> dummy_point;
    std::vector<double> data;
    lpo_2d.initialize_one_particle_property(dummy_point, data);

    // The LPO particles are initialized. With the same seed, the outcome should
    // always be the same, so test that for seed = 1. Forthermore, in the data
    // I can only really test that the first entry is the water content (0) and
    // that every first entry of each particle is 1/n_grains = 1/10 = 0.1.
    //for(unsigned int i = 0; i < data.size(); ++i)
    //  std::cout << "REQUIRE(data[" << i << "] == Approx(" << data[i] << "));" << std::endl;
    std::cout << "flag 2" << std::endl;
    //REQUIRE(data.size() == 50);
    REQUIRE(data[0] == Approx(0));
    REQUIRE(data[1] == Approx(0.2));
    REQUIRE(data[2] == Approx(0.159063));
    REQUIRE(data[3] == Approx(-0.11941));
    REQUIRE(data[4] == Approx(-0.0888556));
    REQUIRE(data[5] == Approx(-0.990362));
    REQUIRE(data[6] == Approx(0.2));
    REQUIRE(data[7] == Approx(0.409534));
    REQUIRE(data[8] == Approx(-0.340175));
    REQUIRE(data[9] == Approx(0.760572));
    REQUIRE(data[10] == Approx(0.639715));
    REQUIRE(data[11] == Approx(0.2));
    REQUIRE(data[12] == Approx(0.181749));
    REQUIRE(data[13] == Approx(0.896388));
    REQUIRE(data[14] == Approx(0.96535));
    REQUIRE(data[15] == Approx(-0.0843498));
    REQUIRE(data[16] == Approx(0.2));
    REQUIRE(data[17] == Approx(-0.677912));
    REQUIRE(data[18] == Approx(-0.184765));
    REQUIRE(data[19] == Approx(-0.589107));
    REQUIRE(data[20] == Approx(0.715529));
    REQUIRE(data[21] == Approx(0.2));
    REQUIRE(data[22] == Approx(0.741772));
    REQUIRE(data[23] == Approx(0.148462));
    REQUIRE(data[24] == Approx(0.0584955));
    REQUIRE(data[25] == Approx(-0.985796));
    REQUIRE(data[26] == Approx(0.2));
    REQUIRE(data[27] == Approx(0.0152744));
    REQUIRE(data[28] == Approx(-0.913698));
    REQUIRE(data[29] == Approx(-0.997266));
    REQUIRE(data[30] == Approx(0.0154452));
    REQUIRE(data[31] == Approx(0.2));
    REQUIRE(data[32] == Approx(0.828733));
    REQUIRE(data[33] == Approx(0.103259));
    REQUIRE(data[34] == Approx(0.437761));
    REQUIRE(data[35] == Approx(-0.73192));
    REQUIRE(data[36] == Approx(0.2));
    REQUIRE(data[37] == Approx(-0.912544));
    REQUIRE(data[38] == Approx(0.195949));
    REQUIRE(data[39] == Approx(-0.0787487));
    REQUIRE(data[40] == Approx(0.777138));
    REQUIRE(data[41] == Approx(0.2));
    REQUIRE(data[42] == Approx(0.185986));
    REQUIRE(data[43] == Approx(-0.973828));
    REQUIRE(data[44] == Approx(0.51699));
    REQUIRE(data[45] == Approx(0.210064));
    REQUIRE(data[46] == Approx(0.2));
    REQUIRE(data[47] == Approx(0.641855));
    REQUIRE(data[48] == Approx(0.56884));
    REQUIRE(data[49] == Approx(0.748989));
    REQUIRE(data[50] == Approx(-0.608861));


    // test function update_one_particle_property;

    // test function get_property_information

    // test function compute_runge_kutta

    // test function compute_derivatives
    /*
    compute_derivatives(const std::vector<double> &volume_fractions,
                                      const std::vector<Tensor<2,3> > &a_cosine_matrices,
                                      const SymmetricTensor<2,dim2> &strain_rate_nondimensional,
                                      const Tensor<2,dim2> &velocity_gradient_tensor_nondimensional,
                                      const DeformationType deformation_type,
                                      const std::array<double,4> &ref_resolved_shear_stress) const
    */

    std::vector<double> volume_fractions(5,0.2);
    std::cout << "test compute derivatives 1.0.1" << std::endl;
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
    std::cout << "test compute derivatives 1.0.2" << std::endl;
    a_cosine_matrices[0][0][0] = 0.5;
    a_cosine_matrices[0][0][1] = 0.5;
    a_cosine_matrices[0][0][2] = 0.5;
    a_cosine_matrices[0][1][0] = 0.5;
    a_cosine_matrices[0][1][1] = 0.5;
    a_cosine_matrices[0][1][2] = 0.5;
    a_cosine_matrices[0][2][0] = 0.5;
    a_cosine_matrices[0][2][1] = 0.5;
    a_cosine_matrices[0][2][2] = 0.5;

    std::cout << "test compute derivatives 2.0.0" << std::endl;
    a_cosine_matrices[1][0][0] = 0.1;
    a_cosine_matrices[1][0][1] = 0.2;
    a_cosine_matrices[1][0][2] = 0.3;
    a_cosine_matrices[1][1][0] = 0.4;
    a_cosine_matrices[1][1][1] = 0.5;
    a_cosine_matrices[1][1][2] = 0.6;
    a_cosine_matrices[1][2][0] = 0.7;
    a_cosine_matrices[1][2][1] = 0.8;
    a_cosine_matrices[1][2][2] = 0.9;
    std::cout << "test compute derivatives 3.0.0" << std::endl;

    a_cosine_matrices[2][0][0] = 0.1;
    a_cosine_matrices[2][0][1] = 0.2;
    a_cosine_matrices[2][0][2] = 0.3;
    a_cosine_matrices[2][1][0] = 0.4;
    a_cosine_matrices[2][1][1] = 0.5;
    a_cosine_matrices[2][1][2] = 0.6;
    a_cosine_matrices[2][2][0] = 0.7;
    a_cosine_matrices[2][2][1] = 0.8;
    a_cosine_matrices[2][2][2] = 0.9;
    std::cout << "test compute derivatives 4.0.0" << std::endl;

    a_cosine_matrices[3][0][0] = 0.1;
    a_cosine_matrices[3][0][1] = 0.2;
    a_cosine_matrices[3][0][2] = 0.3;
    a_cosine_matrices[3][1][0] = 0.4;
    a_cosine_matrices[3][1][1] = 0.5;
    a_cosine_matrices[3][1][2] = 0.6;
    a_cosine_matrices[3][2][0] = 0.7;
    a_cosine_matrices[3][2][1] = 0.8;
    a_cosine_matrices[3][2][2] = 0.9;
    std::cout << "test compute derivatives 5.0.0" << std::endl;

    a_cosine_matrices[4][0][0] = 0.1;
    a_cosine_matrices[4][0][1] = 0.2;
    a_cosine_matrices[4][0][2] = 0.3;
    a_cosine_matrices[4][1][0] = 0.4;
    a_cosine_matrices[4][1][1] = 0.5;
    a_cosine_matrices[4][1][2] = 0.6;
    a_cosine_matrices[4][2][0] = 0.7;
    a_cosine_matrices[4][2][1] = 0.8;
    a_cosine_matrices[4][2][2] = 0.9;
    std::cout << "test compute derivatives 6.0.0" << std::endl;
    // init a
    SymmetricTensor<2,dim2> strain_rate_nondimensional;
    strain_rate_nondimensional[0][1] = 0.5959;
    std::cout << "test compute derivatives 7.0.0" << std::endl;
    // init eta
    Tensor<2,dim2> velocity_gradient_tensor_nondimensional;
    velocity_gradient_tensor_nondimensional[0][1] = 2.0* 0.5959;
    velocity_gradient_tensor_nondimensional[1][0] = 2.0* 0.5959;
    std::cout << "test compute derivatives 8.0.0" << std::endl;
    // init grad_u
    Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::A_type;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.
    std::cout << "test compute derivatives 9.0.0" << std::endl;


    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;
    std::cout << "test compute derivatives 10" << std::endl;
    derivatives = lpo_2d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             deformation_type, ref_resolved_shear_stress);
    std::cout << "test compute derivatives 11" << std::endl;
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      std::cout << derivatives.first[i] << std::endl;
  }

  std::cout << "test compute derivatives 12" << std::endl;

  std::cout << std::endl << std::endl << "test compute derivatives part 2" << std::endl;
  {
    const int dim3=3;
    // first test initialization 2d.
    Particle::Property::LPO<dim3> lpo_3d;
    ParameterHandler prm;
    lpo_3d.declare_parameters(prm);

    prm.enter_subsection("Postprocess");
    {
      prm.enter_subsection("Particles");
      {
        //prm.set("Number of particles","1"); // 2
        prm.enter_subsection("LPO");
        {
          prm.set("Random number seed","1"); // 2
          prm.set("Number of grains per praticle","5"); //10000;
          /*mobility = prm.get_double("Mobility"); //50;
          x_olivine = prm.get_double("Volume fraction olivine"); // 0.5;
          stress_exponent = prm.get_double("Stress exponents"); //3.5;
          exponent_p = prm.get_double("Exponents p"); //1.5;
          nucliation_efficientcy = prm.get_double("Nucliation efficientcy"); //5;
          threshold_GBS = prm.get_double("Threshold GBS"); //0.0;*/
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    lpo_3d.parse_parameters(prm);
    lpo_3d.initialize();


    Point<dim3> dummy_point;
    std::vector<double> data;
    lpo_3d.initialize_one_particle_property(dummy_point, data);

    std::vector<double> volume_fractions(5,0.2);
    std::cout << "test compute derivatives 1.0.1" << std::endl;
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
    std::cout << "test compute derivatives 1.0.2" << std::endl;
    a_cosine_matrices[0][0][0] = 0.5;
    a_cosine_matrices[0][0][1] = 0.5;
    a_cosine_matrices[0][0][2] = 0.5;
    a_cosine_matrices[0][1][0] = 0.5;
    a_cosine_matrices[0][1][1] = 0.5;
    a_cosine_matrices[0][1][2] = 0.5;
    a_cosine_matrices[0][2][0] = 0.5;
    a_cosine_matrices[0][2][1] = 0.5;
    a_cosine_matrices[0][2][2] = 0.5;

    std::cout << "test compute derivatives 2.0.0" << std::endl;
    a_cosine_matrices[1][0][0] = 0.1;
    a_cosine_matrices[1][0][1] = 0.2;
    a_cosine_matrices[1][0][2] = 0.3;
    a_cosine_matrices[1][1][0] = 0.4;
    a_cosine_matrices[1][1][1] = 0.5;
    a_cosine_matrices[1][1][2] = 0.6;
    a_cosine_matrices[1][2][0] = 0.7;
    a_cosine_matrices[1][2][1] = 0.8;
    a_cosine_matrices[1][2][2] = 0.9;
    std::cout << "test compute derivatives 3.0.0" << std::endl;

    a_cosine_matrices[2][0][0] = 0.1;
    a_cosine_matrices[2][0][1] = 0.2;
    a_cosine_matrices[2][0][2] = 0.3;
    a_cosine_matrices[2][1][0] = 0.4;
    a_cosine_matrices[2][1][1] = 0.5;
    a_cosine_matrices[2][1][2] = 0.6;
    a_cosine_matrices[2][2][0] = 0.7;
    a_cosine_matrices[2][2][1] = 0.8;
    a_cosine_matrices[2][2][2] = 0.9;
    std::cout << "test compute derivatives 4.0.0" << std::endl;

    a_cosine_matrices[3][0][0] = 0.1;
    a_cosine_matrices[3][0][1] = 0.2;
    a_cosine_matrices[3][0][2] = 0.3;
    a_cosine_matrices[3][1][0] = 0.4;
    a_cosine_matrices[3][1][1] = 0.5;
    a_cosine_matrices[3][1][2] = 0.6;
    a_cosine_matrices[3][2][0] = 0.7;
    a_cosine_matrices[3][2][1] = 0.8;
    a_cosine_matrices[3][2][2] = 0.9;
    std::cout << "test compute derivatives 5.0.0" << std::endl;

    a_cosine_matrices[4][0][0] = 0.1;
    a_cosine_matrices[4][0][1] = 0.2;
    a_cosine_matrices[4][0][2] = 0.3;
    a_cosine_matrices[4][1][0] = 0.4;
    a_cosine_matrices[4][1][1] = 0.5;
    a_cosine_matrices[4][1][2] = 0.6;
    a_cosine_matrices[4][2][0] = 0.7;
    a_cosine_matrices[4][2][1] = 0.8;
    a_cosine_matrices[4][2][2] = 0.9;
    std::cout << "test compute derivatives 6.0.0" << std::endl;
    // init a
    SymmetricTensor<2,dim3> strain_rate_nondimensional; // e
    strain_rate_nondimensional[0][0] = 7.5;
    strain_rate_nondimensional[0][1] = 8;
    strain_rate_nondimensional[0][2] = 8.5;
    //strain_rate_nondimensional[1][0] = 9;
    strain_rate_nondimensional[1][1] = 9.5;
    strain_rate_nondimensional[1][2] = 10;
    //strain_rate_nondimensional[2][0] = 10.5;
    //strain_rate_nondimensional[2][1] = 11;
    strain_rate_nondimensional[2][2] = 11.5;
    std::cout << "test compute derivatives 7.0.0" << std::endl;
    // init eta
    Tensor<2,dim3> velocity_gradient_tensor_nondimensional; // l
    velocity_gradient_tensor_nondimensional[0][0] = 2;
    velocity_gradient_tensor_nondimensional[0][1] = 2.5;
    velocity_gradient_tensor_nondimensional[0][2] = 3;
    velocity_gradient_tensor_nondimensional[1][0] = 3.5;
    velocity_gradient_tensor_nondimensional[1][1] = 4;
    velocity_gradient_tensor_nondimensional[1][2] = 4.5;
    velocity_gradient_tensor_nondimensional[2][0] = 5;
    velocity_gradient_tensor_nondimensional[2][1] = 5.5;
    velocity_gradient_tensor_nondimensional[2][2] = 6;
    std::cout << "test compute derivatives 8.0.0" << std::endl;
    // init grad_u
    Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::A_type;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.
    std::cout << "test compute derivatives 9.0.0" << std::endl;


    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;
    std::cout << "test compute derivatives 10" << std::endl;
    derivatives = lpo_3d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             deformation_type, ref_resolved_shear_stress);
    std::cout << "test compute derivatives 11" << std::endl;
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      std::cout << derivatives.first[i] << std::endl;
  }
  /*
    REQUIRE(out.get_additional_output<AdditionalOutputs1<dim> >() == NULL);

    out.additional_outputs.push_back(std::make_shared<AdditionalOutputs1<dim> > (1, 1));

    REQUIRE(out.get_additional_output<AdditionalOutputs1<dim> >() != NULL);

    Material1<dim> mat;
    mat.evaluate(in, out);

    REQUIRE(out.get_additional_output<AdditionalOutputs1<dim> >()->additional_material_output1[0] == 42.0);

    // test const version of get_additional_output:
    {
      const MaterialModelOutputs<dim> &const_out = out;
      REQUIRE(const_out.get_additional_output<AdditionalOutputs1<dim> >() != NULL);
      const AdditionalOutputs1<dim> *a = const_out.get_additional_output<AdditionalOutputs1<dim> >();
      REQUIRE(a != NULL);
    }*/
}
