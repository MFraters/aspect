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
#include <aspect/particle/property/lpo_elastic_tensor.h>
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
    REQUIRE(data[0] == Approx(0)); // default water value
    REQUIRE(data[1] == Approx(0.5)); // default volume fraction olivine
    REQUIRE(data[2] == Approx(0.2));
    REQUIRE(data[3] == Approx(0.159063));
    REQUIRE(data[4] == Approx(-0.11941));
    REQUIRE(data[5] == Approx(-0.0888556));
    REQUIRE(data[6] == Approx(-0.990362));
    REQUIRE(data[7] == Approx(0.2));
    REQUIRE(data[8] == Approx(0.409534));
    REQUIRE(data[9] == Approx(-0.340175));
    REQUIRE(data[10] == Approx(0.760572));
    REQUIRE(data[11] == Approx(0.639715));
    REQUIRE(data[12] == Approx(0.2));
    REQUIRE(data[13] == Approx(0.181749));
    REQUIRE(data[14] == Approx(0.896388));
    REQUIRE(data[15] == Approx(0.96535));
    REQUIRE(data[16] == Approx(-0.0843498));
    REQUIRE(data[17] == Approx(0.2));
    REQUIRE(data[18] == Approx(-0.677912));
    REQUIRE(data[19] == Approx(-0.184765));
    REQUIRE(data[20] == Approx(-0.589107));
    REQUIRE(data[21] == Approx(0.715529));
    REQUIRE(data[22] == Approx(0.2));
    REQUIRE(data[23] == Approx(0.741772));
    REQUIRE(data[24] == Approx(0.148462));
    REQUIRE(data[25] == Approx(0.0584955));
    REQUIRE(data[26] == Approx(-0.985796));
    REQUIRE(data[27] == Approx(0.2));
    REQUIRE(data[28] == Approx(0.0152744));
    REQUIRE(data[29] == Approx(-0.913698));
    REQUIRE(data[30] == Approx(-0.997266));
    REQUIRE(data[31] == Approx(0.0154452));
    REQUIRE(data[32] == Approx(0.2));
    REQUIRE(data[33] == Approx(0.828733));
    REQUIRE(data[34] == Approx(0.103259));
    REQUIRE(data[35] == Approx(0.437761));
    REQUIRE(data[36] == Approx(-0.73192));
    REQUIRE(data[37] == Approx(0.2));
    REQUIRE(data[38] == Approx(-0.912544));
    REQUIRE(data[39] == Approx(0.195949));
    REQUIRE(data[40] == Approx(-0.0787487));
    REQUIRE(data[41] == Approx(0.777138));
    REQUIRE(data[42] == Approx(0.2));
    REQUIRE(data[43] == Approx(0.185986));
    REQUIRE(data[44] == Approx(-0.973828));
    REQUIRE(data[45] == Approx(0.51699));
    REQUIRE(data[46] == Approx(0.210064));
    REQUIRE(data[47] == Approx(0.2));
    REQUIRE(data[48] == Approx(0.641855));
    REQUIRE(data[49] == Approx(0.56884));
    REQUIRE(data[50] == Approx(0.748989));
    REQUIRE(data[51] == Approx(-0.608861));


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

TEST_CASE("LPO elastic tensor")
{
  double volume_fraction_olivine = 0.7;
  std::vector<double> volume_fractions_olivine(8);
  std::vector<dealii::Tensor<2,3> > a_cosine_matrices_olivine(8);
  std::vector<double> volume_fractions_enstatite(8);
  std::vector<dealii::Tensor<2,3> > a_cosine_matrices_enstatite(8);
  dealii::Tensor<2,6> reference_elastic_tensor;
  dealii::Tensor<2,6> computed_elastic_tensor;

  // All these numbers are directly from the Fortran D-Rex
  // Had to fix the random seed to get consistent awnsers.
  // Fixed the random set to an array filled with zeros.
  a_cosine_matrices_olivine[0][0][0] = -0.87492387659370430;
  a_cosine_matrices_olivine[0][0][1] = -0.47600252255715020;
  a_cosine_matrices_olivine[0][0][2] = -0.10151800968122601;
  a_cosine_matrices_olivine[0][1][0] = 0.13036031917262200;
  a_cosine_matrices_olivine[0][1][1] = -2.6406769698713056E-002;
  a_cosine_matrices_olivine[0][1][2] = -0.99232315682823224;
  a_cosine_matrices_olivine[0][2][0] = 0.46957683898444408;
  a_cosine_matrices_olivine[0][2][1] = -0.87974004016081919;
  a_cosine_matrices_olivine[0][2][2] = 8.6098835151007691E-002;
  a_cosine_matrices_olivine[1][0][0] = -0.98046837857873570;
  a_cosine_matrices_olivine[1][0][1] = 0.19463893778994429;
  a_cosine_matrices_olivine[1][0][2] = -2.8239743415760400E-002;
  a_cosine_matrices_olivine[1][1][0] = 6.8942963409018593E-002;
  a_cosine_matrices_olivine[1][1][1] = 0.20565757913350766;
  a_cosine_matrices_olivine[1][1][2] = -0.97619251975954824;
  a_cosine_matrices_olivine[1][2][0] = -0.18419735979776022;
  a_cosine_matrices_olivine[1][2][1] = -0.95907282258851756;
  a_cosine_matrices_olivine[1][2][2] = -0.21505972600848655;
  a_cosine_matrices_olivine[2][0][0] = -0.60475851993637786;
  a_cosine_matrices_olivine[2][0][1] = 0.70843624030907060;
  a_cosine_matrices_olivine[2][0][2] = -0.36543256811748415;
  a_cosine_matrices_olivine[2][1][0] = 0.14301138974031016;
  a_cosine_matrices_olivine[2][1][1] = -0.35406726088673490;
  a_cosine_matrices_olivine[2][1][2] = -0.92567249145068109;
  a_cosine_matrices_olivine[2][2][0] = -0.78533679283378455;
  a_cosine_matrices_olivine[2][2][1] = -0.61106385979914768;
  a_cosine_matrices_olivine[2][2][2] = 0.11279851062549377;
  a_cosine_matrices_olivine[3][0][0] = 0.86791495614143876;
  a_cosine_matrices_olivine[3][0][1] = 0.11797028562530290;
  a_cosine_matrices_olivine[3][0][2] = -0.48441508819120616;
  a_cosine_matrices_olivine[3][1][0] = 0.29623368357270952;
  a_cosine_matrices_olivine[3][1][1] = 0.66075310029034506;
  a_cosine_matrices_olivine[3][1][2] = 0.69114249890201696;
  a_cosine_matrices_olivine[3][2][0] = 0.40054408016972737;
  a_cosine_matrices_olivine[3][2][1] = -0.74231595966649988;
  a_cosine_matrices_olivine[3][2][2] = 0.53869116329275069;
  a_cosine_matrices_olivine[4][0][0] = 6.2142369991145308E-002;
  a_cosine_matrices_olivine[4][0][1] = 0.99798524450925208;
  a_cosine_matrices_olivine[4][0][2] = 1.4733749724082583E-002;
  a_cosine_matrices_olivine[4][1][0] = 0.16313304505497195;
  a_cosine_matrices_olivine[4][1][1] = 3.9946223138032123E-003;
  a_cosine_matrices_olivine[4][1][2] = -0.99141604427573704;
  a_cosine_matrices_olivine[4][2][0] = -0.98947772660983668;
  a_cosine_matrices_olivine[4][2][1] = 6.3633828841832010E-002;
  a_cosine_matrices_olivine[4][2][2] = -0.16253511002663851;
  a_cosine_matrices_olivine[5][0][0] = 0.95811162445076037;
  a_cosine_matrices_olivine[5][0][1] = -0.24047036409864109;
  a_cosine_matrices_olivine[5][0][2] = -0.18394917461469307;
  a_cosine_matrices_olivine[5][1][0] = -0.26945849516984494;
  a_cosine_matrices_olivine[5][1][1] = -0.95216866624585605;
  a_cosine_matrices_olivine[5][1][2] = -0.14816174866727450;
  a_cosine_matrices_olivine[5][2][0] = -0.13948857525086517;
  a_cosine_matrices_olivine[5][2][1] = 0.18918653693665904;
  a_cosine_matrices_olivine[5][2][2] = -0.97685775746677384;
  a_cosine_matrices_olivine[6][0][0] = 0.54599545795872684;
  a_cosine_matrices_olivine[6][0][1] = -0.79260950430410815;
  a_cosine_matrices_olivine[6][0][2] = -0.27534584625644454;
  a_cosine_matrices_olivine[6][1][0] = -0.83567561116582212;
  a_cosine_matrices_olivine[6][1][1] = -0.55085590166106368;
  a_cosine_matrices_olivine[6][1][2] = -6.8746495709015629E-002;
  a_cosine_matrices_olivine[6][2][0] = -9.6081601263635075E-002;
  a_cosine_matrices_olivine[6][2][1] = 0.26479508638287669;
  a_cosine_matrices_olivine[6][2][2] = -0.96221681521835400;
  a_cosine_matrices_olivine[7][0][0] = 0.14540407652501869;
  a_cosine_matrices_olivine[7][0][1] = -0.61649901222701298;
  a_cosine_matrices_olivine[7][0][2] = -0.77881206212059828;
  a_cosine_matrices_olivine[7][1][0] = -0.59216315099851768;
  a_cosine_matrices_olivine[7][1][1] = -0.68282211925029168;
  a_cosine_matrices_olivine[7][1][2] = 0.43341920614609419;
  a_cosine_matrices_olivine[7][2][0] = -0.79819736735552915;
  a_cosine_matrices_olivine[7][2][1] = 0.39350172081601731;
  a_cosine_matrices_olivine[7][2][2] = -0.46322016633770996;

  volume_fractions_olivine[0] = 2.5128593570287589E-002;
  volume_fractions_olivine[1] = 0.83128842847575013;
  volume_fractions_olivine[2] = 2.4387041141724769E-002;
  volume_fractions_olivine[3] = 2.4763275182773107E-002;
  volume_fractions_olivine[4] = 2.4801714431754770E-002;
  volume_fractions_olivine[5] = 2.3943562805875843E-002;
  volume_fractions_olivine[6] = 2.1493810045792379E-002;
  volume_fractions_olivine[7] = 2.4193574346041427E-002;

  a_cosine_matrices_enstatite[0][0][0] = -0.66168933252008499;
  a_cosine_matrices_enstatite[0][0][1] = -0.27722421136423192;
  a_cosine_matrices_enstatite[0][0][2] = 0.70016104334335305;
  a_cosine_matrices_enstatite[0][1][0] = -0.45052346117292291;
  a_cosine_matrices_enstatite[0][1][1] = -0.59946225647123619;
  a_cosine_matrices_enstatite[0][1][2] = -0.66370395454608444;
  a_cosine_matrices_enstatite[0][2][0] = 0.60350910238307343;
  a_cosine_matrices_enstatite[0][2][1] = -0.75116099269655345;
  a_cosine_matrices_enstatite[0][2][2] = 0.27207473610935284;
  a_cosine_matrices_enstatite[1][0][0] = 0.70309122563849258;
  a_cosine_matrices_enstatite[1][0][1] = -0.23393734574397834;
  a_cosine_matrices_enstatite[1][0][2] = -0.67775637808568567;
  a_cosine_matrices_enstatite[1][1][0] = 0.68298185906364617;
  a_cosine_matrices_enstatite[1][1][1] = -6.6406501211459870E-002;
  a_cosine_matrices_enstatite[1][1][2] = 0.73168002139436472;
  a_cosine_matrices_enstatite[1][2][0] = -0.21654097292603913;
  a_cosine_matrices_enstatite[1][2][1] = -0.97001835897279087;
  a_cosine_matrices_enstatite[1][2][2] = 0.11341644508992850;
  a_cosine_matrices_enstatite[2][0][0] = -0.37892641930874921;
  a_cosine_matrices_enstatite[2][0][1] = 0.92610670487847191;
  a_cosine_matrices_enstatite[2][0][2] = 3.9338024409460159E-002;
  a_cosine_matrices_enstatite[2][1][0] = -0.10014673867324062;
  a_cosine_matrices_enstatite[2][1][1] = 2.1922389732184896E-004;
  a_cosine_matrices_enstatite[2][1][2] = -0.99514251026853373;
  a_cosine_matrices_enstatite[2][2][0] = -0.92466780075091537;
  a_cosine_matrices_enstatite[2][2][1] = -0.37833500123197994;
  a_cosine_matrices_enstatite[2][2][2] = 9.2539397053637271E-002;
  a_cosine_matrices_enstatite[3][0][0] = 0.73973803569795438;
  a_cosine_matrices_enstatite[3][0][1] = -0.58084801380447959;
  a_cosine_matrices_enstatite[3][0][2] = -0.35134329241205425;
  a_cosine_matrices_enstatite[3][1][0] = 0.30883050284698427;
  a_cosine_matrices_enstatite[3][1][1] = -0.17349072757172229;
  a_cosine_matrices_enstatite[3][1][2] = 0.94047567596335968;
  a_cosine_matrices_enstatite[3][2][0] = -0.60686583881969203;
  a_cosine_matrices_enstatite[3][2][1] = -0.79564553570809793;
  a_cosine_matrices_enstatite[3][2][2] = 5.0387432541084881E-002;
  a_cosine_matrices_enstatite[4][0][0] = 4.2730545437086563E-002;
  a_cosine_matrices_enstatite[4][0][1] = 0.99790446393350407;
  a_cosine_matrices_enstatite[4][0][2] = 4.9677348699965519E-002;
  a_cosine_matrices_enstatite[4][1][0] = 0.24607319829629554;
  a_cosine_matrices_enstatite[4][1][1] = 3.7594987859276820E-002;
  a_cosine_matrices_enstatite[4][1][2] = -0.97243353311600111;
  a_cosine_matrices_enstatite[4][2][0] = -0.97326576151708344;
  a_cosine_matrices_enstatite[4][2][1] = 5.2990062123162360E-002;
  a_cosine_matrices_enstatite[4][2][2] = -0.24420040893728351;
  a_cosine_matrices_enstatite[5][0][0] = -0.92112012021624434;
  a_cosine_matrices_enstatite[5][0][1] = 0.17590913101287936;
  a_cosine_matrices_enstatite[5][0][2] = 0.34726602737040008;
  a_cosine_matrices_enstatite[5][1][0] = -0.27292179963530816;
  a_cosine_matrices_enstatite[5][1][1] = -0.92793421776529450;
  a_cosine_matrices_enstatite[5][1][2] = -0.25387354780599874;
  a_cosine_matrices_enstatite[5][2][0] = 0.27758135269283285;
  a_cosine_matrices_enstatite[5][2][1] = -0.32862450412044819;
  a_cosine_matrices_enstatite[5][2][2] = 0.90274831467569783;
  a_cosine_matrices_enstatite[6][0][0] = -4.6028553384029648E-002;
  a_cosine_matrices_enstatite[6][0][1] = -0.82828827663204008;
  a_cosine_matrices_enstatite[6][0][2] = -0.56150509327989218;
  a_cosine_matrices_enstatite[6][1][0] = -0.62780239844730912;
  a_cosine_matrices_enstatite[6][1][1] = -0.41264414122407550;
  a_cosine_matrices_enstatite[6][1][2] = 0.66416683570654200;
  a_cosine_matrices_enstatite[6][2][0] = -0.78150657001547463;
  a_cosine_matrices_enstatite[6][2][1] = 0.37975977805647526;
  a_cosine_matrices_enstatite[6][2][2] = -0.50060220610535922;
  a_cosine_matrices_enstatite[7][0][0] = -0.69974343277254114;
  a_cosine_matrices_enstatite[7][0][1] = -0.60514581197791628;
  a_cosine_matrices_enstatite[7][0][2] = 0.38074455421574593;
  a_cosine_matrices_enstatite[7][1][0] = 0.27181144918830025;
  a_cosine_matrices_enstatite[7][1][1] = -0.71756260285634454;
  a_cosine_matrices_enstatite[7][1][2] = -0.64160793166977359;
  a_cosine_matrices_enstatite[7][2][0] = 0.66130789657394939;
  a_cosine_matrices_enstatite[7][2][1] = -0.34496383827187421;
  a_cosine_matrices_enstatite[7][2][2] = 0.66656470768691123;

  volume_fractions_enstatite[0] = 2.4802805652404735E-002;
  volume_fractions_enstatite[1] = 2.4917064186342413E-002;
  volume_fractions_enstatite[2] = 2.4790722064907112E-002;
  volume_fractions_enstatite[3] = 2.4932751194567736E-002;
  volume_fractions_enstatite[4] = 2.4896744967217745E-002;
  volume_fractions_enstatite[5] = 0.79902139535472505;
  volume_fractions_enstatite[6] = 2.4861121542485400E-002;
  volume_fractions_enstatite[7] = 5.1777395037349808E-002;

  reference_elastic_tensor[0][0] = 282.99195951271281;
  reference_elastic_tensor[0][1] = 74.161110997372660;
  reference_elastic_tensor[0][2] = 69.528000044099443;
  reference_elastic_tensor[0][3] = 0.85449958913948032;
  reference_elastic_tensor[0][4] = 0.15156631865980030;
  reference_elastic_tensor[0][5] = -10.295196344728696;
  reference_elastic_tensor[1][0] = 74.161110997372674;
  reference_elastic_tensor[1][1] = 223.44404040361212;
  reference_elastic_tensor[1][2] = 70.938305212304968;
  reference_elastic_tensor[1][3] = 1.4935368335052783;
  reference_elastic_tensor[1][4] = -1.5555954844298270;
  reference_elastic_tensor[1][5] = -3.5461136157235025;
  reference_elastic_tensor[2][0] = 69.528000044099443;
  reference_elastic_tensor[2][1] = 70.938305212304940;
  reference_elastic_tensor[2][2] = 208.89404139751849;
  reference_elastic_tensor[2][3] = 0.48656291118743428;
  reference_elastic_tensor[2][4] = 0.49424401802786699;
  reference_elastic_tensor[2][5] = -0.15651560991351129;
  reference_elastic_tensor[3][0] = 0.85449958913948365;
  reference_elastic_tensor[3][1] = 1.4935368335052732;
  reference_elastic_tensor[3][2] = 0.48656291118742612;
  reference_elastic_tensor[3][3] = 71.502776245041289;
  reference_elastic_tensor[3][4] = -2.3939425644793477;
  reference_elastic_tensor[3][5] = 0.62714033620769472;
  reference_elastic_tensor[4][0] = 0.15156631865980935;
  reference_elastic_tensor[4][1] = -1.5555954844298292;
  reference_elastic_tensor[4][2] = 0.49424401802786488;
  reference_elastic_tensor[4][3] = -2.3939425644793459;
  reference_elastic_tensor[4][4] = 78.565442407530099;
  reference_elastic_tensor[4][5] = -0.69890743507815323;
  reference_elastic_tensor[5][0] = -10.295196344728710;
  reference_elastic_tensor[5][1] = -3.5461136157235131;
  reference_elastic_tensor[5][2] = -0.15651560991350758;
  reference_elastic_tensor[5][3] = 0.62714033620769460;
  reference_elastic_tensor[5][4] = -0.69890743507815523;
  reference_elastic_tensor[5][5] = 80.599981331604567;



  aspect::Particle::Property::LpoElasticTensor<3> lpo_elastic_tensor;
  computed_elastic_tensor = lpo_elastic_tensor.compute_elastic_tensor(volume_fraction_olivine,
                                                                      volume_fractions_olivine,
                                                                      a_cosine_matrices_olivine,
                                                                      volume_fractions_enstatite,
                                                                      a_cosine_matrices_enstatite
                                                                     );


  for (size_t i = 0; i < 6; i++)
    {
      for (size_t j = 0; j < 6; j++)
        {
          REQUIRE(computed_elastic_tensor[i][j] == Approx(reference_elastic_tensor[i][j]));
        }
    }


}
