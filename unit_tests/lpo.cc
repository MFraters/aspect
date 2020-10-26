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
#include <aspect/postprocess/particle_lpo.h>
#include <deal.II/base/array_view.h>
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


/**
 * Compare the given two std::array<double,3> entries with an epsilon (using Catch::Approx)
 */
inline void compare_3d_arrays_approx(
  const std::array<double,3> &computed,
  const std::array<double,3> &expected)
{
  CHECK(computed.size() == expected.size());
  for (unsigned int i=0; i< computed.size(); ++i)
    {
      INFO("vector index i=" << i << ": ");
      CHECK(computed[i] == Approx(expected[i]));
    }
}


/**
 * Compare the given two std::array<double,3> entries with an epsilon (using Catch::Approx)
 */
inline void compare_3d_arrays_approx(
  const std::vector<double> &computed,
  const std::vector<double> &expected)
{
  CHECK(computed.size() == expected.size());
  for (unsigned int i=0; i< computed.size(); ++i)
    {
      INFO("vector index i=" << i << ": ");
      CHECK(computed[i] == Approx(expected[i]));
    }
}



/**
 * Compare two rotation matrices
 */
inline void compare_rotation_matrices_approx(
  const std::array<std::array<double,3>,3> &computed,
  const std::array<std::array<double,3>,3> &expected)
{
  // sign of eigenvector is not important
  INFO("rotation matrices are not the same: \n" <<
       "expected = " << expected[0][0] << " " << expected[0][1] << " " << expected[0][2] << "\n" <<
       "           " << expected[1][0] << " " << expected[1][1] << " " << expected[1][2] << "\n" <<
       "           " << expected[2][0] << " " << expected[2][1] << " " << expected[2][2] << "\n" <<
       "computed = " << computed[0][0] << " " << computed[0][1] << " " << computed[0][2] << "\n" <<
       "           " << computed[1][0] << " " << computed[1][1] << " " << computed[1][2] << "\n" <<
       "           " << computed[2][0] << " " << computed[2][1] << " " << computed[2][2] << "\n" );
  CHECK((
          (computed[0][0] == Approx(expected[0][0]) && computed[0][1] == Approx(expected[0][1]) && computed[0][2] == Approx(expected[0][2]) &&
           computed[1][0] == Approx(expected[1][0]) && computed[1][1] == Approx(expected[1][1]) && computed[1][2] == Approx(expected[1][2]) &&
           computed[2][0] == Approx(expected[2][0]) && computed[2][1] == Approx(expected[2][1]) && computed[2][2] == Approx(expected[2][2]))
          ||
          (computed[0][0] == Approx(-expected[0][0]) && computed[0][1] == Approx(-expected[0][1]) && computed[0][2] == Approx(-expected[0][2]) &&
           computed[1][0] == Approx(-expected[1][0]) && computed[1][1] == Approx(-expected[1][1]) && computed[1][2] == Approx(-expected[1][2]) &&
           computed[2][0] == Approx(-expected[2][0]) && computed[2][1] == Approx(-expected[2][1]) && computed[2][2] == Approx(-expected[2][2]))));
}

/**
 * Compare two rotation matrices
 */
inline void compare_rotation_matrices_approx(
  const dealii::Tensor<2,3> &computed,
  const dealii::Tensor<2,3> &expected)
{
  // sign of eigenvector is not important
  INFO("rotation matrices are not the same: \n" <<
       "expected = " << expected[0][0] << " " << expected[0][1] << " " << expected[0][2] << "\n" <<
       "           " << expected[1][0] << " " << expected[1][1] << " " << expected[1][2] << "\n" <<
       "           " << expected[2][0] << " " << expected[2][1] << " " << expected[2][2] << "\n" <<
       "computed = " << computed[0][0] << " " << computed[0][1] << " " << computed[0][2] << "\n" <<
       "           " << computed[1][0] << " " << computed[1][1] << " " << computed[1][2] << "\n" <<
       "           " << computed[2][0] << " " << computed[2][1] << " " << computed[2][2] << "\n" );
  const double tol = 1e-14;
  CHECK((
          ((computed[0][0] == Approx(expected[0][0]) || std::fabs(computed[0][0] < tol))
           && (computed[0][1] == Approx(expected[0][1]) || std::fabs(computed[0][1] < tol))
           && (computed[0][2] == Approx(expected[0][2]) || std::fabs(computed[0][2] < tol))
           && (computed[1][0] == Approx(expected[1][0]) || std::fabs(computed[1][0] < tol))
           && (computed[1][1] == Approx(expected[1][1]) || std::fabs(computed[1][1] < tol))
           && (computed[1][2] == Approx(expected[1][2]) || std::fabs(computed[1][2] < tol))
           && (computed[2][0] == Approx(expected[2][0]) || std::fabs(computed[2][0] < tol))
           && (computed[2][1] == Approx(expected[2][1]) || std::fabs(computed[2][1] < tol))
           && (computed[2][2] == Approx(expected[2][2]) || std::fabs(computed[2][2] < tol)))
          ||
          ((computed[0][0] == Approx(-expected[0][0]) || std::fabs(computed[0][0] < tol))
           && (computed[0][1] == Approx(-expected[0][1]) || std::fabs(computed[0][1] < tol))
           && (computed[0][2] == Approx(-expected[0][2]) || std::fabs(computed[0][2] < tol))
           && (computed[1][0] == Approx(-expected[1][0]) || std::fabs(computed[1][0] < tol))
           && (computed[1][1] == Approx(-expected[1][1]) || std::fabs(computed[1][1] < tol))
           && (computed[1][2] == Approx(-expected[1][2]) || std::fabs(computed[1][2] < tol))
           && (computed[2][0] == Approx(-expected[2][0]) || std::fabs(computed[2][0] < tol))
           && (computed[2][1] == Approx(-expected[2][1]) || std::fabs(computed[2][1] < tol))
           && (computed[2][2] == Approx(-expected[2][2]) || std::fabs(computed[2][2] < tol)))));
}

TEST_CASE("Fabric determination function")
{
  using namespace aspect;
  using namespace Particle::Property;
  LPO<3> lpo;
  double MPa = 1e6;

  CHECK(lpo.determine_deformation_type(379.*MPa, 0.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(381.*MPa, 0.) == DeformationType::D_type);
  CHECK(lpo.determine_deformation_type(0.*MPa, 100.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 50.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 50.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(379.*MPa, 50.) == DeformationType::D_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 49.) == DeformationType::D_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 75.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 100.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 100.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(379.*MPa, 100.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 100.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 200.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 200.) == DeformationType::A_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 200.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 200.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 200.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 200.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 300.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 300.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 300.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 300.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 300.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 300.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 380.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 380.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 380.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 380.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 380.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 380.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 380.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 400.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 400.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 400.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 400.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 400.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 400.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 400.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 600.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 600.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 600.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 600.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 600.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 600.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 600.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 800.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 800.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 800.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 800.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 800.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 800.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 800.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 1000.) == DeformationType::E_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 1000.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 1000.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 1000.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 1000.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 1000.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 1000.) == DeformationType::B_type);

  CHECK(lpo.determine_deformation_type(20.*MPa, 1200.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(100.*MPa, 1200.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(200.*MPa, 1200.) == DeformationType::C_type);
  CHECK(lpo.determine_deformation_type(340.*MPa, 1200.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(360.*MPa, 1200.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(380.*MPa, 1200.) == DeformationType::B_type);
  CHECK(lpo.determine_deformation_type(480.*MPa, 1200.) == DeformationType::B_type);
}

TEST_CASE("Euler angle functions")
{
  using namespace aspect;
  {
    Postprocess::LPO<3> lpo;
    std::array<std::array<double,3>,3> array = {{{{0.36,0.48,-0.8}},{{-0.8,0.6,0}}, {{0.48,0.64, 0.6}}}};
    dealii::Tensor<2,3> rot1;
    rot1[0][0] = 0.36;
    rot1[0][1] = 0.48;
    rot1[0][2] = -0.8;

    rot1[1][0] = -0.8;
    rot1[1][1] = 0.6;
    rot1[1][2] = 0;

    rot1[1][0] = 0.48;
    rot1[1][1] = 0.64;
    rot1[1][2] = 0.6;

    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot1);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot1);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }

  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{20,30,40}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(20,30,40);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }

  // note that in the case theta is 0 or phi a dimension is lost
  // see: https://en.wikipedia.org/wiki/Gimbal_lock. We set phi1
  // to 0 and compute the corresponding phi2. The resulting direction
  // (cosine matrix) should be the same, but the euler angles will change
  // the first time.
  // For theta = 0, the sum of phi1 and phi2 should still be the same.
  // For theta = phi, the difference between phi1 and phi2 should remain the same.
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0,0,0}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(0,0,0);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }

  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0,0,80}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(20,0,60);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0.0,0,70}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(30.0,0,40);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0,0,140}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(240,0,260);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0,180,20}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(20,180,40);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0,180,40}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(20,180,60);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }
  {
    Postprocess::LPO<3> lpo;
    std::vector<double> ea0 = {{0.0,180,20}};
    auto rot0 = lpo.euler_angles_to_rotation_matrix(20,-180,40);
    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot0);
    compare_3d_arrays_approx(ea1,ea0);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot0);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }

}


TEST_CASE("LPO")
{
  using namespace dealii;
  using namespace aspect;

  std::cout << "test compute derivatives 1" << std::endl;
  {
    // first test initialization 2d.
    const int dim2=2;

    Particle::Property::LPO<dim2> lpo_2d;
    std::cout << "test compute derivatives 1.0.1" << std::endl;
    ParameterHandler prm;
    std::cout << "test compute derivatives 1.0.2" << std::endl;
    lpo_2d.declare_parameters(prm);
    std::cout << "test compute derivatives 1.0.3" << std::endl;
    prm.declare_entry("World builder file", "", dealii::Patterns::Anything(), "");
    prm.set("World builder file", "/home/fraters/Documents/post-doc/2019-02-01-magali/code/aspect/aspect-C/build-dg-u-nj/advection_test.wb");
    std::cout << "test compute derivatives 1.0.4" << std::endl;
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
          nucleation_efficientcy = prm.get_double("Nucleation efficientcy"); //5;
          threshold_GBS = prm.get_double("Threshold GBS"); //0.0;*/
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
    prm.leave_subsection ();

    std::cout << "test compute derivatives 1.0" << std::endl;
    lpo_2d.parse_parameters(prm);
    std::cout << "test compute derivatives 1.1" << std::endl;
    lpo_2d.initialize();
    std::cout << "test compute derivatives 1.2" << std::endl;


    Point<dim2> dummy_point;
    std::vector<double> data;
    lpo_2d.initialize_one_particle_property(dummy_point, data);
    std::cout << "test compute derivatives 1.3" << std::endl;

    // The LPO particles are initialized. With the same seed, the outcome should
    // always be the same, so test that for seed = 1. Forthermore, in the data
    // I can only really test that the first entry is the water content (0) and
    // that every first entry of each particle is 1/n_grains = 1/10 = 0.1.
    CHECK(data[0] == Approx(0)); // default water value
    CHECK(data[1] == Approx(0.5)); // default volume fraction olivine
    CHECK(data[2] == Approx(0.2));
    CHECK(data[3] == Approx(0.159063));
    CHECK(data[4] == Approx(-0.11941));
    CHECK(data[5] == Approx(0.9800204275));
    CHECK(data[6] == Approx(-0.0888556));
    CHECK(data[7] == Approx(-0.990362));
    CHECK(data[8] == Approx(-0.1062486256));
    CHECK(data[9] == Approx(0.983261702));
    CHECK(data[10] == Approx(-0.0701800114));
    CHECK(data[11] == Approx(-0.1681403917));
    CHECK(data[12] == Approx(0.2));
    CHECK(data[13] == Approx(0.4095335744));
    CHECK(data[14] == Approx(-0.3401753011));
    CHECK(data[15] == Approx(0.8465004524));
    CHECK(data[16] == Approx(0.7605716382));
    CHECK(data[17] == Approx(0.639714977));
    CHECK(data[18] == Approx(-0.1108852174));
    CHECK(data[19] == Approx(-0.5037986052));
    CHECK(data[20] == Approx(0.6892354553));
    CHECK(data[21] == Approx(0.5207124471));
    CHECK(data[22] == Approx(0.2));
    CHECK(data[32] == Approx(0.2));
    CHECK(data[42] == Approx(0.2));
    CHECK(data[52] == Approx(0.2));
    CHECK(data[62] == Approx(0.2));
    CHECK(data[72] == Approx(0.2));
    CHECK(data[82] == Approx(0.2));
    CHECK(data[92] == Approx(0.2));

    std::vector<double> volume_fractions(5,0.2);
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
    std::cout << "test compute derivatives 2.0.0" << std::endl;
    a_cosine_matrices[0][0][0] = 0.5;
    a_cosine_matrices[0][0][1] = 0.5;
    a_cosine_matrices[0][0][2] = 0.5;
    a_cosine_matrices[0][1][0] = 0.5;
    a_cosine_matrices[0][1][1] = 0.5;
    a_cosine_matrices[0][1][2] = 0.5;
    a_cosine_matrices[0][2][0] = 0.5;
    a_cosine_matrices[0][2][1] = 0.5;
    a_cosine_matrices[0][2][2] = 0.5;

    a_cosine_matrices[1][0][0] = 0.1;
    a_cosine_matrices[1][0][1] = 0.2;
    a_cosine_matrices[1][0][2] = 0.3;
    a_cosine_matrices[1][1][0] = 0.4;
    a_cosine_matrices[1][1][1] = 0.5;
    a_cosine_matrices[1][1][2] = 0.6;
    a_cosine_matrices[1][2][0] = 0.7;
    a_cosine_matrices[1][2][1] = 0.8;
    a_cosine_matrices[1][2][2] = 0.9;

    a_cosine_matrices[2][0][0] = 0.1;
    a_cosine_matrices[2][0][1] = 0.2;
    a_cosine_matrices[2][0][2] = 0.3;
    a_cosine_matrices[2][1][0] = 0.4;
    a_cosine_matrices[2][1][1] = 0.5;
    a_cosine_matrices[2][1][2] = 0.6;
    a_cosine_matrices[2][2][0] = 0.7;
    a_cosine_matrices[2][2][1] = 0.8;
    a_cosine_matrices[2][2][2] = 0.9;

    a_cosine_matrices[3][0][0] = 0.1;
    a_cosine_matrices[3][0][1] = 0.2;
    a_cosine_matrices[3][0][2] = 0.3;
    a_cosine_matrices[3][1][0] = 0.4;
    a_cosine_matrices[3][1][1] = 0.5;
    a_cosine_matrices[3][1][2] = 0.6;
    a_cosine_matrices[3][2][0] = 0.7;
    a_cosine_matrices[3][2][1] = 0.8;
    a_cosine_matrices[3][2][2] = 0.9;

    a_cosine_matrices[4][0][0] = 0.1;
    a_cosine_matrices[4][0][1] = 0.2;
    a_cosine_matrices[4][0][2] = 0.3;
    a_cosine_matrices[4][1][0] = 0.4;
    a_cosine_matrices[4][1][1] = 0.5;
    a_cosine_matrices[4][1][2] = 0.6;
    a_cosine_matrices[4][2][0] = 0.7;
    a_cosine_matrices[4][2][1] = 0.8;
    a_cosine_matrices[4][2][2] = 0.9;

    // init a
    SymmetricTensor<2,3> strain_rate_nondimensional;
    strain_rate_nondimensional[0][1] = 0.5959;

    // init eta
    Tensor<2,3> velocity_gradient_tensor_nondimensional;
    velocity_gradient_tensor_nondimensional[0][1] = 2.0* 0.5959;
    velocity_gradient_tensor_nondimensional[1][0] = 2.0* 0.5959;

    // init grad_u
    Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::A_type;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.

    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;
    std::cout << "test compute derivatives 10" << std::endl;
    derivatives = lpo_2d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             deformation_type, ref_resolved_shear_stress);
    std::cout << "test compute derivatives 11" << std::endl;
    // The correct analytical solution to check against
    double solution[5] = {0.63011275122, -0.157528187805, -0.157528187805, -0.157528187805 ,-0.157528187805};
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      REQUIRE(derivatives.first[i] == Approx(solution[i]));
  }

  std::cout << "test compute derivatives 12" << std::endl;

  std::cout << std::endl << std::endl << "test compute derivatives part 2" << std::endl;
  {
    // secondly test initialization 3d.
    // This should be exactly the same as the 2d version
    const int dim3=3;

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
          nucleation_efficientcy = prm.get_double("Nucleation efficientcy"); //5;
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

    // The LPO particles are initialized. With the same seed, the outcome should
    // always be the same, so test that for seed = 1. Forthermore, in the data
    // I can only really test that the first entry is the water content (0) and
    // that every first entry of each particle is 1/n_grains = 1/10 = 0.1.
    CHECK(data[0] == Approx(0)); // default water value
    CHECK(data[1] == Approx(0.5)); // default volume fraction olivine
    CHECK(data[2] == Approx(0.2));
    CHECK(data[3] == Approx(0.159063));
    CHECK(data[4] == Approx(-0.11941));
    CHECK(data[5] == Approx(0.9800204275));
    CHECK(data[6] == Approx(-0.0888556));
    CHECK(data[7] == Approx(-0.990362));
    CHECK(data[8] == Approx(-0.1062486256));
    CHECK(data[9] == Approx(0.983261702));
    CHECK(data[10] == Approx(-0.0701800114));
    CHECK(data[11] == Approx(-0.1681403917));
    CHECK(data[12] == Approx(0.2));
    CHECK(data[13] == Approx(0.4095335744));
    CHECK(data[14] == Approx(-0.3401753011));
    CHECK(data[15] == Approx(0.8465004524));
    CHECK(data[16] == Approx(0.7605716382));
    CHECK(data[17] == Approx(0.639714977));
    CHECK(data[18] == Approx(-0.1108852174));
    CHECK(data[19] == Approx(-0.5037986052));
    CHECK(data[20] == Approx(0.6892354553));
    CHECK(data[21] == Approx(0.5207124471));
    CHECK(data[22] == Approx(0.2));
    CHECK(data[32] == Approx(0.2));
    CHECK(data[42] == Approx(0.2));
    CHECK(data[52] == Approx(0.2));
    CHECK(data[62] == Approx(0.2));
    CHECK(data[72] == Approx(0.2));
    CHECK(data[82] == Approx(0.2));
    CHECK(data[92] == Approx(0.2));

    std::vector<double> volume_fractions(5,0.2);
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
    std::cout << "test compute derivatives 1.0.0" << std::endl;
    a_cosine_matrices[0][0][0] = 0.5;
    a_cosine_matrices[0][0][1] = 0.5;
    a_cosine_matrices[0][0][2] = 0.5;
    a_cosine_matrices[0][1][0] = 0.5;
    a_cosine_matrices[0][1][1] = 0.5;
    a_cosine_matrices[0][1][2] = 0.5;
    a_cosine_matrices[0][2][0] = 0.5;
    a_cosine_matrices[0][2][1] = 0.5;
    a_cosine_matrices[0][2][2] = 0.5;

    a_cosine_matrices[1][0][0] = 0.1;
    a_cosine_matrices[1][0][1] = 0.2;
    a_cosine_matrices[1][0][2] = 0.3;
    a_cosine_matrices[1][1][0] = 0.4;
    a_cosine_matrices[1][1][1] = 0.5;
    a_cosine_matrices[1][1][2] = 0.6;
    a_cosine_matrices[1][2][0] = 0.7;
    a_cosine_matrices[1][2][1] = 0.8;
    a_cosine_matrices[1][2][2] = 0.9;

    a_cosine_matrices[2][0][0] = 0.1;
    a_cosine_matrices[2][0][1] = 0.2;
    a_cosine_matrices[2][0][2] = 0.3;
    a_cosine_matrices[2][1][0] = 0.4;
    a_cosine_matrices[2][1][1] = 0.5;
    a_cosine_matrices[2][1][2] = 0.6;
    a_cosine_matrices[2][2][0] = 0.7;
    a_cosine_matrices[2][2][1] = 0.8;
    a_cosine_matrices[2][2][2] = 0.9;

    a_cosine_matrices[3][0][0] = 0.1;
    a_cosine_matrices[3][0][1] = 0.2;
    a_cosine_matrices[3][0][2] = 0.3;
    a_cosine_matrices[3][1][0] = 0.4;
    a_cosine_matrices[3][1][1] = 0.5;
    a_cosine_matrices[3][1][2] = 0.6;
    a_cosine_matrices[3][2][0] = 0.7;
    a_cosine_matrices[3][2][1] = 0.8;
    a_cosine_matrices[3][2][2] = 0.9;

    a_cosine_matrices[4][0][0] = 0.1;
    a_cosine_matrices[4][0][1] = 0.2;
    a_cosine_matrices[4][0][2] = 0.3;
    a_cosine_matrices[4][1][0] = 0.4;
    a_cosine_matrices[4][1][1] = 0.5;
    a_cosine_matrices[4][1][2] = 0.6;
    a_cosine_matrices[4][2][0] = 0.7;
    a_cosine_matrices[4][2][1] = 0.8;
    a_cosine_matrices[4][2][2] = 0.9;

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

    // init grad_u
    Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::A_type;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.

    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;

    derivatives = lpo_3d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             deformation_type, ref_resolved_shear_stress);
    std::cout << "test compute derivatives 11" << std::endl;
    // The correct analytical solution to check against
    double solution[5] = {0.63011275122, -0.157528187805, -0.157528187805, -0.157528187805 ,-0.157528187805};
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      REQUIRE(derivatives.first[i] == Approx(solution[i]));
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

TEST_CASE("LPO elastic tensor transform functions")
{
  dealii::SymmetricTensor<2,6> reference_elastic_tensor({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21});

// first test whether the functions are invertable
  {
    dealii::SymmetricTensor<2,6> result_up_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_up_down[i][j]));
          }
      }
  }
  {
    dealii::SymmetricTensor<2,6> result_down_up = aspect::Particle::Property::LpoElasticTensor<3>::transform_21D_vector_to_6x6_matrix(aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(reference_elastic_tensor));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_down_up[i][j]));
          }
      }
  }
  {
    dealii::SymmetricTensor<2,6> result_up_2down_up = aspect::Particle::Property::LpoElasticTensor<3>::transform_21D_vector_to_6x6_matrix(aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_21D_vector(aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor)));

    for (size_t i = 0; i < 6; i++)
      {
        for (size_t j = 0; j < 6; j++)
          {
            REQUIRE(reference_elastic_tensor[i][j] == Approx(result_up_2down_up[i][j]));
          }
      }
  }

// test rotations
  // rotation matrix
  dealii::Tensor<2,3> rotation_tensor;

  {
    // fill the rotation matrix with a rotations in all directions
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = radians;
      double beta = radians;
      double gamma = radians;
      rotation_tensor[0][0] = cos(alpha) * cos(beta);
      rotation_tensor[0][1] = sin(alpha) * cos(beta);
      rotation_tensor[0][2] = -sin(beta);
      rotation_tensor[1][0] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha)*cos(gamma);
      rotation_tensor[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha)*cos(gamma);
      rotation_tensor[1][2] = cos(beta) * sin(gamma);
      rotation_tensor[2][0] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha)*sin(gamma);
      rotation_tensor[2][1] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha)*sin(gamma);
      rotation_tensor[2][2] = cos(beta) * cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(
                                                               aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(
                                                                 aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor),rotation_tensor));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(reference_elastic_tensor,rotation_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::Tensor<4,3> result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(result_up_10_rotate, rotation_tensor);
          result_5_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(result_5_rotate, rotation_tensor);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              // This test doesn't work when rotating in all rotations at the same time.
              //REQUIRE(result_1_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }
  {
    // fill the rotation matrix with a rotations in the alpha direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = radians;
      double beta = 0;
      double gamma = 0;
      rotation_tensor[0][0] = cos(alpha) * cos(beta);
      rotation_tensor[0][1] = sin(alpha) * cos(beta);
      rotation_tensor[0][2] = -sin(beta);
      rotation_tensor[1][0] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha)*cos(gamma);
      rotation_tensor[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha)*cos(gamma);
      rotation_tensor[1][2] = cos(beta) * sin(gamma);
      rotation_tensor[2][0] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha)*sin(gamma);
      rotation_tensor[2][1] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha)*sin(gamma);
      rotation_tensor[2][2] = cos(beta) * cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(
                                                               aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(
                                                                 aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor),rotation_tensor));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(reference_elastic_tensor,rotation_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::Tensor<4,3> result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(result_up_10_rotate, rotation_tensor);
          result_5_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(result_5_rotate, rotation_tensor);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }
  {
    // fill the rotation matrix with a rotations in the beta direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = 0;
      double beta = radians;
      double gamma = 0;
      rotation_tensor[0][0] = cos(alpha) * cos(beta);
      rotation_tensor[0][1] = sin(alpha) * cos(beta);
      rotation_tensor[0][2] = -sin(beta);
      rotation_tensor[1][0] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha)*cos(gamma);
      rotation_tensor[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha)*cos(gamma);
      rotation_tensor[1][2] = cos(beta) * sin(gamma);
      rotation_tensor[2][0] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha)*sin(gamma);
      rotation_tensor[2][1] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha)*sin(gamma);
      rotation_tensor[2][2] = cos(beta) * cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(
                                                               aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(
                                                                 aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor),rotation_tensor));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(reference_elastic_tensor,rotation_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::Tensor<4,3> result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(result_up_10_rotate, rotation_tensor);
          result_5_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(result_5_rotate, rotation_tensor);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }

  {
    // fill the rotation matrix with a rotations in the gamma direction
    {
      double radians = (dealii::numbers::PI/180.0)*(360/5); //0.35*dealii::numbers::PI; //(dealii::numbers::PI/180.0)*36;
      double alpha = 0;
      double beta = 0;
      double gamma = radians;
      rotation_tensor[0][0] = cos(alpha) * cos(beta);
      rotation_tensor[0][1] = sin(alpha) * cos(beta);
      rotation_tensor[0][2] = -sin(beta);
      rotation_tensor[1][0] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha)*cos(gamma);
      rotation_tensor[1][1] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha)*cos(gamma);
      rotation_tensor[1][2] = cos(beta) * sin(gamma);
      rotation_tensor[2][0] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha)*sin(gamma);
      rotation_tensor[2][1] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha)*sin(gamma);
      rotation_tensor[2][2] = cos(beta) * cos(gamma);
    }

    {
      dealii::SymmetricTensor<2,6> result_up_1_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(
                                                               aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(
                                                                 aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(reference_elastic_tensor),rotation_tensor));
      dealii::SymmetricTensor<2,6> result_1_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(reference_elastic_tensor,rotation_tensor);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_1_rotate[i][j] == Approx(result_up_1_rotate_down[i][j]));
            }
        }

      dealii::Tensor<4,3> result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::transform_6x6_matrix_to_4th_order_tensor(result_up_1_rotate_down);

      dealii::SymmetricTensor<2,6> result_5_rotate = result_1_rotate;

      for (size_t i = 0; i < 4; i++)
        {
          result_up_10_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_4th_order_tensor(result_up_10_rotate, rotation_tensor);
          result_5_rotate = aspect::Particle::Property::LpoElasticTensor<3>::rotate_6x6_matrix(result_5_rotate, rotation_tensor);
        }

      dealii::SymmetricTensor<2,6> result_up_10_rotate_down = aspect::Particle::Property::LpoElasticTensor<3>::transform_4th_order_tensor_to_6x6_matrix(result_up_10_rotate);

      for (size_t i = 0; i < 6; i++)
        {
          for (size_t j = 0; j < 6; j++)
            {
              REQUIRE(result_5_rotate[i][j] == Approx(result_up_10_rotate_down[i][j]));
              REQUIRE(result_5_rotate[i][j] == Approx(reference_elastic_tensor[i][j]));
            }
        }
    }
  }

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


  // test store and load functions
  // the first and last element should not be changed
  // by these functions.
  std::vector<double> array_ref = {0.0,
                                   1.,2.,3.,4.,5,6,
                                   7.,8.,9.,10,11,12,
                                   13,14,15,16,17,18,
                                   19,20,21,22,23,24,
                                   25,26,27,28,29,30,
                                   31,32,33,34,35,36,
                                   37
                                  };

  std::vector<double> array = {0.0,
                               1.,2.,3.,4.,5.,6.,
                               7.,8.,9.,10,11,12,
                               13,14,15,16,17,18,
                               19,20,21,22,23,24,
                               25,26,27,28,29,30,
                               31,32,33,34,35,36,
                               37
                              };

  // There used be be 36 unique entries, but now because we are using the
  // symmetric tensor, there are only 21 unique entries.
  std::vector<double> array_plus_100 = {0.0,
                                        101.,102.,103.,104.,105.,106.,
                                        107.,108.,109.,110.,111.,112.,
                                        113.,114.,115.,116.,117.,118.,
                                        119.,120.,121.,22.,23.,24.,
                                        25.,26.,27.,28.,29.,30.,
                                        31.,32.,33.,34.,35.,36.,
                                        37.
                                       };

  unsigned int lpo_data_position = 1;
  dealii::ArrayView<double> data(&array[0],38);
  dealii::SymmetricTensor<2,6> tensor = dealii::SymmetricTensor<2,6>();
  lpo_elastic_tensor.load_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    CHECK(data[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);

  lpo_elastic_tensor.store_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < array.size() ; ++i)
    CHECK(data[i] == array_ref[i]);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)] += 100;


  lpo_elastic_tensor.store_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < array.size() ; ++i)
    CHECK(data[i] == array_plus_100[i]);

  lpo_elastic_tensor.load_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    CHECK(data[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    CHECK(array_plus_100[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);

  for (unsigned int i = 0; i < array.size() ; ++i)
    REQUIRE(data[i] == array_plus_100[i]);
}
