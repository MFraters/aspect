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

#include <aspect/particle/property/decompose_elastic_matrix.h>
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

  CHECK(lpo.determine_deformation_type(379.*MPa, 0.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(381.*MPa, 0.) == DeformationType::OlivineDFabric);
  CHECK(lpo.determine_deformation_type(0.*MPa, 100.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 50.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 50.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(379.*MPa, 50.) == DeformationType::OlivineDFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 49.) == DeformationType::OlivineDFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 75.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 100.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 100.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(379.*MPa, 100.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 100.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 200.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 200.) == DeformationType::OlivineAFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 200.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 200.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 200.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 200.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 300.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 300.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 300.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 300.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 300.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 300.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 380.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 380.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 380.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 380.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 380.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 380.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 380.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 400.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 400.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 400.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 400.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 400.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 400.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 400.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 600.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 600.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 600.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 600.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 600.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 600.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 600.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 800.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 800.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 800.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 800.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 800.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 800.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 800.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 1000.) == DeformationType::OlivineEFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 1000.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 1000.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 1000.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 1000.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 1000.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 1000.) == DeformationType::OlivineBFabric);

  CHECK(lpo.determine_deformation_type(20.*MPa, 1200.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(100.*MPa, 1200.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(200.*MPa, 1200.) == DeformationType::OlivineCFabric);
  CHECK(lpo.determine_deformation_type(340.*MPa, 1200.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(360.*MPa, 1200.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(380.*MPa, 1200.) == DeformationType::OlivineBFabric);
  CHECK(lpo.determine_deformation_type(480.*MPa, 1200.) == DeformationType::OlivineBFabric);
}

TEST_CASE("Euler angle functions")
{
  using namespace aspect;
  {
    // tranforming rotation matrices to euler angles and back
    Postprocess::LPO<3> lpo;
    dealii::Tensor<2,3> rot1;
    rot1[0][0] = 0.36;
    rot1[0][1] = 0.48;
    rot1[0][2] = -0.8;

    rot1[1][0] = -0.8;
    rot1[1][1] = 0.6;
    rot1[1][2] = 0;

    rot1[2][0] = 0.48;
    rot1[2][1] = 0.64;
    rot1[2][2] = 0.6;

    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot1);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot1);
    auto ea2 = lpo.euler_angles_from_rotation_matrix(rot2);
    compare_3d_arrays_approx(ea2,ea1);
    auto rot3 = lpo.euler_angles_to_rotation_matrix(ea2[0],ea2[1],ea2[2]);
    compare_rotation_matrices_approx(rot3, rot2);
  }

  {
    // tranforming rotation matrices to euler angles may be nonunique as this
    // example shows. The input rot1 will be different from the output rot2, but
    // rot2 is the same as rot3. This is a case of a rotation matrix which is
    // incompatible with euler angles since [0][0] is nonzero, but to be consistent
    // with ZXZ euler angles the bottom row can't be zero. TODO: Maybe find a way to
    // detect and assert this?
    Postprocess::LPO<3> lpo;
    dealii::Tensor<2,3> rot1;
    rot1[0][0] = 0.36;
    rot1[0][1] = 0.48;
    rot1[0][2] = -0.8;

    rot1[1][0] = -0.8;
    rot1[1][1] = 0.6;
    rot1[1][2] = 0;

    rot1[2][0] = 0.0;
    rot1[2][1] = 0.0;
    rot1[2][2] = 0.0;

    dealii::Tensor<2,3> rot2_expected;
    rot2_expected[0][0] = 0;
    rot2_expected[0][1] = 0;
    rot2_expected[0][2] = -1;

    rot2_expected[1][0] = -1;
    rot2_expected[1][1] = 0.0;
    rot2_expected[1][2] = 0.0;

    rot2_expected[2][0] = 0.0;
    rot2_expected[2][1] = 1.0;
    rot2_expected[2][2] = 0.0;

    auto ea1 = lpo.euler_angles_from_rotation_matrix(rot1);
    auto rot2 = lpo.euler_angles_to_rotation_matrix(ea1[0],ea1[1],ea1[2]);
    compare_rotation_matrices_approx(rot2, rot2_expected);
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

  {
    // first test initialization 2d.
    const int dim2=2;

    Particle::Property::LPO<dim2> lpo_2d;
    ParameterHandler prm;
    lpo_2d.declare_parameters(prm);
    prm.declare_entry("World builder file", "", dealii::Patterns::Anything(), "");
    prm.set("World builder file", "/home/fraters/Documents/post-doc/2019-02-01-magali/code/aspect/aspect-C/build-dg-u-nj/advection_test.wb");
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

    lpo_2d.parse_parameters(prm);
    lpo_2d.initialize();

    Point<dim2> dummy_point;
    std::vector<double> data;
    lpo_2d.initialize_one_particle_property(dummy_point, data);

    // The LPO particles are initialized. With the same seed, the outcome should
    // always be the same, so test that for seed = 1. Forthermore, in the data
    // I can only really test that the first entry is the water content (0) and
    // that every first entry of each particle is 1/n_grains = 1/10 = 0.1.
    CHECK(data[0] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
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
    CHECK(data[52] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
    CHECK(data[53] == Approx(0.5)); // default volume fraction olivine
    CHECK(data[54] == Approx(0.2));
    CHECK(data[64] == Approx(0.2));
    CHECK(data[74] == Approx(0.2));
    CHECK(data[84] == Approx(0.2));
    CHECK(data[94] == Approx(0.2));
    CHECK(data[103] == Approx(-0.816855));

    std::vector<double> volume_fractions(5,0.2);
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
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

    SymmetricTensor<2,3> strain_rate_nondimensional;
    strain_rate_nondimensional[0][1] = 0.5959;

    Tensor<2,3> velocity_gradient_tensor_nondimensional;
    velocity_gradient_tensor_nondimensional[0][1] = 2.0* 0.5959;
    velocity_gradient_tensor_nondimensional[1][0] = 2.0* 0.5959;

    //Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::OlivineAFabric;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.

    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;
    derivatives = lpo_2d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             0.5, ref_resolved_shear_stress);

    // The correct analytical solution to check against
    // Note that this still has to be multiplied with with volume fraction
    // of each graint to get the same solution as D-Rex would get.
    double solution[5] = {3.150563756, -0.787640939, -0.787640939, -0.787640939 ,-0.787640939};
    for (unsigned int iii = 0; iii < derivatives.first.size(); ++iii)
      CHECK(derivatives.first[iii] == Approx(solution[iii]));

    Tensor<2,3> deriv_direction_solution_2_to_5;
    deriv_direction_solution_2_to_5[0][0] = 0;
    deriv_direction_solution_2_to_5[0][1] = 0.00501912;
    deriv_direction_solution_2_to_5[0][2] = 0.0100382;
    deriv_direction_solution_2_to_5[1][0] = -0.00501912;
    deriv_direction_solution_2_to_5[1][1] = 0;
    deriv_direction_solution_2_to_5[1][2] = 0.00501912;
    deriv_direction_solution_2_to_5[2][0] = -0.0100382;
    deriv_direction_solution_2_to_5[2][1] = -0.00501912;
    deriv_direction_solution_2_to_5[2][2] = 0;
    Tensor<2,3> deriv_direction_solution[5] = {Tensor<2,3>(),deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(deriv_direction_solution[index][iii][jjj] == Approx(derivatives.second[index][iii][jjj]));
              }
          }
      }
    // now check if the value is the same as the D-Rex output when contracted with the direction matrix
    Tensor<2,3> deriv_direction_full_solution_2_to_5;
    deriv_direction_full_solution_2_to_5[0][0] = -0.004015284 ;
    deriv_direction_full_solution_2_to_5[0][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[0][2] = 0.0020076485;
    deriv_direction_full_solution_2_to_5[1][0] = -0.00853248;
    deriv_direction_full_solution_2_to_5[1][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[1][2] = 0.0065248576;
    deriv_direction_full_solution_2_to_5[2][0] = -0.0130497151;
    deriv_direction_full_solution_2_to_5[2][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[2][2] = 0.0110420667;
    Tensor<2,3> deriv_direction_full_solution[5] = {Tensor<2,3>(),deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        auto full_solution = a_cosine_matrices[index] * deriv_direction_solution[index];
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(full_solution[iii][jjj] == Approx(deriv_direction_full_solution[index][iii][jjj]));
              }
          }
      }
  }

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
    CHECK(data[0] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
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
    CHECK(data[52] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
    CHECK(data[53] == Approx(0.5)); // default volume fraction olivine
    CHECK(data[54] == Approx(0.2));
    CHECK(data[64] == Approx(0.2));
    CHECK(data[74] == Approx(0.2));
    CHECK(data[84] == Approx(0.2));
    CHECK(data[94] == Approx(0.2));
    CHECK(data[103] == Approx(-0.816855));

    std::vector<double> volume_fractions(5,0.2);
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
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

    SymmetricTensor<2,3> strain_rate_nondimensional;
    strain_rate_nondimensional[0][1] = 0.5959;

    Tensor<2,3> velocity_gradient_tensor_nondimensional;
    velocity_gradient_tensor_nondimensional[0][1] = 2.0* 0.5959;
    velocity_gradient_tensor_nondimensional[1][0] = 2.0* 0.5959;

    //Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::OlivineAFabric;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.

    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;

    derivatives = lpo_3d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             0.5, ref_resolved_shear_stress);

    // The correct analytical solution to check against
    double solution[5] = {3.150563756, -0.787640939, -0.787640939, -0.787640939 ,-0.787640939};
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      CHECK(derivatives.first[i] == Approx(solution[i]));


    Tensor<2,3> deriv_direction_solution_2_to_5;
    deriv_direction_solution_2_to_5[0][0] = 0;
    deriv_direction_solution_2_to_5[0][1] = 0.00501912;
    deriv_direction_solution_2_to_5[0][2] = 0.0100382;
    deriv_direction_solution_2_to_5[1][0] = -0.00501912;
    deriv_direction_solution_2_to_5[1][1] = 0;
    deriv_direction_solution_2_to_5[1][2] = 0.00501912;
    deriv_direction_solution_2_to_5[2][0] = -0.0100382;
    deriv_direction_solution_2_to_5[2][1] = -0.00501912;
    deriv_direction_solution_2_to_5[2][2] = 0;

    Tensor<2,3> deriv_direction_solution[5] = {Tensor<2,3>(),deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(deriv_direction_solution[index][iii][jjj] == Approx(derivatives.second[index][iii][jjj]));
              }
          }
      }
    // now check if the value is the same as the D-Rex output when contracted with the direction matrix
    Tensor<2,3> deriv_direction_full_solution_2_to_5;
    deriv_direction_full_solution_2_to_5[0][0] = -0.004015284 ;
    deriv_direction_full_solution_2_to_5[0][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[0][2] = 0.0020076485;
    deriv_direction_full_solution_2_to_5[1][0] = -0.00853248;
    deriv_direction_full_solution_2_to_5[1][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[1][2] = 0.0065248576;
    deriv_direction_full_solution_2_to_5[2][0] = -0.0130497151;
    deriv_direction_full_solution_2_to_5[2][1] = -0.0010038242;
    deriv_direction_full_solution_2_to_5[2][2] = 0.0110420667;
    Tensor<2,3> deriv_direction_full_solution[5] = {Tensor<2,3>(),deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        auto full_solution = a_cosine_matrices[index] * deriv_direction_solution[index];
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(full_solution[iii][jjj] == Approx(deriv_direction_full_solution[index][iii][jjj]));
              }
          }
      }
  }

  {
    // thirdly test 3d lpo with different strain-rate and velocity gradient tensors.
    // The solution of this test has not been analytically confirmed.
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
    CHECK(data[0] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
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
    CHECK(data[52] == Approx(-1.0));//isnan(data[0])); // default fabric type which is only computed on a update
    CHECK(data[53] == Approx(0.5)); // default volume fraction olivine
    CHECK(data[54] == Approx(0.2));
    CHECK(data[64] == Approx(0.2));
    CHECK(data[74] == Approx(0.2));
    CHECK(data[84] == Approx(0.2));
    CHECK(data[94] == Approx(0.2));
    CHECK(data[103] == Approx(-0.816855));

    std::vector<double> volume_fractions(5,0.2);
    std::vector<dealii::Tensor<2,3> > a_cosine_matrices(5);
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

    SymmetricTensor<2,dim3> strain_rate_nondimensional; // e
    strain_rate_nondimensional[0][0] = 7.5;
    strain_rate_nondimensional[0][1] = 8;
    strain_rate_nondimensional[0][2] = 8.5;
    //strain_rate_nondimensional[1][0] = 8; // symmetry
    strain_rate_nondimensional[1][1] = 9.5;
    strain_rate_nondimensional[1][2] = 10;
    //strain_rate_nondimensional[2][0] = 8.5;// symmetry
    //strain_rate_nondimensional[2][1] = 10; // symmetry
    strain_rate_nondimensional[2][2] = 11.5;

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

    //Particle::Property::DeformationType deformation_type = Particle::Property::DeformationType::OlivineAFabric;
    std::array<double,4> ref_resolved_shear_stress;
    ref_resolved_shear_stress[0] = 1;
    ref_resolved_shear_stress[1] = 2;
    ref_resolved_shear_stress[2] = 3;
    ref_resolved_shear_stress[3] = 1e60; // can't really use nummerical limits max or infinite, because need to be able to square it without becomming infinite. This is the value fortran D-Rex uses.

    std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;

    derivatives = lpo_3d.compute_derivatives(volume_fractions, a_cosine_matrices,
                                             strain_rate_nondimensional, velocity_gradient_tensor_nondimensional,
                                             0.5, ref_resolved_shear_stress);

    // The correct analytical solution to check against
    double solution[5] = {2.5350823696, -0.6337705924, -0.6337705924, -0.6337705924 ,-0.6337705924};
    for (unsigned int i = 0; i < derivatives.first.size(); ++i)
      CHECK(derivatives.first[i] == Approx(solution[i]));


    Tensor<2,3> deriv_direction_solution_1;
    deriv_direction_solution_1[0][0] = 0;
    deriv_direction_solution_1[0][1] = 0.5;
    deriv_direction_solution_1[0][2] = 1.0;
    deriv_direction_solution_1[1][0] = -0.5;
    deriv_direction_solution_1[1][1] = 0;
    deriv_direction_solution_1[1][2] = 0.5;
    deriv_direction_solution_1[2][0] = -1.0;
    deriv_direction_solution_1[2][1] = -0.5;
    deriv_direction_solution_1[2][2] = 0;
    Tensor<2,3> deriv_direction_solution_2_to_5;
    deriv_direction_solution_2_to_5[0][0] = 0;
    deriv_direction_solution_2_to_5[0][1] = 0.5304624591;
    deriv_direction_solution_2_to_5[0][2] = 1.0609249182;
    deriv_direction_solution_2_to_5[1][0] = -0.5304624591;
    deriv_direction_solution_2_to_5[1][1] = 0;
    deriv_direction_solution_2_to_5[1][2] = 0.5304624591;
    deriv_direction_solution_2_to_5[2][0] = -1.0609249182;
    deriv_direction_solution_2_to_5[2][1] = -0.5304624591 ;
    deriv_direction_solution_2_to_5[2][2] = 0;

    Tensor<2,3> deriv_direction_solution[5] = {deriv_direction_solution_1,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5,deriv_direction_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(deriv_direction_solution[index][iii][jjj] == Approx(derivatives.second[index][iii][jjj]));
              }
          }
      }
    // now check if the value is the same as the D-Rex output when contracted with the direction matrix
    Tensor<2,3> deriv_direction_full_solution_1;
    deriv_direction_full_solution_1[0][0] = -0.75;
    deriv_direction_full_solution_1[0][1] = 0.0;
    deriv_direction_full_solution_1[0][2] = 0.75;
    deriv_direction_full_solution_1[1][0] = -0.75;
    deriv_direction_full_solution_1[1][1] = 0;
    deriv_direction_full_solution_1[1][2] = 0.75;
    deriv_direction_full_solution_1[2][0] = -0.75;
    deriv_direction_full_solution_1[2][1] = 0.0;
    deriv_direction_full_solution_1[2][2] = 0.75;
    Tensor<2,3> deriv_direction_full_solution_2_to_5;
    deriv_direction_full_solution_2_to_5[0][0] = -0.4243699673 ;
    deriv_direction_full_solution_2_to_5[0][1] = -0.1060924918;
    deriv_direction_full_solution_2_to_5[0][2] = 0.2121849836;
    deriv_direction_full_solution_2_to_5[1][0] = -0.9017861805;
    deriv_direction_full_solution_2_to_5[1][1] = -0.1060924918;
    deriv_direction_full_solution_2_to_5[1][2] = 0.6896011968;
    deriv_direction_full_solution_2_to_5[2][0] = -1.3792023937;
    deriv_direction_full_solution_2_to_5[2][1] = -0.1060924918;
    deriv_direction_full_solution_2_to_5[2][2] = 1.16701741;
    Tensor<2,3> deriv_direction_full_solution[5] = {deriv_direction_full_solution_1,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5,deriv_direction_full_solution_2_to_5};
    for (size_t index = 0; index < 5; index++)
      {
        auto full_solution = a_cosine_matrices[index] * deriv_direction_solution[index];
        for (size_t iii = 0; iii < 3; iii++)
          {
            for (size_t jjj = 0; jjj < 3; jjj++)
              {
                CHECK(full_solution[iii][jjj] == Approx(deriv_direction_full_solution[index][iii][jjj]));
              }
          }
      }
  }
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

  std::vector<double> volume_fraction_mineral = {0.7,0.3};
  std::vector<std::vector<double> > volume_fractions_grains(2,std::vector<double>(8));
  std::vector<std::vector<dealii::Tensor<2,3> > > a_cosine_matrices_grains(2,std::vector<dealii::Tensor<2,3> >(8));

  dealii::Tensor<2,6> reference_elastic_tensor;
  dealii::Tensor<2,6> computed_elastic_tensor;

  // All these numbers are directly from the Fortran D-Rex
  // Had to fix the random seed to get consistent awnsers.
  // Fixed the random set to an array filled with zeros.
  a_cosine_matrices_grains[0][0][0][0] = -0.87492387659370430;
  a_cosine_matrices_grains[0][0][0][1] = -0.47600252255715020;
  a_cosine_matrices_grains[0][0][0][2] = -0.10151800968122601;
  a_cosine_matrices_grains[0][0][1][0] = 0.13036031917262200;
  a_cosine_matrices_grains[0][0][1][1] = -2.6406769698713056E-002;
  a_cosine_matrices_grains[0][0][1][2] = -0.99232315682823224;
  a_cosine_matrices_grains[0][0][2][0] = 0.46957683898444408;
  a_cosine_matrices_grains[0][0][2][1] = -0.87974004016081919;
  a_cosine_matrices_grains[0][0][2][2] = 8.6098835151007691E-002;
  a_cosine_matrices_grains[0][1][0][0] = -0.98046837857873570;
  a_cosine_matrices_grains[0][1][0][1] = 0.19463893778994429;
  a_cosine_matrices_grains[0][1][0][2] = -2.8239743415760400E-002;
  a_cosine_matrices_grains[0][1][1][0] = 6.8942963409018593E-002;
  a_cosine_matrices_grains[0][1][1][1] = 0.20565757913350766;
  a_cosine_matrices_grains[0][1][1][2] = -0.97619251975954824;
  a_cosine_matrices_grains[0][1][2][0] = -0.18419735979776022;
  a_cosine_matrices_grains[0][1][2][1] = -0.95907282258851756;
  a_cosine_matrices_grains[0][1][2][2] = -0.21505972600848655;
  a_cosine_matrices_grains[0][2][0][0] = -0.60475851993637786;
  a_cosine_matrices_grains[0][2][0][1] = 0.70843624030907060;
  a_cosine_matrices_grains[0][2][0][2] = -0.36543256811748415;
  a_cosine_matrices_grains[0][2][1][0] = 0.14301138974031016;
  a_cosine_matrices_grains[0][2][1][1] = -0.35406726088673490;
  a_cosine_matrices_grains[0][2][1][2] = -0.92567249145068109;
  a_cosine_matrices_grains[0][2][2][0] = -0.78533679283378455;
  a_cosine_matrices_grains[0][2][2][1] = -0.61106385979914768;
  a_cosine_matrices_grains[0][2][2][2] = 0.11279851062549377;
  a_cosine_matrices_grains[0][3][0][0] = 0.86791495614143876;
  a_cosine_matrices_grains[0][3][0][1] = 0.11797028562530290;
  a_cosine_matrices_grains[0][3][0][2] = -0.48441508819120616;
  a_cosine_matrices_grains[0][3][1][0] = 0.29623368357270952;
  a_cosine_matrices_grains[0][3][1][1] = 0.66075310029034506;
  a_cosine_matrices_grains[0][3][1][2] = 0.69114249890201696;
  a_cosine_matrices_grains[0][3][2][0] = 0.40054408016972737;
  a_cosine_matrices_grains[0][3][2][1] = -0.74231595966649988;
  a_cosine_matrices_grains[0][3][2][2] = 0.53869116329275069;
  a_cosine_matrices_grains[0][4][0][0] = 6.2142369991145308E-002;
  a_cosine_matrices_grains[0][4][0][1] = 0.99798524450925208;
  a_cosine_matrices_grains[0][4][0][2] = 1.4733749724082583E-002;
  a_cosine_matrices_grains[0][4][1][0] = 0.16313304505497195;
  a_cosine_matrices_grains[0][4][1][1] = 3.9946223138032123E-003;
  a_cosine_matrices_grains[0][4][1][2] = -0.99141604427573704;
  a_cosine_matrices_grains[0][4][2][0] = -0.98947772660983668;
  a_cosine_matrices_grains[0][4][2][1] = 6.3633828841832010E-002;
  a_cosine_matrices_grains[0][4][2][2] = -0.16253511002663851;
  a_cosine_matrices_grains[0][5][0][0] = 0.95811162445076037;
  a_cosine_matrices_grains[0][5][0][1] = -0.24047036409864109;
  a_cosine_matrices_grains[0][5][0][2] = -0.18394917461469307;
  a_cosine_matrices_grains[0][5][1][0] = -0.26945849516984494;
  a_cosine_matrices_grains[0][5][1][1] = -0.95216866624585605;
  a_cosine_matrices_grains[0][5][1][2] = -0.14816174866727450;
  a_cosine_matrices_grains[0][5][2][0] = -0.13948857525086517;
  a_cosine_matrices_grains[0][5][2][1] = 0.18918653693665904;
  a_cosine_matrices_grains[0][5][2][2] = -0.97685775746677384;
  a_cosine_matrices_grains[0][6][0][0] = 0.54599545795872684;
  a_cosine_matrices_grains[0][6][0][1] = -0.79260950430410815;
  a_cosine_matrices_grains[0][6][0][2] = -0.27534584625644454;
  a_cosine_matrices_grains[0][6][1][0] = -0.83567561116582212;
  a_cosine_matrices_grains[0][6][1][1] = -0.55085590166106368;
  a_cosine_matrices_grains[0][6][1][2] = -6.8746495709015629E-002;
  a_cosine_matrices_grains[0][6][2][0] = -9.6081601263635075E-002;
  a_cosine_matrices_grains[0][6][2][1] = 0.26479508638287669;
  a_cosine_matrices_grains[0][6][2][2] = -0.96221681521835400;
  a_cosine_matrices_grains[0][7][0][0] = 0.14540407652501869;
  a_cosine_matrices_grains[0][7][0][1] = -0.61649901222701298;
  a_cosine_matrices_grains[0][7][0][2] = -0.77881206212059828;
  a_cosine_matrices_grains[0][7][1][0] = -0.59216315099851768;
  a_cosine_matrices_grains[0][7][1][1] = -0.68282211925029168;
  a_cosine_matrices_grains[0][7][1][2] = 0.43341920614609419;
  a_cosine_matrices_grains[0][7][2][0] = -0.79819736735552915;
  a_cosine_matrices_grains[0][7][2][1] = 0.39350172081601731;
  a_cosine_matrices_grains[0][7][2][2] = -0.46322016633770996;

  volume_fractions_grains[0][0] = 2.5128593570287589E-002;
  volume_fractions_grains[0][1] = 0.83128842847575013;
  volume_fractions_grains[0][2] = 2.4387041141724769E-002;
  volume_fractions_grains[0][3] = 2.4763275182773107E-002;
  volume_fractions_grains[0][4] = 2.4801714431754770E-002;
  volume_fractions_grains[0][5] = 2.3943562805875843E-002;
  volume_fractions_grains[0][6] = 2.1493810045792379E-002;
  volume_fractions_grains[0][7] = 2.4193574346041427E-002;

  a_cosine_matrices_grains[1][0][0][0] = -0.66168933252008499;
  a_cosine_matrices_grains[1][0][0][1] = -0.27722421136423192;
  a_cosine_matrices_grains[1][0][0][2] = 0.70016104334335305;
  a_cosine_matrices_grains[1][0][1][0] = -0.45052346117292291;
  a_cosine_matrices_grains[1][0][1][1] = -0.59946225647123619;
  a_cosine_matrices_grains[1][0][1][2] = -0.66370395454608444;
  a_cosine_matrices_grains[1][0][2][0] = 0.60350910238307343;
  a_cosine_matrices_grains[1][0][2][1] = -0.75116099269655345;
  a_cosine_matrices_grains[1][0][2][2] = 0.27207473610935284;
  a_cosine_matrices_grains[1][1][0][0] = 0.70309122563849258;
  a_cosine_matrices_grains[1][1][0][1] = -0.23393734574397834;
  a_cosine_matrices_grains[1][1][0][2] = -0.67775637808568567;
  a_cosine_matrices_grains[1][1][1][0] = 0.68298185906364617;
  a_cosine_matrices_grains[1][1][1][1] = -6.6406501211459870E-002;
  a_cosine_matrices_grains[1][1][1][2] = 0.73168002139436472;
  a_cosine_matrices_grains[1][1][2][0] = -0.21654097292603913;
  a_cosine_matrices_grains[1][1][2][1] = -0.97001835897279087;
  a_cosine_matrices_grains[1][1][2][2] = 0.11341644508992850;
  a_cosine_matrices_grains[1][2][0][0] = -0.37892641930874921;
  a_cosine_matrices_grains[1][2][0][1] = 0.92610670487847191;
  a_cosine_matrices_grains[1][2][0][2] = 3.9338024409460159E-002;
  a_cosine_matrices_grains[1][2][1][0] = -0.10014673867324062;
  a_cosine_matrices_grains[1][2][1][1] = 2.1922389732184896E-004;
  a_cosine_matrices_grains[1][2][1][2] = -0.99514251026853373;
  a_cosine_matrices_grains[1][2][2][0] = -0.92466780075091537;
  a_cosine_matrices_grains[1][2][2][1] = -0.37833500123197994;
  a_cosine_matrices_grains[1][2][2][2] = 9.2539397053637271E-002;
  a_cosine_matrices_grains[1][3][0][0] = 0.73973803569795438;
  a_cosine_matrices_grains[1][3][0][1] = -0.58084801380447959;
  a_cosine_matrices_grains[1][3][0][2] = -0.35134329241205425;
  a_cosine_matrices_grains[1][3][1][0] = 0.30883050284698427;
  a_cosine_matrices_grains[1][3][1][1] = -0.17349072757172229;
  a_cosine_matrices_grains[1][3][1][2] = 0.94047567596335968;
  a_cosine_matrices_grains[1][3][2][0] = -0.60686583881969203;
  a_cosine_matrices_grains[1][3][2][1] = -0.79564553570809793;
  a_cosine_matrices_grains[1][3][2][2] = 5.0387432541084881E-002;
  a_cosine_matrices_grains[1][4][0][0] = 4.2730545437086563E-002;
  a_cosine_matrices_grains[1][4][0][1] = 0.99790446393350407;
  a_cosine_matrices_grains[1][4][0][2] = 4.9677348699965519E-002;
  a_cosine_matrices_grains[1][4][1][0] = 0.24607319829629554;
  a_cosine_matrices_grains[1][4][1][1] = 3.7594987859276820E-002;
  a_cosine_matrices_grains[1][4][1][2] = -0.97243353311600111;
  a_cosine_matrices_grains[1][4][2][0] = -0.97326576151708344;
  a_cosine_matrices_grains[1][4][2][1] = 5.2990062123162360E-002;
  a_cosine_matrices_grains[1][4][2][2] = -0.24420040893728351;
  a_cosine_matrices_grains[1][5][0][0] = -0.92112012021624434;
  a_cosine_matrices_grains[1][5][0][1] = 0.17590913101287936;
  a_cosine_matrices_grains[1][5][0][2] = 0.34726602737040008;
  a_cosine_matrices_grains[1][5][1][0] = -0.27292179963530816;
  a_cosine_matrices_grains[1][5][1][1] = -0.92793421776529450;
  a_cosine_matrices_grains[1][5][1][2] = -0.25387354780599874;
  a_cosine_matrices_grains[1][5][2][0] = 0.27758135269283285;
  a_cosine_matrices_grains[1][5][2][1] = -0.32862450412044819;
  a_cosine_matrices_grains[1][5][2][2] = 0.90274831467569783;
  a_cosine_matrices_grains[1][6][0][0] = -4.6028553384029648E-002;
  a_cosine_matrices_grains[1][6][0][1] = -0.82828827663204008;
  a_cosine_matrices_grains[1][6][0][2] = -0.56150509327989218;
  a_cosine_matrices_grains[1][6][1][0] = -0.62780239844730912;
  a_cosine_matrices_grains[1][6][1][1] = -0.41264414122407550;
  a_cosine_matrices_grains[1][6][1][2] = 0.66416683570654200;
  a_cosine_matrices_grains[1][6][2][0] = -0.78150657001547463;
  a_cosine_matrices_grains[1][6][2][1] = 0.37975977805647526;
  a_cosine_matrices_grains[1][6][2][2] = -0.50060220610535922;
  a_cosine_matrices_grains[1][7][0][0] = -0.69974343277254114;
  a_cosine_matrices_grains[1][7][0][1] = -0.60514581197791628;
  a_cosine_matrices_grains[1][7][0][2] = 0.38074455421574593;
  a_cosine_matrices_grains[1][7][1][0] = 0.27181144918830025;
  a_cosine_matrices_grains[1][7][1][1] = -0.71756260285634454;
  a_cosine_matrices_grains[1][7][1][2] = -0.64160793166977359;
  a_cosine_matrices_grains[1][7][2][0] = 0.66130789657394939;
  a_cosine_matrices_grains[1][7][2][1] = -0.34496383827187421;
  a_cosine_matrices_grains[1][7][2][2] = 0.66656470768691123;

  volume_fractions_grains[1][0] = 2.4802805652404735E-002;
  volume_fractions_grains[1][1] = 2.4917064186342413E-002;
  volume_fractions_grains[1][2] = 2.4790722064907112E-002;
  volume_fractions_grains[1][3] = 2.4932751194567736E-002;
  volume_fractions_grains[1][4] = 2.4896744967217745E-002;
  volume_fractions_grains[1][5] = 0.79902139535472505;
  volume_fractions_grains[1][6] = 2.4861121542485400E-002;
  volume_fractions_grains[1][7] = 5.1777395037349808E-002;

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

  std::vector<unsigned int> deformation_types = {(unsigned int)aspect::Particle::Property::DeformationType::OlivineAFabric,
                                                 (unsigned int)aspect::Particle::Property::DeformationType::Enstatite
                                                };

  aspect::Particle::Property::LpoElasticTensor<3> lpo_elastic_tensor;
  computed_elastic_tensor = lpo_elastic_tensor.compute_elastic_tensor(volume_fraction_mineral,
                                                                      volume_fractions_grains,
                                                                      a_cosine_matrices_grains,
                                                                      deformation_types);


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
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      CHECK(data[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
      {
        if (i < 6)
          {
            CHECK(tensor[ {i,i}] == data[lpo_data_position + i]);
          }
        else
          {
            [&]
            {
              for (unsigned int d = 0, c = 6; d < 6; ++d)
                {
                  for (unsigned int e = d + 1; e < 6; ++e, ++c)
                    {
                      if (c == i)
                        {
                          CHECK(tensor[ {d,e}] == data[lpo_data_position + i]);
                          return;
                        }
                    }
                }
            }();
          }
      }
#endif
    }

  lpo_elastic_tensor.store_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < array.size() ; ++i)
    CHECK(data[i] == array_ref[i]);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)] += 100;
#else
      {
        if (i < 6)
          {
            tensor[ {i,i}] += 100;
          }
        else
          {
            [&]
            {
              for (unsigned int d = 0, c = 6; d < 6; ++d)
                {
                  for (unsigned int e = d + 1; e < 6; ++e, ++c)
                    {
                      if (c == i)
                        {
                          tensor[ {d,e}] += 100;
                          return;
                        }
                    }
                }
            }();
          }
      }
#endif
    }


  lpo_elastic_tensor.store_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < array.size() ; ++i)
    CHECK(data[i] == array_plus_100[i]);

  lpo_elastic_tensor.load_particle_data(lpo_data_position,data,tensor);

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      CHECK(data[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
      {
        if (i < 6)
          {
            CHECK(tensor[ {i,i}] == data[lpo_data_position + i]);
          }
        else
          {
            [&]
            {
              for (unsigned int d = 0, c = 6; d < 6; ++d)
                {
                  for (unsigned int e = d + 1; e < 6; ++e, ++c)
                    {
                      if (c == i)
                        {
                          CHECK(tensor[ {d,e}] == data[lpo_data_position + i]);
                          return;
                        }
                    }
                }
            }();
          }
      }
#endif
    }

  for (unsigned int i = 0; i < dealii::SymmetricTensor<2,6>::n_independent_components ; ++i)
    {
#if DEAL_II_VERSION_GTE(9,3,0)
      CHECK(array_plus_100[lpo_data_position + i] == tensor[dealii::SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
      {
        if (i < 6)
          {
            CHECK(tensor[ {i,i}] == array_plus_100[lpo_data_position + i]);
          }
        else
          {
            [&]
            {
              for (unsigned int d = 0, c = 6; d < 6; ++d)
                {
                  for (unsigned int e = d + 1; e < 6; ++e, ++c)
                    {
                      if (c == i)
                        {
                          CHECK(tensor[ {d,e}] == array_plus_100[lpo_data_position + i]);
                          return;
                        }
                    }
                }
            }();
          }
      }
#endif
    }

  for (unsigned int i = 0; i < array.size() ; ++i)
    REQUIRE(data[i] == array_plus_100[i]);
}


TEST_CASE("LPO elastic tensor decomposition")
{
  using namespace dealii;
  using namespace aspect::Particle::Property;
  {
    auto elastic_composition_class = DecomposeElasticMatrix<3>();

    // the matrix from Browaeys and Chevrot, 2004, Geophys. J. Int. (doi: 10.1111/j.1365-246X.2004.02415.x)
    // Note that the computed norms are not in the papers, but are an extra check.
    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 192;
    full_elastic_matrix[0][1] = 66;
    full_elastic_matrix[0][2] = 60;
    full_elastic_matrix[1][1] = 160;
    full_elastic_matrix[1][2] = 56;
    full_elastic_matrix[2][2] = 272;
    full_elastic_matrix[3][3] = 60;
    full_elastic_matrix[4][4] = 62;
    full_elastic_matrix[5][5] = 49;

    SymmetricTensor<2,6> reference_isotropic_matrix;
    reference_isotropic_matrix[0][0] = 194.7;
    reference_isotropic_matrix[0][1] = 67.3;
    reference_isotropic_matrix[0][2] = 67.3;
    reference_isotropic_matrix[1][1] = 194.7;
    reference_isotropic_matrix[1][2] = 67.3;
    reference_isotropic_matrix[2][2] = 194.7;
    reference_isotropic_matrix[3][3] = 63.7;
    reference_isotropic_matrix[4][4] = 63.7;
    reference_isotropic_matrix[5][5] = 63.7;

    SymmetricTensor<2,6> reference_hex_matrix;
    reference_hex_matrix[0][0] = -21.7;
    reference_hex_matrix[1][1] = -21.7;
    reference_hex_matrix[2][2] = 77.3;
    reference_hex_matrix[1][2] = -9.3;
    reference_hex_matrix[0][2] = -9.3;
    reference_hex_matrix[0][1] = 1.7;
    reference_hex_matrix[3][3] = -2.7;
    reference_hex_matrix[4][4] = -2.7;
    reference_hex_matrix[5][5] = -11.7;

    SymmetricTensor<2,6> reference_T_matrix;
    reference_T_matrix[0][0] = 3;
    reference_T_matrix[1][1] = 3;
    reference_T_matrix[0][1] = -3;
    reference_T_matrix[5][5] = -3;

    SymmetricTensor<2,6> reference_O_matrix;
    reference_O_matrix[0][0] = 16;
    reference_O_matrix[1][1] = -16;
    reference_O_matrix[1][2] = -2;
    reference_O_matrix[0][2] = 2;
    reference_O_matrix[3][3] = -1;
    reference_O_matrix[4][4] = 1;

    const Tensor<1,21> full_elastic_vector = LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(full_elastic_matrix);
    const Tensor<1,21> reference_isotropic_vector = LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(reference_isotropic_matrix);
    const Tensor<1,21> reference_hex_vector = LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(reference_hex_matrix);
    const Tensor<1,21> reference_T_vector = LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(reference_T_matrix);
    const Tensor<1,21> reference_O_vector = LpoElasticTensor<3>::transform_6x6_matrix_to_21D_vector(reference_O_matrix);

    // this matrix is alreay in the SCCS, so no rotation needed, only the  projection.
    CHECK(full_elastic_vector.norm_square() == Approx(198012.0));
    const Tensor<1,21> mono_and_higher_vector = DecomposeElasticMatrix<3>::projection_matrix_tric_to_mono * full_elastic_vector;
    const Tensor<1,21> tric_vector = full_elastic_vector-mono_and_higher_vector;
    for (size_t iii = 0; iii < 21; iii++)
      {
        CHECK(std::abs(tric_vector[iii]) < 1e-15);
      }
    CHECK(tric_vector.norm_square() == Approx(0.0));

    dealii::Tensor<1,9>  mono_and_higher_vector_croped;
    mono_and_higher_vector_croped[0] = mono_and_higher_vector[0];
    mono_and_higher_vector_croped[1] = mono_and_higher_vector[1];
    mono_and_higher_vector_croped[2] = mono_and_higher_vector[2];
    mono_and_higher_vector_croped[3] = mono_and_higher_vector[3];
    mono_and_higher_vector_croped[4] = mono_and_higher_vector[4];
    mono_and_higher_vector_croped[5] = mono_and_higher_vector[5];
    mono_and_higher_vector_croped[6] = mono_and_higher_vector[6];
    mono_and_higher_vector_croped[7] = mono_and_higher_vector[7];
    mono_and_higher_vector_croped[8] = mono_and_higher_vector[8];
    const Tensor<1,9> ortho_and_higher_vector = DecomposeElasticMatrix<3>::projection_matrix_mono_to_ortho*mono_and_higher_vector_croped;
    const Tensor<1,9> mono_vector = mono_and_higher_vector_croped-ortho_and_higher_vector;
    for (size_t iii = 0; iii < 9; iii++)
      {
        CHECK(std::abs(mono_vector[iii]) < 1e-15);
      }
    CHECK(mono_vector.norm_square() == Approx(0.0));

    const Tensor<1,9> tetra_and_higher_vector = DecomposeElasticMatrix<3>::projection_matrix_ortho_to_tetra*ortho_and_higher_vector;
    const Tensor<1,9> ortho_vector = ortho_and_higher_vector-tetra_and_higher_vector;
    for (size_t iii = 0; iii < 9; iii++)
      {
        CHECK(ortho_vector[iii] == Approx(reference_O_vector[iii]));
      }
    CHECK(ortho_vector.norm_square() == Approx(536.0));

    const Tensor<1,9> hexa_and_higher_vector = DecomposeElasticMatrix<3>::projection_matrix_tetra_to_hexa*tetra_and_higher_vector;
    const Tensor<1,9> tetra_vector = tetra_and_higher_vector-hexa_and_higher_vector;
    for (size_t iii = 0; iii < 9; iii++)
      {
        CHECK(tetra_vector[iii] == Approx(reference_T_vector[iii]));
      }
    CHECK(tetra_vector.norm_square() == Approx(72.0));

    const Tensor<1,9> iso_vector = DecomposeElasticMatrix<3>::projection_matrix_hexa_to_iso*hexa_and_higher_vector;
    for (size_t iii = 0; iii < 9; iii++)
      {
        // something weird is going on with the approx, somehow without a very large
        // epsilon it thinks for example that "194.6666666667 == Approx( 194.7 )" is false.
        CHECK(iso_vector[iii] == Approx(reference_isotropic_vector[iii]).epsilon(1e-3));
      }
    CHECK(iso_vector.norm_square() == Approx(189529.3333333334));

    const Tensor<1,9> hexa_vector = hexa_and_higher_vector-iso_vector;
    for (size_t iii = 0; iii < 9; iii++)
      {
        // same as above.
        CHECK(hexa_vector[iii] == Approx(reference_hex_vector[iii]).epsilon(2.5e-2));
      }
    CHECK(hexa_vector.norm_square() == Approx(7874.6666666667));

  }

  {
    // now compute the distribution in a elastic tensor which is not in SCC.
    // We will use the result (with the results of the steps in between) from
    // Cowin and Mehrabadi, 1985, Mech. appl. Math.
    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 1769.50;
    full_elastic_matrix[0][1] = 873.50;
    full_elastic_matrix[0][2] = 838.22;
    full_elastic_matrix[0][3] = -17.68;
    full_elastic_matrix[0][4] = -110.32;
    full_elastic_matrix[0][5] = 144.92;
    full_elastic_matrix[1][0] = 873.50;
    full_elastic_matrix[1][1] = 1846.64;
    full_elastic_matrix[1][2] = 836.66;
    full_elastic_matrix[1][3] = -37.60;
    full_elastic_matrix[1][4] = -32.32;
    full_elastic_matrix[1][5] = 153.80;
    full_elastic_matrix[2][0] = 838.22;
    full_elastic_matrix[2][1] = 836.66;
    full_elastic_matrix[2][2] = 1603.28;
    full_elastic_matrix[2][3] = -29.68;
    full_elastic_matrix[2][4] = -93.52;
    full_elastic_matrix[2][5] = 22.40;
    full_elastic_matrix[3][0] = -17.68;
    full_elastic_matrix[3][1] = -37.60;
    full_elastic_matrix[3][2] = -29.68;
    full_elastic_matrix[3][3] = 438.77;
    full_elastic_matrix[3][4] = 57.68;
    full_elastic_matrix[3][5] = -50.50;
    full_elastic_matrix[4][0] = -110.32;
    full_elastic_matrix[4][1] = -32.32;
    full_elastic_matrix[4][2] = -93.52;
    full_elastic_matrix[4][3] = 57.68;
    full_elastic_matrix[4][4] = 439.79;
    full_elastic_matrix[4][5] = -34.78;
    full_elastic_matrix[5][0] = 144.92;
    full_elastic_matrix[5][1] = 153.80;
    full_elastic_matrix[5][2] = 22.40;
    full_elastic_matrix[5][3] = -50.50;
    full_elastic_matrix[5][4] = -34.78;
    full_elastic_matrix[5][5] = 501.80;
    full_elastic_matrix /= 81.;

    SymmetricTensor<2,3> reference_dilatation_matrix;
    reference_dilatation_matrix[0][0] = 386.80;
    reference_dilatation_matrix[0][1] = 35.68;
    reference_dilatation_matrix[0][2] = -26.24;
    //reference_dilatation_matrix[1][0] = ;
    reference_dilatation_matrix[1][1] = 395.20;
    reference_dilatation_matrix[1][2] = -9.44;
    //reference_dilatation_matrix[2][0] = ;
    //reference_dilatation_matrix[2][1] = ;
    reference_dilatation_matrix[2][2] = 364.24;
    reference_dilatation_matrix /= 9.;

    SymmetricTensor<2,3> reference_voigt_matrix;
    reference_voigt_matrix[0][0] = 33.47;
    reference_voigt_matrix[0][1] = 4.4;
    reference_voigt_matrix[0][2] = -3.14;
    //reference_voigt_matrix[1][0] = ;
    reference_voigt_matrix[1][1] = 34.41;
    reference_voigt_matrix[1][2] = -1.26;
    //reference_voigt_matrix[2][0] = ;
    //reference_voigt_matrix[2][1] = ;
    reference_voigt_matrix[2][2] = 30.64;

    // Q in equation 2.17, but reordered from
    // largest to smallest eigenvalue.
    Tensor<2,3> reference_SCC;
    reference_SCC[2][0] = 2./3.;
    reference_SCC[2][1] = -1./3.;
    reference_SCC[2][2] = 2./3.;
    reference_SCC[1][0] = 1./3.;
    reference_SCC[1][1] = -2./3.;
    reference_SCC[1][2] = -2./3.;
    reference_SCC[0][0] = 2./3.;
    reference_SCC[0][1] = 2./3.;
    reference_SCC[0][2] = -1./3.;

    SymmetricTensor<2,6> reference_rotated_elastic_matrix;
    reference_rotated_elastic_matrix[0][0] = 18;
    reference_rotated_elastic_matrix[0][1] = 9.98;
    reference_rotated_elastic_matrix[0][2] = 10.1;
    //reference_rotated_elastic_matrix[1][0] = ;
    reference_rotated_elastic_matrix[1][1] = 20.2;
    reference_rotated_elastic_matrix[1][2] = 10.07;
    //reference_rotated_elastic_matrix[2][0] = ;
    //reference_rotated_elastic_matrix[2][1] = ;
    reference_rotated_elastic_matrix[2][2] = 27.6;
    reference_rotated_elastic_matrix[3][3] = 6.23;
    reference_rotated_elastic_matrix[4][4] = 5.61;
    reference_rotated_elastic_matrix[5][5] = 4.52;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    for (size_t iii = 0; iii < 3; iii++)
      {
        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            CHECK(reference_dilatation_matrix[iii][jjj] == Approx(dilatation_stiffness_tensor_full[iii][jjj]));
          }
      }

    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    for (size_t iii = 0; iii < 3; iii++)
      {
        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            CHECK(reference_voigt_matrix[iii][jjj] == Approx(voigt_stiffness_tensor_full[iii][jjj]));
          }
      }
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    // chose the permutation which fits the solution given in the paper.
    // This is not a even permutation.
    std::array<unsigned short int, 3> perumation = {{2,1,0}};
    Tensor<2,3> permutated_SCC;
    for (size_t j = 0; j < 3; j++)
      {
        permutated_SCC[j] = unpermutated_SCC[perumation[j]];
      }
    SymmetricTensor<2,6> rotated_elastic_matrix = LpoElasticTensor<3>::rotate_6x6_matrix(full_elastic_matrix,permutated_SCC);

    for (size_t iii = 0; iii < 6; iii++)
      {
        for (size_t jjj = 0; jjj < 6; jjj++)
          {
            CHECK(reference_rotated_elastic_matrix[iii][jjj] == Approx(rotated_elastic_matrix[iii][jjj]).margin(1e-4).epsilon(7.5e-2));
          }
      }
  }
  {
    // This is testing some other elastic matrices of which the properties can
    // be resonaly predicted (for example, mostly hexagonal) and some coverage
    // testing: Case 1.

    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 192;
    full_elastic_matrix[0][1] = 66;
    full_elastic_matrix[0][2] = 56;
    full_elastic_matrix[1][1] = 192;
    full_elastic_matrix[1][2] = 56;
    full_elastic_matrix[2][2] = 272;
    full_elastic_matrix[3][3] = 60;
    full_elastic_matrix[4][4] = 60;
    full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);

    std::array<std::array<double,3>,7 > reference_norms;
    reference_norms[0] = {{0,0,0}}; // no triclinic
    reference_norms[1] = {{0,0,0}}; // no monoclinic
    reference_norms[2] = {{12822.0,0,12822.0}}; // orthorhomic depedent on direction (remember, this is basically just one grain)
    reference_norms[3] = {{1568.0,8712.0,1568.0}}; // tetragonal could be present
    reference_norms[4] = {{2759.3333333333,8437.3333333333,2759.3333333333}}; // hexagonal is expected
    reference_norms[5] = {{247182.6666666667,247182.6666666667,247182.6666666667}}; // isotropic should be the same for every permuation
    reference_norms[6] = {{264332.0,264332.0,264332.0}}; // total should be the same for every direction

    // It is already in a SCC frame, so they should be
    // unit vectors. The order is irrelevant.
    Tensor<2,3> reference_unpermutated_SCC;
    reference_unpermutated_SCC[0][0] = 0;
    reference_unpermutated_SCC[0][1] = 0;
    reference_unpermutated_SCC[0][2] = 1;
    reference_unpermutated_SCC[1][0] = 1;
    reference_unpermutated_SCC[1][1] = 0;
    reference_unpermutated_SCC[1][2] = 0;
    reference_unpermutated_SCC[2][0] = 0;
    reference_unpermutated_SCC[2][1] = 1;
    reference_unpermutated_SCC[2][2] = 0;

    // Permuation 2 has the heighest norm, so reference_unpermutated_SCC
    // is permuated by {1,2,0};
    Tensor<2,3> reference_hexa_permutated_SCC;
    reference_hexa_permutated_SCC[0][0] = 1;
    reference_hexa_permutated_SCC[0][1] = 0;
    reference_hexa_permutated_SCC[0][2] = 0;
    reference_hexa_permutated_SCC[1][0] = 0;
    reference_hexa_permutated_SCC[1][1] = 1;
    reference_hexa_permutated_SCC[1][2] = 0;
    reference_hexa_permutated_SCC[2][0] = 0;
    reference_hexa_permutated_SCC[2][1] = 0;
    reference_hexa_permutated_SCC[2][2] = 1;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("unpermutated_SCC[" << iii << "] = " << unpermutated_SCC[iii][0] << ":" << unpermutated_SCC[iii][1] << ":" << unpermutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_unpermutated_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_unpermutated_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    std::array<std::array<double,3>,7 > norms = DecomposeElasticMatrix<3>::compute_elastic_tensor_SCC_decompositions(unpermutated_SCC, full_elastic_matrix);
    std::array<double,3> totals = {{0,0,0}};
    for (size_t iii = 0; iii < 7; iii++)
      {
        // also check that the total is equal to the full norm.

        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            INFO("i:j = " << iii << ":" << jjj );
            CHECK(reference_norms[iii][jjj] == Approx(norms[iii][jjj]));
            if (iii < 6)
              {
                totals[jjj] += norms[iii][jjj];
              }
          }
      }
    for (size_t jjj = 0; jjj < 3; jjj++)
      {
        INFO("j = " << jjj );
        CHECK(totals[jjj] == Approx(norms[6][jjj]));
      }

    // also check that all the isotropic and totals are the same
    for (size_t iii = 1; iii < 3; iii++)
      {
        CHECK(norms[5][0] == Approx(norms[5][iii]));
        CHECK(norms[6][0] == Approx(norms[6][iii]));
      }

    const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
    std::array<unsigned short int, 3> perumation = DecomposeElasticMatrix<3>::indexed_even_permutation(max_hexagonal_element_index);
    Tensor<2,3> hexa_permutated_SCC;
    for (size_t index = 0; index < 3; index++)
      {
        hexa_permutated_SCC[index] = unpermutated_SCC[perumation[index]];
      }
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("hexa_permutated_SCC[" << iii << "] = " << hexa_permutated_SCC[iii][0] << ":" << hexa_permutated_SCC[iii][1] << ":" << hexa_permutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_hexa_permutated_SCC[iii][0] == Approx(hexa_permutated_SCC[iii][0]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][1] == Approx(hexa_permutated_SCC[iii][1]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][2] == Approx(hexa_permutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_hexa_permutated_SCC[iii][0] == Approx(-hexa_permutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][1] == Approx(-hexa_permutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][2] == Approx(-hexa_permutated_SCC[iii][2]).epsilon(epsilon))));
      }
  }
  {
    // This is testing some other elastic matrices of which the properties can
    // be resonaly predicted (for example, mostly hexagonal) and some coverage
    // testing: Case 2: a more complicated matrix.

    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 225;
    full_elastic_matrix[0][1] = 54;
    full_elastic_matrix[0][2] = 72;
    full_elastic_matrix[1][1] = 214;
    full_elastic_matrix[1][2] = 53;
    full_elastic_matrix[2][2] = 178;
    full_elastic_matrix[3][3] = 78;
    full_elastic_matrix[4][4] = 82;
    full_elastic_matrix[5][5] = 76;

    std::array<std::array<double,3>,7 > reference_norms;
    reference_norms[0] = {{0,0,0}}; // no triclinic
    reference_norms[1] = {{0,0,0}}; // no monoclinic
    reference_norms[2] = {{453.5,1044.0,1113.5}}; // orthorhomic depedent on direction
    reference_norms[3] = {{91.125,84.5,595.125}}; // tetragonal could be present
    reference_norms[4] = {{1350.175,766.3,186.175}}; // hexagonal is expected
    reference_norms[5] = {{222364.2,222364.2,222364.2}}; // isotropic should be the same for every permuation
    reference_norms[6] = {{224259.0,224259.0,224259.0}}; // total should be the same for every direction

    // It is already in a SCC frame, so they should be
    // unit vectors. The order is irrelevant.
    Tensor<2,3> reference_unpermutated_SCC;
    reference_unpermutated_SCC[0][0] = 1;
    reference_unpermutated_SCC[0][1] = 0;
    reference_unpermutated_SCC[0][2] = 0;
    reference_unpermutated_SCC[1][0] = 0;
    reference_unpermutated_SCC[1][1] = 1;
    reference_unpermutated_SCC[1][2] = 0;
    reference_unpermutated_SCC[2][0] = 0;
    reference_unpermutated_SCC[2][1] = 0;
    reference_unpermutated_SCC[2][2] = 1;

    // Permuation 1 has the heighest norm, so reference_unpermutated_SCC
    // is permuated by {0,1,2};
    Tensor<2,3> reference_hexa_permutated_SCC;
    reference_hexa_permutated_SCC[0][0] = 1;
    reference_hexa_permutated_SCC[0][1] = 0;
    reference_hexa_permutated_SCC[0][2] = 0;
    reference_hexa_permutated_SCC[1][0] = 0;
    reference_hexa_permutated_SCC[1][1] = 1;
    reference_hexa_permutated_SCC[1][2] = 0;
    reference_hexa_permutated_SCC[2][0] = 0;
    reference_hexa_permutated_SCC[2][1] = 0;
    reference_hexa_permutated_SCC[2][2] = 1;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("unpermutated_SCC[" << iii << "] = " << unpermutated_SCC[iii][0] << ":" << unpermutated_SCC[iii][1] << ":" << unpermutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_unpermutated_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_unpermutated_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    std::array<std::array<double,3>,7 > norms = DecomposeElasticMatrix<3>::compute_elastic_tensor_SCC_decompositions(unpermutated_SCC, full_elastic_matrix);
    std::array<double,3> totals = {{0,0,0}};
    for (size_t iii = 0; iii < 7; iii++)
      {
        // also check that the total is equal to the full norm.

        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            INFO("i:j = " << iii << ":" << jjj );
            CHECK(reference_norms[iii][jjj] == Approx(norms[iii][jjj]));
            if (iii < 6)
              {
                totals[jjj] += norms[iii][jjj];
              }
          }
      }
    for (size_t jjj = 0; jjj < 3; jjj++)
      {
        INFO("j = " << jjj );
        CHECK(totals[jjj] == Approx(norms[6][jjj]));
      }

    // also check that all the isotropic and totals are the same
    for (size_t iii = 1; iii < 3; iii++)
      {
        CHECK(norms[5][0] == Approx(norms[5][iii]));
        CHECK(norms[6][0] == Approx(norms[6][iii]));
      }

    const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
    std::array<unsigned short int, 3> perumation = DecomposeElasticMatrix<3>::indexed_even_permutation(max_hexagonal_element_index);
    Tensor<2,3> hexa_permutated_SCC;
    for (size_t index = 0; index < 3; index++)
      {
        hexa_permutated_SCC[index] = unpermutated_SCC[perumation[index]];
      }
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("hexa_permutated_SCC[" << iii << "] = " << hexa_permutated_SCC[iii][0] << ":" << hexa_permutated_SCC[iii][1] << ":" << hexa_permutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_hexa_permutated_SCC[iii][0] == Approx(hexa_permutated_SCC[iii][0]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][1] == Approx(hexa_permutated_SCC[iii][1]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][2] == Approx(hexa_permutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_hexa_permutated_SCC[iii][0] == Approx(-hexa_permutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][1] == Approx(-hexa_permutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][2] == Approx(-hexa_permutated_SCC[iii][2]).epsilon(epsilon))));
      }
  }

  {
    // This is testing some other elastic matrices of which the properties can
    // be resonaly predicted (for example, mostly hexagonal) and some coverage
    // testing: Case 3: olivine from experiments as used in D-Rex.

    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 320.71;
    full_elastic_matrix[0][1] = 69.84;
    full_elastic_matrix[0][2] = 71.22;
    full_elastic_matrix[1][1] = 197.25;
    full_elastic_matrix[1][2] = 74.8;
    full_elastic_matrix[2][2] = 234.32;
    full_elastic_matrix[3][3] = 63.77;
    full_elastic_matrix[4][4] = 77.67;
    full_elastic_matrix[5][5] = 78.36;

    std::array<std::array<double,3>,7 > reference_norms;
    reference_norms[0] = {{0,0,0}}; // no triclinic
    reference_norms[1] = {{0,0,0}}; // no monoclinic
    reference_norms[2] = {{4181.95385,689.94905,8020.4222}}; // orthorhomic depedent on direction
    reference_norms[3] = {{1298.2060125,90.3840125,525.5282}}; // tetragonal could be present
    reference_norms[4] = {{4364.6051908333,9064.4319908333,1298.8146533333}}; // hexagonal is expected
    reference_norms[5] = {{282871.5975466666,282871.5975466666,282871.5975466666}}; // isotropic should be the same for every permuation
    reference_norms[6] = {{292716.3626,292716.3626,292716.3626}}; // total should be the same for every direction

    // It is already in a SCC frame, so they should be
    // unit vectors. The order is irrelevant.
    Tensor<2,3> reference_unpermutated_SCC;
    reference_unpermutated_SCC[0][0] = 1;
    reference_unpermutated_SCC[0][1] = 0;
    reference_unpermutated_SCC[0][2] = 0;
    reference_unpermutated_SCC[1][0] = 0;
    reference_unpermutated_SCC[1][1] = 0;
    reference_unpermutated_SCC[1][2] = 1;
    reference_unpermutated_SCC[2][0] = 0;
    reference_unpermutated_SCC[2][1] = 1;
    reference_unpermutated_SCC[2][2] = 0;

    // Permuation 2 has the heighest norm, so reference_unpermutated_SCC
    // is permuated by {1,2,0};
    Tensor<2,3> reference_hexa_permutated_SCC;
    reference_hexa_permutated_SCC[0][0] = 0;
    reference_hexa_permutated_SCC[0][1] = 0;
    reference_hexa_permutated_SCC[0][2] = 1;
    reference_hexa_permutated_SCC[1][0] = 0;
    reference_hexa_permutated_SCC[1][1] = 1;
    reference_hexa_permutated_SCC[1][2] = 0;
    reference_hexa_permutated_SCC[2][0] = 1;
    reference_hexa_permutated_SCC[2][1] = 0;
    reference_hexa_permutated_SCC[2][2] = 0;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("unpermutated_SCC[" << iii << "] = " << unpermutated_SCC[iii][0] << ":" << unpermutated_SCC[iii][1] << ":" << unpermutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_unpermutated_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_unpermutated_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    std::array<std::array<double,3>,7 > norms = DecomposeElasticMatrix<3>::compute_elastic_tensor_SCC_decompositions(unpermutated_SCC, full_elastic_matrix);
    std::array<double,3> totals = {{0,0,0}};
    for (size_t iii = 0; iii < 7; iii++)
      {
        // also check that the total is equal to the full norm.

        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            INFO("i:j = " << iii << ":" << jjj );
            CHECK(reference_norms[iii][jjj] == Approx(norms[iii][jjj]));
            if (iii < 6)
              {
                totals[jjj] += norms[iii][jjj];
              }
          }
      }
    for (size_t jjj = 0; jjj < 3; jjj++)
      {
        INFO("j = " << jjj );
        CHECK(totals[jjj] == Approx(norms[6][jjj]));
      }

    // also check that all the isotropic and totals are the same
    for (size_t iii = 1; iii < 3; iii++)
      {
        CHECK(norms[5][0] == Approx(norms[5][iii]));
        CHECK(norms[6][0] == Approx(norms[6][iii]));
      }

    const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
    std::array<unsigned short int, 3> perumation = DecomposeElasticMatrix<3>::indexed_even_permutation(max_hexagonal_element_index);
    Tensor<2,3> hexa_permutated_SCC;
    for (size_t index = 0; index < 3; index++)
      {
        hexa_permutated_SCC[index] = unpermutated_SCC[perumation[index]];
      }
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("hexa_permutated_SCC[" << iii << "] = " << hexa_permutated_SCC[iii][0] << ":" << hexa_permutated_SCC[iii][1] << ":" << hexa_permutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_hexa_permutated_SCC[iii][0] == Approx(hexa_permutated_SCC[iii][0]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][1] == Approx(hexa_permutated_SCC[iii][1]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][2] == Approx(hexa_permutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_hexa_permutated_SCC[iii][0] == Approx(-hexa_permutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][1] == Approx(-hexa_permutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][2] == Approx(-hexa_permutated_SCC[iii][2]).epsilon(epsilon))));
      }
  }

  {
    // This is testing some other elastic matrices of which the properties can
    // be resonaly predicted (for example, mostly hexagonal) and some coverage
    // testing: Case 4: enstatite from experiments as used in D-Rex.

    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 236.9;
    full_elastic_matrix[0][1] = 79.6;
    full_elastic_matrix[0][2] = 63.2;
    full_elastic_matrix[1][1] = 180.5;
    full_elastic_matrix[1][2] = 56.8;
    full_elastic_matrix[2][2] = 230.4;
    full_elastic_matrix[3][3] = 84.3;
    full_elastic_matrix[4][4] = 79.4;
    full_elastic_matrix[5][5] = 80.1;

    std::array<std::array<double,3>,7 > reference_norms;
    reference_norms[0] = {{0,0,0}}; // no triclinic
    reference_norms[1] = {{0,0,0}}; // no monoclinic
    reference_norms[2] = {{576.245,1514.945,1679.46}}; // orthorhomic depedent on direction
    reference_norms[3] = {{67.86125,199.00125,483.605}}; // tetragonal could be present
    reference_norms[4] = {{2076.64175,1006.80175,557.683}}; // hexagonal is expected
    reference_norms[5] = {{245485.992,245485.992,245485.992}}; // isotropic should be the same for every permuation
    reference_norms[6] = {{248206.74,248206.74,248206.74}}; // total should be the same for every direction

    // It is already in a SCC frame, so they should be
    // unit vectors. The order is irrelevant.
    Tensor<2,3> reference_unpermutated_SCC;
    reference_unpermutated_SCC[0][0] = 1;
    reference_unpermutated_SCC[0][1] = 0;
    reference_unpermutated_SCC[0][2] = 0;
    reference_unpermutated_SCC[1][0] = 0;
    reference_unpermutated_SCC[1][1] = 0;
    reference_unpermutated_SCC[1][2] = 1;
    reference_unpermutated_SCC[2][0] = 0;
    reference_unpermutated_SCC[2][1] = 1;
    reference_unpermutated_SCC[2][2] = 0;

    // Permuation 1 has the heighest norm, so reference_unpermutated_SCC
    // is permuated by {0,1,2};
    Tensor<2,3> reference_hexa_permutated_SCC;
    reference_hexa_permutated_SCC[0][0] = 1;
    reference_hexa_permutated_SCC[0][1] = 0;
    reference_hexa_permutated_SCC[0][2] = 0;
    reference_hexa_permutated_SCC[1][0] = 0;
    reference_hexa_permutated_SCC[1][1] = 0;
    reference_hexa_permutated_SCC[1][2] = 1;
    reference_hexa_permutated_SCC[2][0] = 0;
    reference_hexa_permutated_SCC[2][1] = 1;
    reference_hexa_permutated_SCC[2][2] = 0;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("unpermutated_SCC[" << iii << "] = " << unpermutated_SCC[iii][0] << ":" << unpermutated_SCC[iii][1] << ":" << unpermutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_unpermutated_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_unpermutated_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    std::array<std::array<double,3>,7 > norms = DecomposeElasticMatrix<3>::compute_elastic_tensor_SCC_decompositions(unpermutated_SCC, full_elastic_matrix);
    std::array<double,3> totals = {{0,0,0}};
    for (size_t iii = 0; iii < 7; iii++)
      {
        // also check that the total is equal to the full norm.

        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            INFO("i:j = " << iii << ":" << jjj );
            CHECK(reference_norms[iii][jjj] == Approx(norms[iii][jjj]));
            if (iii < 6)
              {
                totals[jjj] += norms[iii][jjj];
              }
          }
      }
    for (size_t jjj = 0; jjj < 3; jjj++)
      {
        INFO("j = " << jjj );
        CHECK(totals[jjj] == Approx(norms[6][jjj]));
      }

    // also check that all the isotropic and totals are the same
    for (size_t iii = 1; iii < 3; iii++)
      {
        CHECK(norms[5][0] == Approx(norms[5][iii]));
        CHECK(norms[6][0] == Approx(norms[6][iii]));
      }


    const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
    std::array<unsigned short int, 3> perumation = DecomposeElasticMatrix<3>::indexed_even_permutation(max_hexagonal_element_index);
    Tensor<2,3> hexa_permutated_SCC;
    for (size_t index = 0; index < 3; index++)
      {
        hexa_permutated_SCC[index] = unpermutated_SCC[perumation[index]];
      }
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("hexa_permutated_SCC[" << iii << "] = " << hexa_permutated_SCC[iii][0] << ":" << hexa_permutated_SCC[iii][1] << ":" << hexa_permutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_hexa_permutated_SCC[iii][0] == Approx(hexa_permutated_SCC[iii][0]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][1] == Approx(hexa_permutated_SCC[iii][1]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][2] == Approx(hexa_permutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_hexa_permutated_SCC[iii][0] == Approx(-hexa_permutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][1] == Approx(-hexa_permutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][2] == Approx(-hexa_permutated_SCC[iii][2]).epsilon(epsilon))));
      }
  }

  {
    // This is testing some other elastic matrices of which the properties can
    // be resonaly predicted (for example, mostly hexagonal) and some coverage
    // testing: Case 4: from http://solidmechanics.org/Text/Chapter3_2/Chapter3_2.php (Be).
    // This one should have at least in one direction almost all the the anistropy be
    // hexagonal anisotropy.

    SymmetricTensor<2,6> full_elastic_matrix;
    full_elastic_matrix[0][0] = 192.3;
    full_elastic_matrix[0][1] = 26.7;
    full_elastic_matrix[0][2] = 14;
    full_elastic_matrix[1][1] = full_elastic_matrix[0][0];
    full_elastic_matrix[1][2] = full_elastic_matrix[0][2];
    full_elastic_matrix[2][2] = 336.4;
    full_elastic_matrix[3][3] = 162.5;
    full_elastic_matrix[4][4] = full_elastic_matrix[3][3];
    full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);

    std::array<std::array<double,3>,7 > reference_norms;
    reference_norms[0] = {{0,0,0}}; // no triclinic
    reference_norms[1] = {{0,0,0}}; // no monoclinic
    reference_norms[2] = {{16161.695,0.0,16161.695}}; // orthorhomic depedent on direction
    reference_norms[3] = {{2786.31125,1425.78,2786.31125}}; // tetragonal could be present
    reference_norms[4] = {{8079.22575,25601.452,8079.22575}}; // hexagonal is expected
    reference_norms[5] = {{421517.088,421517.088,421517.088}}; // isotropic should be the same for every permuation
    reference_norms[6] = {{448544.32,448544.32,448544.32}}; // total should be the same for every direction

    // It is already in a SCC frame, so they should be
    // unit vectors. The order is irrelevant.
    Tensor<2,3> reference_unpermutated_SCC;
    reference_unpermutated_SCC[0][0] = 0;
    reference_unpermutated_SCC[0][1] = 0;
    reference_unpermutated_SCC[0][2] = 1;
    reference_unpermutated_SCC[1][0] = 1;
    reference_unpermutated_SCC[1][1] = 0;
    reference_unpermutated_SCC[1][2] = 0;
    reference_unpermutated_SCC[2][0] = 0;
    reference_unpermutated_SCC[2][1] = 1;
    reference_unpermutated_SCC[2][2] = 0;

    // Permuation 2 has the heighest norm, so reference_unpermutated_SCC
    // is permuated by {1,2,0};
    Tensor<2,3> reference_hexa_permutated_SCC;
    reference_hexa_permutated_SCC[0][0] = 1;
    reference_hexa_permutated_SCC[0][1] = 0;
    reference_hexa_permutated_SCC[0][2] = 0;
    reference_hexa_permutated_SCC[1][0] = 0;
    reference_hexa_permutated_SCC[1][1] = 1;
    reference_hexa_permutated_SCC[1][2] = 0;
    reference_hexa_permutated_SCC[2][0] = 0;
    reference_hexa_permutated_SCC[2][1] = 0;
    reference_hexa_permutated_SCC[2][2] = 1;

    const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_dilatation_stiffness_tensor(full_elastic_matrix);
    const SymmetricTensor<2,3> voigt_stiffness_tensor_full = DecomposeElasticMatrix<3>::compute_voigt_stiffness_tensor(full_elastic_matrix);
    Tensor<2,3> unpermutated_SCC = DecomposeElasticMatrix<3>::compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("unpermutated_SCC[" << iii << "] = " << unpermutated_SCC[iii][0] << ":" << unpermutated_SCC[iii][1] << ":" << unpermutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_unpermutated_SCC[iii][0] == Approx(unpermutated_SCC[iii][0]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][1] == Approx(unpermutated_SCC[iii][1]).epsilon(epsilon)
               && reference_unpermutated_SCC[iii][2] == Approx(unpermutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_unpermutated_SCC[iii][0] == Approx(-unpermutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][1] == Approx(-unpermutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_unpermutated_SCC[iii][2] == Approx(-unpermutated_SCC[iii][2]).epsilon(epsilon))));
      }
    std::array<std::array<double,3>,7 > norms = DecomposeElasticMatrix<3>::compute_elastic_tensor_SCC_decompositions(unpermutated_SCC, full_elastic_matrix);
    std::array<double,3> totals = {{0,0,0}};
    for (size_t iii = 0; iii < 7; iii++)
      {
        // also check that the total is equal to the full norm.

        for (size_t jjj = 0; jjj < 3; jjj++)
          {
            INFO("i:j = " << iii << ":" << jjj );
            CHECK(reference_norms[iii][jjj] == Approx(norms[iii][jjj]));
            if (iii < 6)
              {
                totals[jjj] += norms[iii][jjj];
              }
          }
      }
    for (size_t jjj = 0; jjj < 3; jjj++)
      {
        INFO("j = " << jjj );
        CHECK(totals[jjj] == Approx(norms[6][jjj]));
      }

    // also check that all the isotropic and totals are the same
    for (size_t iii = 1; iii < 3; iii++)
      {
        CHECK(norms[5][0] == Approx(norms[5][iii]));
        CHECK(norms[6][0] == Approx(norms[6][iii]));
      }


    const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
    std::array<unsigned short int, 3> perumation = DecomposeElasticMatrix<3>::indexed_even_permutation(max_hexagonal_element_index);
    Tensor<2,3> hexa_permutated_SCC;
    for (size_t index = 0; index < 3; index++)
      {
        hexa_permutated_SCC[index] = unpermutated_SCC[perumation[index]];
      }
    for (size_t iii = 0; iii < 3; iii++)
      {
        const double epsilon = 1e-3;
        INFO("hexa_permutated_SCC[" << iii << "] = " << hexa_permutated_SCC[iii][0] << ":" << hexa_permutated_SCC[iii][1] << ":" << hexa_permutated_SCC[iii][2]);
        // The direction of the eigenvectors is not important, but they do need to be consistent
        CHECK((reference_hexa_permutated_SCC[iii][0] == Approx(hexa_permutated_SCC[iii][0]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][1] == Approx(hexa_permutated_SCC[iii][1]).epsilon(epsilon)
               && reference_hexa_permutated_SCC[iii][2] == Approx(hexa_permutated_SCC[iii][2]).epsilon(epsilon)
               || (reference_hexa_permutated_SCC[iii][0] == Approx(-hexa_permutated_SCC[iii][0]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][1] == Approx(-hexa_permutated_SCC[iii][1]).epsilon(epsilon)
                   && reference_hexa_permutated_SCC[iii][2] == Approx(-hexa_permutated_SCC[iii][2]).epsilon(epsilon))));
      }
  }
}
