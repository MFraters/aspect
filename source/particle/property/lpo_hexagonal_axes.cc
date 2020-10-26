/*
  Copyright (C) 2015 - 2017 by the authors of the ASPECT code.

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

//#include <cstdlib>
#include <aspect/particle/property/lpo_hexagonal_axes.h>
#include <aspect/particle/property/lpo_elastic_tensor.h>
#include <aspect/particle/property/lpo.h>
#include <aspect/particle/world.h>

#include <aspect/utilities.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {


      template <int dim>
      LpoHexagonalAxes<dim>::LpoHexagonalAxes ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;
      }

      template <int dim>
      void
      LpoHexagonalAxes<dim>::initialize ()
      {
        // todo: check wheter this works correctly. Since the get_random_number function takes a reference
        // to the random_number_generator function, changing the function should mean that I have to update the
        // get_random_number function as well. But I will need to test this.
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);
        //std::cout << ">>> random_number_seed+my_rank = " << random_number_seed+my_rank << ", random_number_seed = " << random_number_seed << std::endl;

        const Particle::Property::Manager<dim> &manager = this->get_particle_world().get_property_manager();
        //AssertThrow(manager.plugin_name_exists("lpo"),
        //            ExcMessage("No lpo property plugin found."));
        AssertThrow(manager.plugin_name_exists("lpo elastic tensor"),
                    ExcMessage("No lpo elastic tensor property plugin found."));
        Assert(manager.plugin_name_exists("lpo hexagonal axes"),
               ExcMessage("No hexagonal axes property plugin found."));

        //AssertThrow(manager.check_plugin_order("lpo","lpo hexagonal axes"),
        //            ExcMessage("To use the lpo hexagonal axes plugin, the lpo plugin need to be defined before this plugin."));

        AssertThrow(manager.check_plugin_order("lpo elastic tensor","lpo hexagonal axes"),
                    ExcMessage("To use the lpo hexagonal axes plugin, the lpo elastic tensor plugin need to be defined before this plugin."));

        //lpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo"));
        lpo_elastic_tensor_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo elastic tensor"));
//std::cout << "flag 1" << std::endl;
      }



      template <int dim>
      void
      LpoHexagonalAxes<dim>::initialize_one_particle_property(const Point<dim> &,
                                                              std::vector<double> &data) const
      {
//std::cout << "flag 10" << std::endl;

        /*double water_content = 0;
        double volume_fraction_olivine = 0;
        std::vector<double> volume_fractions_olivine(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
        std::vector<double> volume_fractions_enstatite(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);*/
        SymmetricTensor<2,6> elastic_matrix;

        /*elastic_matrix[0][0] = 192.;
        elastic_matrix[0][1] = 66.;
        elastic_matrix[0][2] = 60.;
        elastic_matrix[1][1] = 160.;
        elastic_matrix[1][2] = 56.;
        elastic_matrix[2][2] = 272.;
        elastic_matrix[3][3] = 60.;
        elastic_matrix[4][4] = 62.;
        elastic_matrix[5][5] = 49.;*/
        //std::cout << "elastic_matrix.norm = " << elastic_matrix.norm() << std::endl;

        //std::cout << "lpo_data_position = " << lpo_data_position << ", n_grains = " << n_grains << std::endl;
        /*Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         n_grains,
                                                         water_content,
                                                         volume_fraction_olivine,
                                                         volume_fractions_olivine,
                                                         a_cosine_matrices_olivine,
                                                         volume_fractions_enstatite,
                                                         a_cosine_matrices_enstatite);*/

        Particle::Property::LpoElasticTensor<dim>::load_particle_data(lpo_elastic_tensor_data_position,
                                                                      data,
                                                                      elastic_matrix);



        // These eigenvectors uniquely define the symmetry cartesian coordiante system (SCCS),
        // but we need to find the order in which

        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix);

        double elastic_vector_norm = elastic_vector.norm();
        //std::cout << "elastic tensor norm = " << elastic_vector_norm << std::endl;
        const SymmetricTensor<2,3> dilatation_stiffness_tensor = compute_dilatation_stiffness_tensor(elastic_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor = compute_voigt_stiffness_tensor(elastic_matrix);

        const std::pair<double,double> bulk_and_shear_moduli = compute_bulk_and_shear_moduli(dilatation_stiffness_tensor, voigt_stiffness_tensor);
        const double bulk_modulus = bulk_and_shear_moduli.first;
        const double shear_modulus = bulk_and_shear_moduli.second;


        //std::cout << std::endl << "bulk_modulus = " << bulk_modulus << ", shear_modulus = " << shear_modulus << std::endl;

        const Tensor<1,9> elastic_isotropic_approximation = compute_isotropic_approximation(bulk_modulus, shear_modulus);

        //std::cout << "elastic_isotropic_approximation = " << elastic_isotropic_approximation.norm() << std::endl;
        Tensor<1,21> anisotropic_elastic_vector = elastic_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            anisotropic_elastic_vector[i] -= elastic_isotropic_approximation[i];
          }
        // ANIS
        const double elastic_anisotropic_approximation = anisotropic_elastic_vector.norm();

        //std::cout << " -- " << std::endl;
        Tensor<2,3> unpermutated_SCC = compute_unpermutated_SCC(dilatation_stiffness_tensor, voigt_stiffness_tensor);

        // This return the minimal hexagonal projected elastic matrix as a tensor<2,6> and the corresponding SCC as a Tensor<2,3>
        std::pair<SymmetricTensor<2,6>,Tensor<2,3> > elastic_minimum_hexagonal_projection = compute_minimum_hexagonal_projection(unpermutated_SCC, elastic_matrix, elastic_vector_norm);

        const Tensor<1,21> elastic_minimum_hexagonal_projection_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_minimum_hexagonal_projection.first);

        const Tensor<1,9> hexagonal_elastic_vector = project_onto_hexagonal_symmetry(elastic_minimum_hexagonal_projection_vector);

        // std::cout << " -- " << std::endl;
        Tensor<1,21> hexagonal_elastic_vector_residual = elastic_minimum_hexagonal_projection_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            hexagonal_elastic_vector_residual[i] -= hexagonal_elastic_vector[i];
          }

        // DC5
        const double hexagonal_elastic_residual = hexagonal_elastic_vector_residual.norm();

        //std::cout << "hexagonal_elastic_vector = " << hexagonal_elastic_vector << std::endl;
        //std::cout << "elastic_minimum_hexagonal_projection_vector = " << elastic_minimum_hexagonal_projection_vector << ", (" << elastic_minimum_hexagonal_projection_vector.norm() << ")" << std::endl;
        //std::cout << "elastic_minimum_hexagonal_projection.first = " << elastic_minimum_hexagonal_projection.first << std::endl;


        //std::cout << "hexagonal_elastic_residual = " << hexagonal_elastic_residual << std::endl;




        //{
        const SymmetricTensor<2,3> dilatation_stiffness_tensor_projected = compute_dilatation_stiffness_tensor(elastic_minimum_hexagonal_projection.first);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_projected = compute_voigt_stiffness_tensor(elastic_minimum_hexagonal_projection.first);

        const std::pair<double,double> bulk_and_shear_moduli_projected = compute_bulk_and_shear_moduli(dilatation_stiffness_tensor, voigt_stiffness_tensor);
        const double bulk_modulus_projected = bulk_and_shear_moduli.first;
        const double shear_modulus_projected = bulk_and_shear_moduli.second;


        //std::cout << std::endl << "bulk_modulus_projected = " << bulk_modulus_projected << ", shear_modulus_projected = " << shear_modulus_projected << std::endl;

        const Tensor<1,9> elastic_isotropic_approximation_projected = compute_isotropic_approximation(bulk_modulus_projected, shear_modulus_projected);
        //std::cout << "elastic_isotropic_approximation = " << elastic_isotropic_approximation_projected.norm() << std::endl;

        Tensor<1,21> anisotropic_elastic_vector_projected = elastic_minimum_hexagonal_projection_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            anisotropic_elastic_vector_projected[i] -= elastic_isotropic_approximation_projected[i];
          }
        // ANIS
        const double elastic_anisotropic_approximation_vector_projected = anisotropic_elastic_vector_projected.norm();

        const double hex_percentage = ((elastic_anisotropic_approximation_vector_projected - hexagonal_elastic_residual)/elastic_minimum_hexagonal_projection_vector.norm())*100.;
        //data[data_position] = new_percentage;
        //std::cout << "elastic_anisotropic_approximation_vector_projected = " << elastic_anisotropic_approximation_vector_projected << ", hexagonal_elastic_residual = " << hexagonal_elastic_residual << ", elastic_minimum_hexagonal_projection_vector.norm() = " << elastic_minimum_hexagonal_projection_vector.norm() << std::endl;
        //}

        // percentage = (ANIS-DC5)/XN*100
        // This is not the same equation as in Browaeys and Chevrot gfi 2004, but it
        // does seem to give the same result for hexagonal:
        // 100percent = N^{-2}(X)[N^2(X_{tric})+N^2(X_{mon})+N^2(X_{ort})+N^2(X_{tet})+N^2(X_{hex})+N^2(X_{iso})],
        // where N(X) is defined as sqrt(X_i X_i).
        const double total_anis_percentage = ((elastic_anisotropic_approximation)/elastic_vector_norm)*100.;

        const double percentage = ((elastic_anisotropic_approximation - hexagonal_elastic_residual)/elastic_vector_norm)*100.;
        //const double percentage = ((elastic_isotropic_approximation.norm() - hexagonal_elastic_residual)/elastic_vector_norm)*100;
        //const double percentage2 = (elastic_anisotropic_approximation*elastic_anisotropic_approximation - hexagonal_elastic_residual*hexagonal_elastic_residual)/(elastic_vector_norm*elastic_vector_norm)*100;
        const double percentage2 = (elastic_isotropic_approximation.norm()*elastic_isotropic_approximation.norm())/(elastic_vector_norm*elastic_vector_norm)*100;
        //std::cout << "--> new_percentage = " << hex_percentage << ", total_anis_percentage = " << total_anis_percentage << ", percentage hex of anis = " << (hex_percentage/total_anis_percentage)*100.0<< ", old percentage = " << percentage << ", percentage2 = " << percentage2 << ", iso/ansi = " << elastic_anisotropic_approximation << ", hexa/DC5 = " << hexagonal_elastic_residual << ", full/XN = " << elastic_vector_norm << std::endl;

        data.push_back(total_anis_percentage);
        data.push_back(hex_percentage);


        Tensor<2,3> SCC = elastic_minimum_hexagonal_projection.second;

        data.push_back(SCC[2][0]);
        data.push_back(SCC[2][1]);
        data.push_back(SCC[2][2]);
        data.push_back(SCC[1][0]);
        data.push_back(SCC[1][1]);
        data.push_back(SCC[1][2]);
        data.push_back(SCC[0][0]);
        data.push_back(SCC[0][1]);
        data.push_back(SCC[0][2]);

        SymmetricTensor<2,6> elastic_isotropic_tensor;

        elastic_isotropic_tensor[0][0] = 194.7;
        elastic_isotropic_tensor[0][1] = 67.3;
        elastic_isotropic_tensor[0][2] = 67.3;
        elastic_isotropic_tensor[1][1] = 194.7;
        elastic_isotropic_tensor[1][2] = 67.3;
        elastic_isotropic_tensor[2][2] = 194.7;
        elastic_isotropic_tensor[3][3] = 63.7;
        elastic_isotropic_tensor[4][4] = 63.7;
        elastic_isotropic_tensor[5][5] = 63.7;
        const Tensor<1,21> elastic_isotropic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_isotropic_tensor);
        //std::cout << "elastic_isotropic_tensor = " << elastic_isotropic_tensor << ", elastic_isotropic_tensor norm = " << elastic_isotropic_tensor.norm() << ", elastic_isotropic_vector.norm = " << elastic_isotropic_vector.norm() << ", 1: " << 100.0*elastic_isotropic_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*elastic_isotropic_vector.norm()*elastic_isotropic_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;


        //compute_hexagonal_axes(Tensor<2,6> &elastic_matrix);
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


        const Tensor<1,21> reference_hex_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_hex_matrix);
        //std::cout << "reference_hex_vector = " << reference_hex_vector << ", reference_hex_matrix norm = " << reference_hex_matrix.norm() << ", reference_hex_vector.norm = " << reference_hex_vector.norm() << ", 1: " << 100.0*reference_hex_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_hex_vector.norm()*reference_hex_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;

        //compute_hexagonal_axes(Tensor<2,6> &elastic_matrix);
        SymmetricTensor<2,6> reference_T_matrix;
        reference_T_matrix[0][0] = 3;
        reference_T_matrix[1][1] = 3;
        reference_T_matrix[0][1] = -3;
        reference_T_matrix[5][5] = -3;
        const Tensor<1,21> reference_T_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_T_matrix);
        //std::cout << "reference_T_vector = " << reference_T_vector << ", reference_T_matrix norm = " << reference_T_matrix.norm() << ", reference_T_vector.norm = " << reference_T_vector.norm() << ", 1: " << 100.0*reference_T_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_T_vector.norm()*reference_T_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;


        SymmetricTensor<2,6> reference_O_matrix;
        reference_O_matrix[0][0] = 16;
        reference_O_matrix[1][1] = -16;
        reference_O_matrix[1][2] = -2;
        reference_O_matrix[0][2] = 2;
        reference_O_matrix[3][3] = -1;
        reference_O_matrix[4][4] = 1;
        const Tensor<1,21> reference_O_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_O_matrix);
        //std::cout << "reference_O_vector = " << reference_O_vector << ", reference_O_matrix norm = " << reference_O_matrix.norm() << ", reference_hex_vector.norm = " << reference_O_vector.norm() << ", 1: " << 100.0*reference_O_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_O_vector.norm()*reference_O_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;



        //std::vector<Tensor<2,3> > weighted_olivine_a_matrices = random_draw_volume_weighting(volume_fractions_olivine, a_cosine_matrices_olivine, n_samples);
        //std::vector<Tensor<2,3> > weighted_enstatite_a_matrices = random_draw_volume_weighting(volume_fractions_enstatite, a_cosine_matrices_enstatite, n_samples);
        /*std::cout << "new weighted_olivine_a_matrices:" << std::endl;
        for (size_t grain_i = 0; grain_i < weighted_enstatite_a_matrices.size(); grain_i++)
        {std::cout << "grain = " << grain_i << std::endl;
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                std::cout << weighted_olivine_a_matrices[grain_i][i][j] << " ";
            }
            std::cout << std::endl;
        }
        }*/
        /*
                std::array<std::array<double,3>,3> hexagonal_axes = compute_hexagonal_axes(elastic_matrix);

                // olivine
                for (unsigned int i = 0; i < 3; i++)
                  for (unsigned int j = 0; j < 3; j++)
                    data.push_back(hexagonal_axes[i][j]);
        */
      }

      template <int dim>
      void
      LpoHexagonalAxes<dim>::update_one_particle_property(const unsigned int data_position,
                                                          const Point<dim> &position,
                                                          const Vector<double> &solution,
                                                          const std::vector<Tensor<1,dim> > &gradients,
                                                          const ArrayView<double> &data) const
      {

        //const Particle::Property::Manager<dim> &manager = this->get_particle_world().get_property_manager();
        //auto id_index = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("id"));
        //std::cout << "data_position = " << data_position << ", lpo_elastic_matrix_data_position = " << lpo_elastic_matrix_data_position << std::endl;
        //std::cout << "data[0] = " << data[0] << ", data[1] = " << data[1] << ", data[data_position] = " << data[data_position] << ", data_position[lpo_elastic_tensor_data_position] = " << data[lpo_elastic_tensor_data_position] << std::endl;
        //std::cout << "position = " << position << std::endl;
        SymmetricTensor<2,6> elastic_matrix;


        //std::cout << "lpo_data_position = " << lpo_data_position << ", n_grains = " << n_grains << std::endl;
        /*Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         n_grains,
                                                         water_content,
                                                         volume_fraction_olivine,
                                                         volume_fractions_olivine,
                                                         a_cosine_matrices_olivine,
                                                         volume_fractions_enstatite,
                                                         a_cosine_matrices_enstatite);*/

        Particle::Property::LpoElasticTensor<dim>::load_particle_data(lpo_elastic_tensor_data_position,
                                                                      data,
                                                                      elastic_matrix);



        /*elastic_matrix[0][0] = 192.;
        elastic_matrix[0][1] = 66.;
        elastic_matrix[0][2] = 60.;
        elastic_matrix[1][1] = 160.;
        elastic_matrix[1][2] = 56.;
        elastic_matrix[2][2] = 272.;
        elastic_matrix[3][3] = 60.;
        elastic_matrix[4][4] = 62.;
        elastic_matrix[5][5] = 49.;*/


        elastic_matrix[0][0] = 1769.50;
        elastic_matrix[0][1] = 873.50;
        elastic_matrix[0][2] = 838.22;
        elastic_matrix[0][3] = -17.68;
        elastic_matrix[0][4] = -110.32;
        elastic_matrix[0][5] = 144.92;
        elastic_matrix[1][0] = 873.50;
        elastic_matrix[1][1] = 1846.64;
        elastic_matrix[1][2] = 836.66;
        elastic_matrix[1][3] = -37.60;
        elastic_matrix[1][4] = -32.32;
        elastic_matrix[1][5] = 153.80;
        elastic_matrix[2][0] = 838.22;
        elastic_matrix[2][1] = 836.66;
        elastic_matrix[2][2] = 1603.28;
        elastic_matrix[2][3] = -29.68;
        elastic_matrix[2][4] = -93.52;
        elastic_matrix[2][5] = 22.40;
        elastic_matrix[3][0] = -17.68;
        elastic_matrix[3][1] = -37.60;
        elastic_matrix[3][2] = -29.68;
        elastic_matrix[3][3] = 438.77;
        elastic_matrix[3][4] = 57.68;
        elastic_matrix[3][5] = -50.50;
        elastic_matrix[4][0] = -110.32;
        elastic_matrix[4][1] = -32.32;
        elastic_matrix[4][2] = -93.52;
        elastic_matrix[4][3] = 57.68;
        elastic_matrix[4][4] = 439.79;
        elastic_matrix[4][5] = -34.78;
        elastic_matrix[5][0] = 144.92;
        elastic_matrix[5][1] = 153.80;
        elastic_matrix[5][2] = 22.40;
        elastic_matrix[5][3] = -50.50;
        elastic_matrix[5][4] = -34.78;
        elastic_matrix[5][5] = 501.80;

        std::cout << "elastic_matrix = " << std::endl;
        for (size_t i = 0; i < 6; i++)
          {
            for (size_t j = 0; j < 6; j++)
              {
                std::cout << elastic_matrix[i][j] << " ";
              }
            std::cout << std::endl;
          }
        std::cout << std::endl;


        elastic_matrix = elastic_matrix * (1./81.);

        // These eigenvectors uniquely define the symmetry cartesian coordiante system (SCCS),
        // but we need to find the order in which

        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix);

        double elastic_vector_norm = elastic_vector.norm();
        //std::cout << "elastic tensor norm = " << elastic_vector_norm << std::endl;
        const SymmetricTensor<2,3> dilatation_stiffness_tensor = compute_dilatation_stiffness_tensor(elastic_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor = compute_voigt_stiffness_tensor(elastic_matrix);


        std::cout << "voigt_stiffness_tensor = " << voigt_stiffness_tensor << std::endl;
        std::cout << "dilatation_stiffness_tensor = " << dilatation_stiffness_tensor*9 << std::endl;

        const std::pair<double,double> bulk_and_shear_moduli = compute_bulk_and_shear_moduli(dilatation_stiffness_tensor, voigt_stiffness_tensor);
        const double bulk_modulus = bulk_and_shear_moduli.first;
        const double shear_modulus = bulk_and_shear_moduli.second;


        //std::cout << std::endl << "bulk_modulus = " << bulk_modulus << ", shear_modulus = " << shear_modulus << std::endl;

        const Tensor<1,9> elastic_isotropic_approximation = compute_isotropic_approximation(bulk_modulus, shear_modulus);

        //std::cout << "elastic_isotropic_approximation = " << elastic_isotropic_approximation.norm() << std::endl;
        Tensor<1,21> anisotropic_elastic_vector = elastic_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            anisotropic_elastic_vector[i] -= elastic_isotropic_approximation[i];
          }
        // ANIS
        const double elastic_anisotropic_approximation = anisotropic_elastic_vector.norm();

        std::cout << " -- " << std::endl;
        Tensor<2,3> unpermutated_SCC = compute_unpermutated_SCC(dilatation_stiffness_tensor, voigt_stiffness_tensor);

        // This return the minimal hexagonal projected elastic matrix as a tensor<2,6> and the corresponding SCC as a Tensor<2,3>
        std::pair<SymmetricTensor<2,6>,Tensor<2,3> > elastic_minimum_hexagonal_projection = compute_minimum_hexagonal_projection(unpermutated_SCC, elastic_matrix, elastic_vector_norm);

        const Tensor<1,21> elastic_minimum_hexagonal_projection_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_minimum_hexagonal_projection.first);

        const Tensor<1,9> hexagonal_elastic_vector = project_onto_hexagonal_symmetry(elastic_minimum_hexagonal_projection_vector);

        std::cout << " -- " << std::endl;
        Tensor<1,21> hexagonal_elastic_vector_residual = elastic_minimum_hexagonal_projection_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            hexagonal_elastic_vector_residual[i] -= hexagonal_elastic_vector[i];
          }

        // DC5
        const double hexagonal_elastic_residual = hexagonal_elastic_vector_residual.norm();

        //std::cout << "hexagonal_elastic_vector = " << hexagonal_elastic_vector << std::endl;
        //std::cout << "elastic_minimum_hexagonal_projection_vector = " << elastic_minimum_hexagonal_projection_vector << ", (" << elastic_minimum_hexagonal_projection_vector.norm() << ")" << std::endl;
        //std::cout << "elastic_minimum_hexagonal_projection.first = " << elastic_minimum_hexagonal_projection.first << std::endl;


        //std::cout << "hexagonal_elastic_residual = " << hexagonal_elastic_residual << std::endl;




        //{
        const SymmetricTensor<2,3> dilatation_stiffness_tensor_projected = compute_dilatation_stiffness_tensor(elastic_minimum_hexagonal_projection.first);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_projected = compute_voigt_stiffness_tensor(elastic_minimum_hexagonal_projection.first);

        std::cout << "dilatation_stiffness_tensor: " << dilatation_stiffness_tensor << ", " << dilatation_stiffness_tensor_projected << std::endl;
        std::cout << "voigt_stiffness_tensor: " << voigt_stiffness_tensor << ", " << voigt_stiffness_tensor_projected << std::endl;


        Tensor<2,3> hex_projected_SCC = compute_unpermutated_SCC(dilatation_stiffness_tensor_projected, voigt_stiffness_tensor_projected);


        /**
         * compute hexagonal formed tensor
         */
        Tensor<1,21> elastic_minimum_hexagonal_projection_vector_2;

        const Tensor<1,9> hexagonal_elastic_vector_2 = project_onto_hexagonal_symmetry(elastic_minimum_hexagonal_projection_vector);

        for (size_t i = 0; i < 9; i++)
          {
            elastic_minimum_hexagonal_projection_vector_2[i] += hexagonal_elastic_vector_2[i];
          }

        auto elastic_tensor_hexagonal_projection_2 = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(elastic_minimum_hexagonal_projection_vector_2);



        std::pair<SymmetricTensor<2,6>,Tensor<2,3> > elastic_minimum_hexagonal_projection_2 = compute_minimum_hexagonal_projection(hex_projected_SCC, elastic_tensor_hexagonal_projection_2, elastic_tensor_hexagonal_projection_2.norm());

        std::cout << "elastic_minimum_hexagonal_projection 1: " << std::endl;
        for (size_t i = 0; i < 6; i++)
          {
            for (size_t j = 0; j < 6; j++)
              {
                std::cout << elastic_minimum_hexagonal_projection.first[i][j] << " ";
              }
            std::cout << std::endl;

          }
        std::cout << std::endl;
        std::cout << "elastic_minimum_hexagonal_projection 2: "  << std::endl;
        for (size_t i = 0; i < 6; i++)
          {
            for (size_t j = 0; j < 6; j++)
              {
                std::cout << elastic_minimum_hexagonal_projection_2.first[i][j] << " ";
              }
            std::cout << std::endl;

          }
        std::cout << std::endl;
        const std::pair<double,double> bulk_and_shear_moduli_projected = compute_bulk_and_shear_moduli(dilatation_stiffness_tensor, voigt_stiffness_tensor);
        const double bulk_modulus_projected = bulk_and_shear_moduli.first;
        const double shear_modulus_projected = bulk_and_shear_moduli.second;


        //std::cout << std::endl << "bulk_modulus_projected = " << bulk_modulus_projected << ", shear_modulus_projected = " << shear_modulus_projected << std::endl;

        const Tensor<1,9> elastic_isotropic_approximation_projected = compute_isotropic_approximation(bulk_modulus_projected, shear_modulus_projected);
        //std::cout << "elastic_isotropic_approximation = " << elastic_isotropic_approximation_projected.norm() << std::endl;

        Tensor<1,21> anisotropic_elastic_vector_projected = elastic_minimum_hexagonal_projection_vector;
        // now compute how much is left over in the origional elastic vector
        for (size_t i = 0; i < 9; i++)
          {
            anisotropic_elastic_vector_projected[i] -= elastic_isotropic_approximation_projected[i];
          }
        // ANIS
        const double elastic_anisotropic_approximation_vector_projected = anisotropic_elastic_vector_projected.norm();

        const double hex_percentage = ((elastic_anisotropic_approximation_vector_projected - hexagonal_elastic_residual)/elastic_minimum_hexagonal_projection_vector.norm())*100.;

        //std::cout << "elastic_anisotropic_approximation_vector_projected = " << elastic_anisotropic_approximation_vector_projected << ", hexagonal_elastic_residual = " << hexagonal_elastic_residual << ", elastic_minimum_hexagonal_projection_vector.norm() = " << elastic_minimum_hexagonal_projection_vector.norm() << std::endl;
        //}

        // percentage = (ANIS-DC5)/XN*100
        // This is not the same equation as in Browaeys and Chevrot gfi 2004, but it
        // does seem to give the same result for hexagonal:
        // 100percent = N^{-2}(X)[N^2(X_{tric})+N^2(X_{mon})+N^2(X_{ort})+N^2(X_{tet})+N^2(X_{hex})+N^2(X_{iso})],
        // where N(X) is defined as sqrt(X_i X_i).
        const double total_anis_percentage = ((elastic_anisotropic_approximation)/elastic_vector_norm)*100.;

        const double percentage = ((elastic_anisotropic_approximation - hexagonal_elastic_residual)/elastic_vector_norm)*100.;
        //const double percentage = ((elastic_isotropic_approximation.norm() - hexagonal_elastic_residual)/elastic_vector_norm)*100;
        //const double percentage2 = (elastic_anisotropic_approximation*elastic_anisotropic_approximation - hexagonal_elastic_residual*hexagonal_elastic_residual)/(elastic_vector_norm*elastic_vector_norm)*100;
        const double percentage2 = (elastic_isotropic_approximation.norm()*elastic_isotropic_approximation.norm())/(elastic_vector_norm*elastic_vector_norm)*100;
        //std::cout << "--> new_percentage = " << hex_percentage << ", total_anis_percentage = " << total_anis_percentage << ", percentage hex of anis = " << (hex_percentage/total_anis_percentage)*100.0<< ", old percentage = " << percentage << ", percentage2 = " << percentage2 << ", iso/ansi = " << elastic_anisotropic_approximation << ", hexa/DC5 = " << hexagonal_elastic_residual << ", full/XN = " << elastic_vector_norm << std::endl;
        data[data_position] = total_anis_percentage;
        data[data_position+1] = hex_percentage;

        Tensor<2,3> projected_SCC = elastic_minimum_hexagonal_projection.second;

        std::cout << "unpermutated_SCC = " << unpermutated_SCC << std::endl;
        std::cout << "  projected_SCC = " << projected_SCC << std::endl;
        std::cout << "hxprojected_SCC = " << hex_projected_SCC << std::endl;


        /**
         * Now compute my own hexagonal projection direction
         */
        /*std::cout << "Now compute my own hexagonal projection direction..." << std::endl;
        // start with elastic_vector and project it onto the hexagonal space

        Tensor<1,9> hexagonal_projection_of_elastic_tensor = project_onto_hexagonal_symmetry(elastic_vector);

        Tensor<1,21> hexagonal_projection_of_elastic_tensor_full;

        for (size_t i = 0; i < 9; i++)
        {
          hexagonal_projection_of_elastic_tensor_full[i] = hexagonal_projection_of_elastic_tensor[i];
        }


        // create elastic tensor from that
        auto elastic_matrix_hexagonal_projection_3 = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(hexagonal_projection_of_elastic_tensor_full);
        */
        /*SymmetricTensor<2,6> elastic_matrix_hexagonal_projection_3;
        elastic_matrix_hexagonal_projection_3[0][0] = (3./8.)*(elastic_matrix[0][0] + elastic_matrix[1][1]) + 0.25 *elastic_matrix[0][1] + 0.5*elastic_matrix[5][5];
        elastic_matrix_hexagonal_projection_3[1][1] = (3./8.)*(elastic_matrix[0][0] + elastic_matrix[1][1]) + 0.25 *elastic_matrix[0][1] + 0.5*elastic_matrix[5][5];
        elastic_matrix_hexagonal_projection_3[2][2] = elastic_matrix[2][2];
        elastic_matrix_hexagonal_projection_3[1][2] = 0.5*(elastic_matrix[0][2] + elastic_matrix[1][2]);
        elastic_matrix_hexagonal_projection_3[0][2] = 0.5*(elastic_matrix[0][2] + elastic_matrix[1][2]);
        elastic_matrix_hexagonal_projection_3[0][1] = (1./8.)*(elastic_matrix[0][0] + elastic_matrix[1][1]) + 0.75 *elastic_matrix[0][1] - 0.5*elastic_matrix[5][5];
        elastic_matrix_hexagonal_projection_3[3][3] = 0.5*(elastic_matrix[3][3] + elastic_matrix[4][4]);
        elastic_matrix_hexagonal_projection_3[4][4] = 0.5*(elastic_matrix[3][3] + elastic_matrix[4][4]);
        elastic_matrix_hexagonal_projection_3[5][5] = 0.5*(elastic_matrix_hexagonal_projection_3[0][0] + elastic_matrix_hexagonal_projection_3[0][1]);*/


        // compute SCC from this tensor

        //const SymmetricTensor<2,3> dilatation_stiffness_tensor_projected_3 = compute_dilatation_stiffness_tensor(elastic_matrix_hexagonal_projection_3);
        //const SymmetricTensor<2,3> voigt_stiffness_tensor_projected_3 = compute_voigt_stiffness_tensor(elastic_matrix_hexagonal_projection_3);

        //Tensor<2,3> hex_projected_SCC_3 = compute_unpermutated_SCC(dilatation_stiffness_tensor_projected_3, voigt_stiffness_tensor_projected_3);


        //std::cout << "hex_projected_SCC_3 = " << hex_projected_SCC_3 << std::endl;


        // poging 4: use bowaeys projection matrices
        /*        SymmetricTensor<2,21> projection_matrix_tric_to_mono;
        projection_matrix_tric_to_mono[0][0] = 1.0;
        projection_matrix_tric_to_mono[1][1] = 1.0;
        projection_matrix_tric_to_mono[2][2] = 1.0;
        projection_matrix_tric_to_mono[3][3] = 1.0;
        projection_matrix_tric_to_mono[4][4] = 1.0;
        projection_matrix_tric_to_mono[5][5] = 1.0;
        projection_matrix_tric_to_mono[6][6] = 1.0;
        projection_matrix_tric_to_mono[7][7] = 1.0;
        projection_matrix_tric_to_mono[8][8] = 1.0;
        projection_matrix_tric_to_mono[11][11] = 1.0;
        projection_matrix_tric_to_mono[14][14] = 1.0;
        projection_matrix_tric_to_mono[17][17] = 1.0;
        projection_matrix_tric_to_mono[20][20] = 1.0;


        SymmetricTensor<2,21> projection_matrix_mono_to_ortho;
        projection_matrix_mono_to_ortho[0][0] = 1.0;
        projection_matrix_mono_to_ortho[1][1] = 1.0;
        projection_matrix_mono_to_ortho[2][2] = 1.0;
        projection_matrix_mono_to_ortho[3][3] = 1.0;
        projection_matrix_mono_to_ortho[4][4] = 1.0;
        projection_matrix_mono_to_ortho[5][5] = 1.0;
        projection_matrix_mono_to_ortho[6][6] = 1.0;
        projection_matrix_mono_to_ortho[7][7] = 1.0;
        projection_matrix_mono_to_ortho[8][8] = 1.0;

        SymmetricTensor<2,21> projection_matrix_ortho_to_tetra;
        projection_matrix_ortho_to_tetra[0][0] = 0.5;
        projection_matrix_ortho_to_tetra[0][1] = 0.5;
        projection_matrix_ortho_to_tetra[1][1] = 0.5;
        projection_matrix_ortho_to_tetra[2][2] = 1.0;
        projection_matrix_ortho_to_tetra[3][3] = 0.5;
        projection_matrix_ortho_to_tetra[3][4] = 0.5;
        projection_matrix_ortho_to_tetra[4][4] = 0.5;
        projection_matrix_ortho_to_tetra[5][5] = 1.0;
        projection_matrix_ortho_to_tetra[6][6] = 0.5;
        projection_matrix_ortho_to_tetra[6][7] = 0.5;
        projection_matrix_ortho_to_tetra[7][7] = 0.5;
        projection_matrix_ortho_to_tetra[8][8] = 1.0;

        SymmetricTensor<2,21> projection_matrix_tetra_to_hexa;
        projection_matrix_tetra_to_hexa[0][0] = 3./8.;
        projection_matrix_tetra_to_hexa[0][1] = 3./8.;
        projection_matrix_tetra_to_hexa[1][1] = 3./8.;
        projection_matrix_tetra_to_hexa[2][2] = 1.0;
        projection_matrix_tetra_to_hexa[3][3] = 0.5;
        projection_matrix_tetra_to_hexa[3][4] = 0.5;
        projection_matrix_tetra_to_hexa[4][4] = 0.5;
        projection_matrix_tetra_to_hexa[5][5] = 3./4.;
        projection_matrix_tetra_to_hexa[6][6] = 0.5;
        projection_matrix_tetra_to_hexa[6][7] = 0.5;
        projection_matrix_tetra_to_hexa[7][7] = 0.5;
        projection_matrix_tetra_to_hexa[8][8] = 0.5;
        projection_matrix_tetra_to_hexa[5][0] = 1./(4.*std::sqrt(2.0));
        projection_matrix_tetra_to_hexa[5][1] = 1./(4.*std::sqrt(2.0));
        projection_matrix_tetra_to_hexa[8][0] = 0.25;
        projection_matrix_tetra_to_hexa[8][1] = 0.25;
        projection_matrix_tetra_to_hexa[8][5] = -1/(2*std::sqrt(2.0));


        SymmetricTensor<2,21> projection_matrix_hexa_to_iso;
        projection_matrix_hexa_to_iso[0][0] = 3./15.;
        projection_matrix_hexa_to_iso[0][1] = 3./15.;
        projection_matrix_hexa_to_iso[0][2] = 3./15.;
        projection_matrix_hexa_to_iso[1][1] = 3./15.;
        projection_matrix_hexa_to_iso[1][2] = 3./15.;
        projection_matrix_hexa_to_iso[2][2] = 3./15.;
        projection_matrix_hexa_to_iso[3][3] = 4./15.;
        projection_matrix_hexa_to_iso[3][4] = 4./15.;
        projection_matrix_hexa_to_iso[3][5] = 4./15.;
        projection_matrix_hexa_to_iso[4][4] = 4./15.;
        projection_matrix_hexa_to_iso[4][5] = 4./15.;
        projection_matrix_hexa_to_iso[5][5] = 4./15.;
        projection_matrix_hexa_to_iso[6][6] = 1./5.;
        projection_matrix_hexa_to_iso[6][7] = 1./5.;
        projection_matrix_hexa_to_iso[6][8] = 1./5.;
        projection_matrix_hexa_to_iso[7][7] = 1./5.;
        projection_matrix_hexa_to_iso[7][8] = 1./5.;
        projection_matrix_hexa_to_iso[8][8] = 1./5.;

        projection_matrix_hexa_to_iso[0][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][6] = 2./15.;
        projection_matrix_hexa_to_iso[0][7] = 2./15.;
        projection_matrix_hexa_to_iso[0][8] = 2./15.;
        projection_matrix_hexa_to_iso[1][6] = 2./15.;
        projection_matrix_hexa_to_iso[1][7] = 2./15.;
        projection_matrix_hexa_to_iso[1][8] = 2./15.;
        projection_matrix_hexa_to_iso[2][6] = 2./15.;
        projection_matrix_hexa_to_iso[2][7] = 2./15.;
        projection_matrix_hexa_to_iso[2][8] = 2./15.;
        projection_matrix_hexa_to_iso[3][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[3][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[3][8] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][8] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][8] = -std::sqrt(2.0)/15.;
        */
        SymmetricTensor<2,6> full_elastic_matrix; //elastic_matrix
        full_elastic_matrix[0][0] = 192;//225;//192.;
        full_elastic_matrix[0][1] = 66;//54;//66.;
        full_elastic_matrix[0][2] = 60;//72;//60.;
        full_elastic_matrix[1][1] = 160;//214;//160.;
        full_elastic_matrix[1][2] = 56;//53;//56.;
        full_elastic_matrix[2][2] = 272;//178;//272.;
        full_elastic_matrix[3][3] = 60;//78;//60.;
        full_elastic_matrix[4][4] = 62;//82;//62.;
        full_elastic_matrix[5][5] = 49;//76;//49.;

        //full_elastic_matrix[0][0] = 225;//192.;
        //full_elastic_matrix[0][1] = 54;//66.;
        //full_elastic_matrix[0][2] = 72;//60.;
        //full_elastic_matrix[1][1] = 214;//160.;
        //full_elastic_matrix[1][2] = 53;//56.;
        //full_elastic_matrix[2][2] = 178;//272.;
        //full_elastic_matrix[3][3] = 78;//60.;
        //full_elastic_matrix[4][4] = 82;//62.;
        //full_elastic_matrix[5][5] = 76;//49.;


        full_elastic_matrix[0][0] = 192;//225;//192.;
        full_elastic_matrix[0][1] = 66;//54;//66.;
        full_elastic_matrix[0][2] = 56;//72;//60.;
        full_elastic_matrix[1][1] = 192;//214;//160.;
        full_elastic_matrix[1][2] = 56;//53;//56.;
        full_elastic_matrix[2][2] = 272;//178;//272.;
        full_elastic_matrix[3][3] = 60;//78;//60.;
        full_elastic_matrix[4][4] = 60;//82;//62.;
        full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);//76;//49.;

        // from solimechanics.org (Be)
        full_elastic_matrix[0][0] = 192.3;//225;//192.;
        full_elastic_matrix[0][1] = 26.7;//54;//66.;
        full_elastic_matrix[0][2] = 14;//72;//60.;
        full_elastic_matrix[1][1] = full_elastic_matrix[0][0];
        full_elastic_matrix[1][2] = full_elastic_matrix[0][2];//53;//56.;
        full_elastic_matrix[2][2] = 336.4;//178;//272.;
        full_elastic_matrix[3][3] = 162.5;//78;//60.;
        full_elastic_matrix[4][4] = full_elastic_matrix[3][3];//82;//62.;
        full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);//76;//49.;

        //full_elastic_matrix = elastic_matrix;

        std::cout << "poging 4: " << std::endl;
        const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = compute_dilatation_stiffness_tensor(full_elastic_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_full = compute_voigt_stiffness_tensor(full_elastic_matrix);
        Tensor<2,3> SCC_full = compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);


        std::pair<SymmetricTensor<2,6>,Tensor<2,3> > elastic_minimum_hexagonal_projection_4 = compute_minimum_hexagonal_projection(SCC_full, full_elastic_matrix, full_elastic_matrix.norm());

        auto SCC_full_rot = elastic_minimum_hexagonal_projection_4.second;
        std::cout << "SCC_full = " << SCC_full << ", SCC_full_rot = " << elastic_minimum_hexagonal_projection_4.second << std::endl;

        auto full_elastic_matrix_rot = LpoElasticTensor<dim>::rotate_6x6_matrix(full_elastic_matrix,SCC_full_rot);

        std::cout << "full_elastic_matrix_rot = " << full_elastic_matrix_rot<< std::endl;

        const Tensor<1,21> full_elastic_vector_rot = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(full_elastic_matrix_rot);

        //std::cout << "projection_matrix_tetra_to_hexa= " << std::endl;
        for (size_t i = 0; i < 9; i++)
          {
            for (size_t j = 0; j < 9; j++)
              {
                //std::cout << projection_matrix_hexa_to_iso[i][j] << " ";
              }
            //std::cout << std::endl;
          }
        //std::cout << std::endl;

        auto full_norm_square = full_elastic_vector_rot.norm_square();
        //auto full_norm = full_elastic_vector_rot.norm();
        //auto full_norm_matrix = full_elastic_matrix.norm();
        //auto full_norm_square_matrix =  full_elastic_matrix.norm()* full_elastic_matrix.norm();
        std::cout << " ===> full = " << full_elastic_matrix << ", norm = " << full_elastic_vector_rot.norm() << ", full_square_norm = " << full_norm_square << std::endl;
        auto mono_and_higher_vector = projection_matrix_tric_to_mono*full_elastic_vector_rot;
        auto tric_vector = full_elastic_vector_rot-mono_and_higher_vector;
        auto tric = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tric_vector);
        std::cout << " ===> tric = " << tric << ", norm = " << tric_vector.norm() << ", perc = " << (tric_vector.norm_square()/full_norm_square)*100 << std::endl;

        auto ortho_and_higher_vector = projection_matrix_mono_to_ortho*mono_and_higher_vector;
        auto mono_vector = mono_and_higher_vector-ortho_and_higher_vector;
        auto mono = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(mono_vector);
        std::cout << " ===> mono = " << mono << ", norm = " << mono_vector.norm() << ", perc = " << (mono_vector.norm_square()/full_norm_square)*100 << std::endl;


        //std::cout << "ortho_and_higher_vector = " << ortho_and_higher_vector << std::endl;
        //std::cout << "ortho_and_higher_matrix = " << LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(ortho_and_higher_vector) << std::endl;
        auto tetra_and_higher_vector = projection_matrix_ortho_to_tetra*ortho_and_higher_vector;
        auto ortho_vector = ortho_and_higher_vector-tetra_and_higher_vector;
        auto ortho = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(ortho_vector);
        std::cout << " ===> ortho = " << ortho << ", norm = " << ortho_vector.norm() << ", perc = " << (ortho_vector.norm_square()/full_norm_square)*100 << std::endl;
        //std::cout << " ===> ortho = " <<  ortho << ", tetra_and_higher_vector = " << tetra_and_higher_vector << ", norm = " << tetra_and_higher_vector.norm() << std::endl;

        //std::cout << ortho_and_higher_vector
        //std::cout << "projection_matrix_tetra_to_hexa*ortho_and_higher_vector = " << projection_matrix_tetra_to_hexa*ortho_and_higher_vector << ", norm = " << (projection_matrix_tetra_to_hexa*ortho_and_higher_vector).norm() << std::endl;
        //std::cout << "projection_matrix_tetra_to_hexa*tetra_and_higher_vector -> matrix = " << LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(projection_matrix_tetra_to_hexa*tetra_and_higher_vector) << std::endl;

        auto hexa_and_higher_vector = projection_matrix_tetra_to_hexa*tetra_and_higher_vector;
        auto tetra_vector = tetra_and_higher_vector-hexa_and_higher_vector;
        auto tetra = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_vector);
        std::cout << " hexa_and_higher_vector = " << hexa_and_higher_vector << ", tetra_vector = " << tetra_vector << ", tetra_and_higher_vector = " << tetra_and_higher_vector << std::endl;
        std::cout << " ===> tetra = " << tetra << ", norm = " << tetra_vector.norm() << ", perc = " << (tetra_vector.norm_square()/full_norm_square)*100 << std::endl;//LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_and_higher_vector-projection_matrix_tetra_to_hexa*tetra_and_higher_vector) << std::endl;


        auto iso_vector = projection_matrix_hexa_to_iso*hexa_and_higher_vector;
        auto hexa_vector = hexa_and_higher_vector-iso_vector;
        auto hexa = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(hexa_vector);
        auto iso = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(iso_vector);
        std::cout << " ===> hexa = " << hexa << ", norm = " << hexa_vector.norm() << ", perc = " << (hexa_vector.norm_square()/full_norm_square)*100  << std::endl;
        std::cout << " ===> iso = " << iso << ", norm = " << iso_vector.norm() << ", perc = " << (iso_vector.norm_square()/full_norm_square)*100 << std::endl;

        std::cout << "inv perc ani = " << ((full_elastic_vector_rot - iso_vector).norm_square()/full_norm_square)*100. << std::endl;//<< ", sqrt prec = " <<  ((full_elastic_vector_rot - iso_vector).norm()/full_norm)*100.<<std::endl;

        auto total_anisotropic = hexa_vector.norm_square() + tetra_vector.norm_square() + ortho_vector.norm_square() + mono_vector.norm_square() + tric_vector.norm_square();
        std::cout << "%of anisotropic: hexa =  " << (hexa_vector.norm_square()/total_anisotropic)*100. << ", tetra = " << (tetra_vector.norm_square()/total_anisotropic)*100.
                  << ", ortho = " << (ortho_vector.norm_square()/total_anisotropic)*100 << ", mono = " << (mono_vector.norm_square()/total_anisotropic)*100 << ", tric = " << (tric_vector.norm_square()/total_anisotropic)*100 << std::endl;



        /*// compute SCC and rotate
        auto tetra_and_higher_matrix = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_and_higher_vector);
        const SymmetricTensor<2,3> dilatation_stiffness_tensor_tetra_and_higher = compute_dilatation_stiffness_tensor(tetra_and_higher_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_tetra_and_higher = compute_voigt_stiffness_tensor(tetra_and_higher_matrix);
        Tensor<2,3> SCC_tetra_and_higher = compute_unpermutated_SCC(dilatation_stiffness_tensor_tetra_and_higher, voigt_stiffness_tensor_tetra_and_higher);

        auto tetra_and_higher_matrix_rot = LpoElasticTensor<dim>::rotate_6x6_matrix(tetra_and_higher_matrix,SCC_tetra_and_higher);
        auto tetra_and_higher_vector_rot = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(LpoElasticTensor<dim>::rotate_6x6_matrix(tetra_and_higher_matrix,SCC_tetra_and_higher));

        auto ortho_vector_rot = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(LpoElasticTensor<dim>::rotate_6x6_matrix( ortho_and_higher_matrix,SCC_tetra_and_higher));

        auto tetra_vector = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_and_higher_vector_rot-projection_matrix_tetra_to_hexa*tetra_and_higher_vector_rot);
        std::cout << " ===> tetra = " <<  tetra_vector << ", tetra_and_higher_vector_rot= " << tetra_and_higher_vector_rot << ", ortho_vector_rot = " << ortho_vector_rot <<  std::endl;
        std::cout << " ======> projection_matrix_tetra_to_hexa*ortho_and_higher_vector= " << projection_matrix_tetra_to_hexa*ortho_and_higher_vector << ", tetra_and_higher_matrix_rot= " << tetra_and_higher_matrix_rot << std::endl;

        */


        data[data_position+2]  = hex_projected_SCC[2][0];
        data[data_position+3]  = hex_projected_SCC[2][1];
        data[data_position+4]  = hex_projected_SCC[2][2];
        data[data_position+5]  = hex_projected_SCC[1][0];
        data[data_position+6]  = hex_projected_SCC[1][1];
        data[data_position+7]  = hex_projected_SCC[1][2];
        data[data_position+8]  = hex_projected_SCC[0][0];
        data[data_position+9]  = hex_projected_SCC[0][1];
        data[data_position+10] = hex_projected_SCC[0][2];

        //                Assert(elastic_anisotropic_approximation >= hexagonal_elastic_residual,
        //ExcMessage("the hexagonal part of the anisotropy (" + std::to_string(hexagonal_elastic_residual) +
        //           ") is larger than the total amount of anisotropy (" + std::to_string(elastic_anisotropic_approximation) +
        //           ")."))
        //data.push_back(percentage);
        /*
                SymmetricTensor<2,6> elastic_isotropic_tensor;

                elastic_isotropic_tensor[0][0] = 194.7;
                elastic_isotropic_tensor[0][1] = 67.3;
                elastic_isotropic_tensor[0][2] = 67.3;
                elastic_isotropic_tensor[1][1] = 194.7;
                elastic_isotropic_tensor[1][2] = 67.3;
                elastic_isotropic_tensor[2][2] = 194.7;
                elastic_isotropic_tensor[3][3] = 63.7;
                elastic_isotropic_tensor[4][4] = 63.7;
                elastic_isotropic_tensor[5][5] = 63.7;
                const Tensor<1,21> elastic_isotropic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_isotropic_tensor);
                std::cout << "elastic_isotropic_tensor = " << elastic_isotropic_tensor << ", elastic_isotropic_tensor norm = " << elastic_isotropic_tensor.norm() << ", elastic_isotropic_vector.norm = " << elastic_isotropic_vector.norm() << ", 1: " << 100.0*elastic_isotropic_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*elastic_isotropic_vector.norm()*elastic_isotropic_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;


                //compute_hexagonal_axes(Tensor<2,6> &elastic_matrix);
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


                const Tensor<1,21> reference_hex_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_hex_matrix);
                std::cout << "reference_hex_vector = " << reference_hex_vector << ", reference_hex_matrix norm = " << reference_hex_matrix.norm() << ", reference_hex_vector.norm = " << reference_hex_vector.norm() << ", 1: " << 100.0*reference_hex_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_hex_vector.norm()*reference_hex_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;

                //compute_hexagonal_axes(Tensor<2,6> &elastic_matrix);
                SymmetricTensor<2,6> reference_T_matrix;
                reference_T_matrix[0][0] = 3;
                reference_T_matrix[1][1] = 3;
                reference_T_matrix[0][1] = -3;
                reference_T_matrix[5][5] = -3;
                const Tensor<1,21> reference_T_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_T_matrix);
                std::cout << "reference_T_vector = " << reference_T_vector << ", reference_T_matrix norm = " << reference_T_matrix.norm() << ", reference_T_vector.norm = " << reference_T_vector.norm() << ", 1: " << 100.0*reference_T_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_T_vector.norm()*reference_T_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;


                SymmetricTensor<2,6> reference_O_matrix;
                reference_O_matrix[0][0] = 16;
                reference_O_matrix[1][1] = -16;
                reference_O_matrix[1][2] = -2;
                reference_O_matrix[0][2] = 2;
                reference_O_matrix[3][3] = -1;
                reference_O_matrix[4][4] = 1;
                const Tensor<1,21> reference_O_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(reference_O_matrix);
                std::cout << "reference_O_vector = " << reference_O_vector << ", reference_O_matrix norm = " << reference_O_matrix.norm() << ", reference_hex_vector.norm = " << reference_O_vector.norm() << ", 1: " << 100.0*reference_O_vector.norm()/(elastic_vector_norm) << ", 2: " << 100.0*reference_O_vector.norm()*reference_O_vector.norm()/(elastic_vector_norm*elastic_vector_norm) << std::endl;
        */


      }

      /*        template <int dim>
        void
        LpoElasticTensor<dim>::load_particle_data(unsigned int lpo_data_position,
                                                  const ArrayView<double> &data,
                                                  double anisotropic_percentage,
                                                  double hexagonal_percentage)
        {

        }*/

      template<int dim>
      std::array<unsigned short int, 3>
      LpoHexagonalAxes<dim>::indexed_even_permutation(const unsigned short int index) const
      {
        // there are 6 permutations, but only the odd or even are needed. We use the even
        // permutation here.
        switch (index)
          {
            case 0 :
              return {0,1,2};
            case 1 :
              return {1,2,0};
            case 2:
              return {2,0,1};
            default:
              AssertThrow(false,ExcMessage("Provided index larger then 2 (" + std::to_string(index)+ ")."));
              return {0,0,0};
          }

      }

      template<int dim>
      SymmetricTensor<2,3>
      LpoHexagonalAxes<dim>::compute_voigt_stiffness_tensor(const SymmetricTensor<2,6> &elastic_matrix) const
      {
        /**
         * the Voigt stiffness tensor (see Browaeys and chevrot, 2004)
         * It defines the stress needed to cause an isotropic strain in the
         * material
         */
        SymmetricTensor<2,3> voigt_stiffness_tensor;
        voigt_stiffness_tensor[0][0]=elastic_matrix[0][0]+elastic_matrix[5][5]+elastic_matrix[4][4];
        voigt_stiffness_tensor[1][1]=elastic_matrix[5][5]+elastic_matrix[1][1]+elastic_matrix[3][3];
        voigt_stiffness_tensor[2][2]=elastic_matrix[4][4]+elastic_matrix[3][3]+elastic_matrix[2][2];
        voigt_stiffness_tensor[1][0]=elastic_matrix[0][5]+elastic_matrix[1][5]+elastic_matrix[3][4];
        voigt_stiffness_tensor[2][0]=elastic_matrix[0][4]+elastic_matrix[2][4]+elastic_matrix[3][5];
        voigt_stiffness_tensor[2][1]=elastic_matrix[1][3]+elastic_matrix[2][3]+elastic_matrix[4][5];

        return voigt_stiffness_tensor;
      }


      template<int dim>
      SymmetricTensor<2,3>
      LpoHexagonalAxes<dim>::compute_dilatation_stiffness_tensor(const SymmetricTensor<2,6> &elastic_matrix) const
      {
        /**
         * The dilatational stiffness tensor (see Browaeys and chevrot, 2004)
         * It defines the stress to cause isotropic dilatation in the material.
         */
        SymmetricTensor<2,3> dilatation_stiffness_tensor;
        for (size_t i = 0; i < 3; i++)
          {
            dilatation_stiffness_tensor[0][0]=elastic_matrix[0][i]+dilatation_stiffness_tensor[0][0];
            dilatation_stiffness_tensor[1][1]=elastic_matrix[1][i]+dilatation_stiffness_tensor[1][1];
            dilatation_stiffness_tensor[2][2]=elastic_matrix[2][i]+dilatation_stiffness_tensor[2][2];
            dilatation_stiffness_tensor[1][0]=elastic_matrix[5][i]+dilatation_stiffness_tensor[1][0];
            dilatation_stiffness_tensor[2][0]=elastic_matrix[4][i]+dilatation_stiffness_tensor[2][0];
            dilatation_stiffness_tensor[2][1]=elastic_matrix[3][i]+dilatation_stiffness_tensor[2][1];
          }
        return dilatation_stiffness_tensor;
      }

      template<int dim>
      std::pair<double,double>
      LpoHexagonalAxes<dim>::compute_bulk_and_shear_moduli(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                                           const SymmetricTensor<2,3> &voigt_stiffness_tensor) const
      {
        double bulk_modulus = 0; // K
        double shear_modulus = 0; // G

        for (size_t i = 0; i < 3; i++)
          {
            bulk_modulus = bulk_modulus + dilatation_stiffness_tensor[i][i];
            shear_modulus = shear_modulus + voigt_stiffness_tensor[i][i];
          }
        shear_modulus = (3.0*shear_modulus - bulk_modulus)/30.0;
        bulk_modulus = bulk_modulus/9.0;
        //shear_modulus = shear_modulus/10.0-(3.0*bulk_modulus/10.0);

        return std::make_pair(bulk_modulus, shear_modulus);
      }

      template<int dim>
      Tensor<1,9>
      LpoHexagonalAxes<dim>::compute_isotropic_approximation(const double bulk_modulus,
                                                             const double shear_modulus) const
      {
        Tensor<1,9> isotropic_approximation;
        isotropic_approximation[0] = bulk_modulus + 4.*shear_modulus/3.;
        isotropic_approximation[1] = isotropic_approximation[0];
        isotropic_approximation[2] = isotropic_approximation[0];
        isotropic_approximation[3] = std::sqrt(2.0) * (bulk_modulus - 2.0*shear_modulus/3.0);
        isotropic_approximation[4] = isotropic_approximation[3];
        isotropic_approximation[5] = isotropic_approximation[3];
        isotropic_approximation[6] = 2.0*shear_modulus;
        isotropic_approximation[7] = isotropic_approximation[6];
        isotropic_approximation[8] = isotropic_approximation[6];

        return isotropic_approximation;
      }



      template<int dim>
      Tensor<2,3>
      LpoHexagonalAxes<dim>::compute_unpermutated_SCC(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                                      const SymmetricTensor<2,3> &voigt_stiffness_tensor) const
      {
        // computing the eigenvector of this matrix
        const std::array<std::pair<double,Tensor<1,3,double> >, 3> voigt_eigenvectors_a = eigenvectors(voigt_stiffness_tensor, SymmetricTensorEigenvectorMethod::jacobi);
        const std::array<std::pair<double,Tensor<1,3,double> >, 3> dilatation_eigenvectors_a = eigenvectors(dilatation_stiffness_tensor, SymmetricTensorEigenvectorMethod::jacobi);

        std::cout << "voigt_eigenvalues      = " << voigt_eigenvectors_a[0].first << ", " << voigt_eigenvectors_a[1].first << ", " << voigt_eigenvectors_a[2].first << std::endl;
        std::cout << "dilatation_eigenvalues = " << dilatation_eigenvectors_a[0].first << ", " << dilatation_eigenvectors_a[1].first << ", " << dilatation_eigenvectors_a[2].first << std::endl;

        std::cout << "voigt_eigenvectors:      " << voigt_eigenvectors_a[0].second << ", " << voigt_eigenvectors_a[1].second << ", " << voigt_eigenvectors_a[2].second << std::endl;
        std::cout << "dilatation_eigenvectors: " << dilatation_eigenvectors_a[0].second << ", " << dilatation_eigenvectors_a[1].second << ", " << dilatation_eigenvectors_a[2].second << std::endl;
        std::vector<Tensor<1,3,double> > unpermutated_SCC(3);
        // averaging eigenvectors
        // the next function looks for the smallest angle
        // and returns the corresponding vecvo index for that
        // vector.
        size_t NDVC = 0;
        for (size_t i1 = 0; i1 < 3; i1++)
          {
            NDVC = 0;
            double ADVC = 10.0;
            //double SCN = 0.0;
            for (size_t i2 = 0; i2 < 3; i2++)
              {
                double SDV = dilatation_eigenvectors_a[i1].second*voigt_eigenvectors_a[i2].second;
                if (std::abs(SDV) >= 1.0)
                  SDV=std::copysign(1.0,SDV);
                double ADV = std::acos(SDV);
                if (ADV < ADVC)
                  {
                    NDVC=std::copysign(1.0, SDV)*i2;
                    ADVC = ADV;
                  }
              }

            // Adds/substracting to vecdi the vecvo with the smallest
            // angle times the i2/j index with the sign of SVD to vecdi
            // (effectively turning the eigenvector),and then nomalizing it.
            unpermutated_SCC[i1] = 0.5*(dilatation_eigenvectors_a[i1].second + (double)NDVC*voigt_eigenvectors_a[std::abs((int)NDVC)].second);
            unpermutated_SCC[i1] = unpermutated_SCC[i1]/unpermutated_SCC[i1].norm();
          }
        //std::cout << "NDVC = " << NDVC << std::endl;

        return Tensor<2,3>(
        {
          {unpermutated_SCC[0][0],unpermutated_SCC[0][1],unpermutated_SCC[0][2]},
          {unpermutated_SCC[1][0],unpermutated_SCC[1][1],unpermutated_SCC[1][2]},
          {unpermutated_SCC[2][0],unpermutated_SCC[2][1],unpermutated_SCC[2][2]}
        });
      }


      template<int dim>
      std::pair<SymmetricTensor<2,6>,Tensor<2,3> >
      LpoHexagonalAxes<dim>::compute_minimum_hexagonal_projection(
        const Tensor<2,3> &unpermutated_SCC,
        const SymmetricTensor<2,6> &elastic_matrix,
        const double elastic_vector_norm) const
      {
        /*std::cout << "unpermutated_SCC= ";
        for (size_t i = 0; i < 3; i++)
          {
            for (size_t j = 0; j < 3; j++)
              {
                std::cout << unpermutated_SCC[i][j] << " ";
              }
          }
        std::cout << std::endl;*/

        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix);
        double lowest_norm = elastic_vector_norm;//*1000.;
        //std::cout << "initial norm = " << lowest_norm << std::endl;
        unsigned short int lowest_norm_permutation = 99;//99;


        /**
         * Try the different permutations to determine what is the best hexagonal projection.
         * This is based on Browaeys and Chevrot (2004), GJI (doi: 10.1111/j.1365-246X.2004.024115.x),
         * which states at the end of paragraph 3.3 that "... an important property of an orthogonal projection
         * is that the distance between a vector $X$ and its orthogonal projection $X_H = p(X)$ on a given
         * subspace is minimum. These two features ensure that the decomposition is optimal once a 3-D Cartesian
         * coordiante systeem is chosen.". The other property they talk about is that "The space of elastic
         * vectors has a finite dimension [...], i.e. using a differnt norm from eq. (2.3 will change disstances
         * but not the resulting decomposition.".
         */
        Tensor<2,3> projected_SCC[3];
        SymmetricTensor<2,6> projected_elastic_matrix[3];
        for (unsigned short int permutation_i = 0; permutation_i < 3; permutation_i++)
          {
            std::cout << "permutation_i = " << permutation_i << std::endl;
            std::array<unsigned short int, 3> perumation = indexed_even_permutation(permutation_i);


            for (size_t j = 0; j < 3; j++)
              {
                projected_SCC[permutation_i][j] = unpermutated_SCC[perumation[j]];
              }

            projected_elastic_matrix[permutation_i] = LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_matrix,(projected_SCC[permutation_i]));

            const Tensor<1,21> projected_elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(projected_elastic_matrix[permutation_i]);

            const Tensor<1,9> hexagonal_elastic_vector = project_onto_hexagonal_symmetry(projected_elastic_vector);

            Tensor<1,21> elastic_vector_tmp = elastic_vector;// projected_elastic_vector;
            //std::cout << "CE1/elastic tensor = " << std::endl;
            for (size_t k = 0; k < 6; k++)
              {
                for (size_t l = 0; l < 6; l++)
                  {
                    //std::cout << "CED(" << k+1 << "," << l+1 << ") = " << elastic_matrix[k][l] << std::endl;
                  }

              }

            std::cout << "X/elastic_vector_tmp       = ";
            for (size_t i = 0; i < 21; i++)
              {
                std::cout << elastic_vector_tmp[i] << " ";
              }
            std::cout << std::endl;


            std::cout << "XH/hexagonal_elastic_vector = ";
            for (size_t i = 0; i < 9; i++)
              {
                std::cout << hexagonal_elastic_vector[i] << " ";
              }
            std::cout << std::endl;


            // now compute how much is left over in the origional elastic vector
            for (size_t substraction_i = 0; substraction_i < 9; substraction_i++)
              {
                elastic_vector_tmp[substraction_i] -= hexagonal_elastic_vector[substraction_i];
              }

            std::cout << "XD/elastic_vector_tmp       = ";
            for (size_t i = 0; i < 21; i++)
              {
                std::cout << elastic_vector_tmp[i] << " ";
              }
            std::cout << std::endl;


            const double current_norm = elastic_vector_tmp.norm();

            if (current_norm < lowest_norm)
              {
                lowest_norm = current_norm;
                lowest_norm_permutation = permutation_i;
              }
            std::cout << "  -> projected SCC for " << permutation_i << "= " << projected_SCC[permutation_i] << " , projected_elastic_vector = " << projected_elastic_vector << std::endl;
            std::cout << "projected_elastic_matrix = " << std::endl;
            for (size_t i = 0; i < 6; i++)
              {
                for (size_t j = 0; j < 6; j++)
                  {
                    std::cout << projected_elastic_matrix[permutation_i][i][j] << " ";
                  }
                std::cout << std::endl;
              }
            std::cout << std::endl;

            std::cout << "   => " << permutation_i << ": lowest_norm = " << lowest_norm << ", current_norm = " << current_norm << ", lowest_norm_permutation = " << lowest_norm_permutation << std::endl;

          }

        //if(lowest_norm_permutation >= 3){
        //  lowest_norm_permutation = 1;
        //}
        AssertThrow(lowest_norm_permutation < 3,
                    ExcMessage("LPO Hexagonal axes plugin could not find a good hexagonal projection: " + std::to_string(lowest_norm_permutation) + ", lowest_norm = " + std::to_string(lowest_norm)));


        return std::make_pair(projected_elastic_matrix[lowest_norm_permutation],projected_SCC[lowest_norm_permutation]);
      }


      template<int dim>
      Tensor<1,9>
      LpoHexagonalAxes<dim>::project_onto_hexagonal_symmetry(const Tensor<1,21> &elastic_vector) const
      {
        return Tensor<1,9>(
        {
          0.375 * (elastic_vector[0] + elastic_vector[1]) + elastic_vector[5]/(std::sqrt(2.0) * 4.0) + 0.25 * elastic_vector[8],                  // 0 // 1
          0.375 * (elastic_vector[0] + elastic_vector[1]) + elastic_vector[5]/(std::sqrt(2.0) * 4.0) + 0.25 * elastic_vector[8],                  // 1 // 2
          elastic_vector[2],                                                                                                                      // 2 // 3
          0.5 * (elastic_vector[3] + elastic_vector[4]),                                                                                          // 3 // 4
          0.5 * (elastic_vector[3] + elastic_vector[4]),                                                                                          // 4 // 5
          (elastic_vector[0] + elastic_vector[1])/(std::sqrt(2.0) * 4.0) + 0.75 * (elastic_vector[5]) - elastic_vector[8]/(std::sqrt(2.0) * 2.0), // 5 // 6
          0.5 * (elastic_vector[6] + elastic_vector[7]),                                                                                          // 6 // 7
          0.5 * (elastic_vector[6] + elastic_vector[7]),                                                                                          // 7 // 8
          0.25 * (elastic_vector[0] + elastic_vector[1]) - elastic_vector[5]/(std::sqrt(2.0) * 2.0) + 0.5 * elastic_vector[8]                     // 8 // 9
        });
      }

      template<int dim>
      std::vector<Tensor<2,3> >
      LpoHexagonalAxes<dim>::random_draw_volume_weighting(std::vector<double> fv,
                                                          std::vector<Tensor<2,3>> matrices,
                                                          unsigned int n_output_grains) const
      {
        // Get volume weighted euler angles, using random draws to convert odf
        // to a discrete number of orientations, weighted by volume
        // 1a. Get index that would sort volume fractions AND
        //ix = np.argsort(fv[q,:]);
        // 1b. Get the sorted volume and angle arrays
        std::vector<double> fv_to_sort = fv;
        std::vector<double> fv_sorted = fv;
        std::vector<Tensor<2,3>> matrices_sorted = matrices;

        unsigned int n_grain = fv_to_sort.size();


        /**
         * ...
         */
        for (int i = n_grain-1; i >= 0; --i)
          {
            unsigned int index_max_fv = std::distance(fv_to_sort.begin(),max_element(fv_to_sort.begin(), fv_to_sort.end()));

            fv_sorted[i] = fv_to_sort[index_max_fv];
            matrices_sorted[i] = matrices[index_max_fv];
            /*Assert(matrices[index_max_fv].size() == 3, ExcMessage("matrices vector (size = " + std::to_string(matrices[index_max_fv].size()) +
                                                                ") should have size 3."));
            Assert(matrices_sorted[i].size() == 3, ExcMessage("matrices_sorted vector (size = " + std::to_string(matrices_sorted[i].size()) +
                                                            ") should have size 3."));*/
            fv_to_sort[index_max_fv] = -1;
          }

        // 2. Get cumulative weight for volume fraction
        std::vector<double> cum_weight(n_grains);
        std::partial_sum(fv_sorted.begin(),fv_sorted.end(),cum_weight.begin());
        // 3. Generate random indices
        boost::random::uniform_real_distribution<> dist(0, 1);
        std::vector<double> idxgrain(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            idxgrain[grain_i] = dist(this->random_number_generator);
            //std::cout << ">>> rand new = " << grain_i << ": "<< idxgrain[grain_i] << std::endl;
          }

        // 4. Find the maximum cum_weight that is less than the random value.
        // the euler angle index is +1. For example, if the idxGrain(g) < cumWeight(1),
        // the index should be 1 not zero)
        std::vector<Tensor<2,3>> matrices_out(n_output_grains);
        for (unsigned int grain_i = 0; grain_i < n_output_grains; ++grain_i)
          {
            unsigned int counter = 0;
            for (unsigned int grain_j = 0; grain_j < n_grains; ++grain_j)
              {
                // find the first cummulative weight which is larger than the random number
                // todo: there are algorithms to do this faster
                if (cum_weight[grain_j] < idxgrain[grain_i])
                  {
                    counter++;
                  }
                else
                  {
                    break;
                  }


                /*Assert(matrices_sorted[counter].size() == 3, ExcMessage("matrices_sorted vector (size = " + std::to_string(matrices_sorted[counter].size()) +
                                                                      ") should have size 3."));*/

                /*Assert(matrices_out[counter].size() == 3, ExcMessage("matrices_out vector (size = " + std::to_string(matrices_out[counter].size()) +
                                                                   ") should have size 3."));*/
              }
            matrices_out[grain_i] = matrices_sorted[counter];
          }
        return matrices_out;
      }



      template <int dim>
      UpdateTimeFlags
      LpoHexagonalAxes<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      LpoHexagonalAxes<dim>::get_needed_update_flags () const
      {
        return update_default;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      LpoHexagonalAxes<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int> > property_information;

        property_information.push_back(std::make_pair("lpo elastic anisotropic percentage",1));
        property_information.push_back(std::make_pair("lpo elastic hexagonal percentage",1));
        property_information.push_back(std::make_pair("lpo elastic hexagonal axis a",3));
        property_information.push_back(std::make_pair("lpo elastic hexagonal axis b",3));
        property_information.push_back(std::make_pair("lpo elastic hexagonal axis c",3));
        //property_information.push_back(std::make_pair("lpo_hexagonal_axes average olivine b axis",3));
        //property_information.push_back(std::make_pair("lpo_hexagonal_axes average olivine c axis",3));

        //property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite a axis",3));
        //property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite b axis",3));
        //property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite c axis",3));

        return property_information;
      }

      template <int dim>
      void
      LpoHexagonalAxes<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoHexagonalAxes");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducable as long as the problem is run with the "
                                 "same amount of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");


              prm.declare_entry ("Number of grains per praticle", "50",
                                 Patterns::Integer (0),
                                 "The number of grains of olivine and the number of grain of enstatite "
                                 "each particle contains.");

              prm.declare_entry ("Mobility", "50",
                                 Patterns::Double(0),
                                 "The intrinsic grain boundary mobility for both olivine and enstatite. "
                                 "Todo: split for olivine and enstatite.");

              prm.declare_entry ("Volume fraction olivine", "0.5",
                                 Patterns::Double(0),
                                 "The volume fraction of the olivine phase (0 is no olivine, 1 is fully olivine). "
                                 "The rest of the volume fraction is set to be entstatite. "
                                 "Todo: if full olivine make not enstite grains and vice-versa.");

              prm.declare_entry ("Stress exponents", "3.5",
                                 Patterns::Double(0),
                                 "This is the power law exponent that characterizes the rheology of the "
                                 "slip systems. It is used in equation 11 of Kaminski et al., 2004. "
                                 "This is used for both olivine and enstatite. Todo: split?");

              prm.declare_entry ("Exponents p", "1.5",
                                 Patterns::Double(0),
                                 "This is exponent p as defined in equation 11 of Kaminski et al., 2004. ");

              prm.declare_entry ("Nucleation efficientcy", "5",
                                 Patterns::Double(0),
                                 "This is the dimensionless nucleation rate as defined in equation 8 of "
                                 "Kaminski et al., 2004. ");

              prm.declare_entry ("Threshold GBS", "0.3",
                                 Patterns::Double(0),
                                 "This is the grain-boundary sliding threshold. ");

              prm.declare_entry ("Number of samples", "0",
                                 Patterns::Double(0),
                                 "This determines how many samples are taken when using the random "
                                 "draw volume averaging. Setting it to zero means that the number of "
                                 "samples is set to be equal to the number of grains.");
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }


      template <int dim>
      void
      LpoHexagonalAxes<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoHexagonalAxes");
            {

              random_number_seed = prm.get_integer ("Random number seed"); // 2
              n_grains = LPO<dim>::get_number_of_grains();
              mobility = prm.get_double("Mobility"); //50;
              x_olivine = prm.get_double("Volume fraction olivine"); // 0.5;
              stress_exponent = prm.get_double("Stress exponents"); //3.5;
              exponent_p = prm.get_double("Exponents p"); //1.5;
              nucleation_efficientcy = prm.get_double("Nucleation efficientcy"); //5;
              threshold_GBS = prm.get_double("Threshold GBS"); //0.0;
              n_samples = prm.get_integer("Number of samples"); // 0
              if (n_samples == 0)
                n_samples = n_grains;
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();

        // setup projection matrices
        // projection_matrix_tric_to_mono
        projection_matrix_tric_to_mono[0][0] = 1.0;
        projection_matrix_tric_to_mono[1][1] = 1.0;
        projection_matrix_tric_to_mono[2][2] = 1.0;
        projection_matrix_tric_to_mono[3][3] = 1.0;
        projection_matrix_tric_to_mono[4][4] = 1.0;
        projection_matrix_tric_to_mono[5][5] = 1.0;
        projection_matrix_tric_to_mono[6][6] = 1.0;
        projection_matrix_tric_to_mono[7][7] = 1.0;
        projection_matrix_tric_to_mono[8][8] = 1.0;
        projection_matrix_tric_to_mono[11][11] = 1.0;
        projection_matrix_tric_to_mono[14][14] = 1.0;
        projection_matrix_tric_to_mono[17][17] = 1.0;
        projection_matrix_tric_to_mono[20][20] = 1.0;


        // projection_matrix_mono_to_ortho;
        projection_matrix_mono_to_ortho[0][0] = 1.0;
        projection_matrix_mono_to_ortho[1][1] = 1.0;
        projection_matrix_mono_to_ortho[2][2] = 1.0;
        projection_matrix_mono_to_ortho[3][3] = 1.0;
        projection_matrix_mono_to_ortho[4][4] = 1.0;
        projection_matrix_mono_to_ortho[5][5] = 1.0;
        projection_matrix_mono_to_ortho[6][6] = 1.0;
        projection_matrix_mono_to_ortho[7][7] = 1.0;
        projection_matrix_mono_to_ortho[8][8] = 1.0;

        // projection_matrix_ortho_to_tetra;
        projection_matrix_ortho_to_tetra[0][0] = 0.5;
        projection_matrix_ortho_to_tetra[0][1] = 0.5;
        projection_matrix_ortho_to_tetra[1][1] = 0.5;
        projection_matrix_ortho_to_tetra[2][2] = 1.0;
        projection_matrix_ortho_to_tetra[3][3] = 0.5;
        projection_matrix_ortho_to_tetra[3][4] = 0.5;
        projection_matrix_ortho_to_tetra[4][4] = 0.5;
        projection_matrix_ortho_to_tetra[5][5] = 1.0;
        projection_matrix_ortho_to_tetra[6][6] = 0.5;
        projection_matrix_ortho_to_tetra[6][7] = 0.5;
        projection_matrix_ortho_to_tetra[7][7] = 0.5;
        projection_matrix_ortho_to_tetra[8][8] = 1.0;

        // projection_matrix_tetra_to_hexa;
        projection_matrix_tetra_to_hexa[0][0] = 3./8.;
        projection_matrix_tetra_to_hexa[0][1] = 3./8.;
        projection_matrix_tetra_to_hexa[1][1] = 3./8.;
        projection_matrix_tetra_to_hexa[2][2] = 1.0;
        projection_matrix_tetra_to_hexa[3][3] = 0.5;
        projection_matrix_tetra_to_hexa[3][4] = 0.5;
        projection_matrix_tetra_to_hexa[4][4] = 0.5;
        projection_matrix_tetra_to_hexa[5][5] = 3./4.;
        projection_matrix_tetra_to_hexa[6][6] = 0.5;
        projection_matrix_tetra_to_hexa[6][7] = 0.5;
        projection_matrix_tetra_to_hexa[7][7] = 0.5;
        projection_matrix_tetra_to_hexa[8][8] = 0.5;
        projection_matrix_tetra_to_hexa[5][0] = 1./(4.*std::sqrt(2.0));
        projection_matrix_tetra_to_hexa[5][1] = 1./(4.*std::sqrt(2.0));
        projection_matrix_tetra_to_hexa[8][0] = 0.25;
        projection_matrix_tetra_to_hexa[8][1] = 0.25;
        projection_matrix_tetra_to_hexa[8][5] = -1/(2*std::sqrt(2.0));


        // projection_matrix_hexa_to_iso;
        projection_matrix_hexa_to_iso[0][0] = 3./15.;
        projection_matrix_hexa_to_iso[0][1] = 3./15.;
        projection_matrix_hexa_to_iso[0][2] = 3./15.;
        projection_matrix_hexa_to_iso[1][1] = 3./15.;
        projection_matrix_hexa_to_iso[1][2] = 3./15.;
        projection_matrix_hexa_to_iso[2][2] = 3./15.;
        projection_matrix_hexa_to_iso[3][3] = 4./15.;
        projection_matrix_hexa_to_iso[3][4] = 4./15.;
        projection_matrix_hexa_to_iso[3][5] = 4./15.;
        projection_matrix_hexa_to_iso[4][4] = 4./15.;
        projection_matrix_hexa_to_iso[4][5] = 4./15.;
        projection_matrix_hexa_to_iso[5][5] = 4./15.;
        projection_matrix_hexa_to_iso[6][6] = 1./5.;
        projection_matrix_hexa_to_iso[6][7] = 1./5.;
        projection_matrix_hexa_to_iso[6][8] = 1./5.;
        projection_matrix_hexa_to_iso[7][7] = 1./5.;
        projection_matrix_hexa_to_iso[7][8] = 1./5.;
        projection_matrix_hexa_to_iso[8][8] = 1./5.;

        projection_matrix_hexa_to_iso[0][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[1][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][3] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][4] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[2][5] = std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[0][6] = 2./15.;
        projection_matrix_hexa_to_iso[0][7] = 2./15.;
        projection_matrix_hexa_to_iso[0][8] = 2./15.;
        projection_matrix_hexa_to_iso[1][6] = 2./15.;
        projection_matrix_hexa_to_iso[1][7] = 2./15.;
        projection_matrix_hexa_to_iso[1][8] = 2./15.;
        projection_matrix_hexa_to_iso[2][6] = 2./15.;
        projection_matrix_hexa_to_iso[2][7] = 2./15.;
        projection_matrix_hexa_to_iso[2][8] = 2./15.;
        projection_matrix_hexa_to_iso[3][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[3][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[3][8] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[4][8] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][6] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][7] = -std::sqrt(2.0)/15.;
        projection_matrix_hexa_to_iso[5][8] = -std::sqrt(2.0)/15.;
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(LpoHexagonalAxes,
                                        "lpo hexagonal axes",
                                        "A plugin in which the particle property tensor is "
                                        "defined as the deformation gradient tensor "
                                        "$\\mathbf F$ this particle has experienced. "
                                        "$\\mathbf F$ can be polar-decomposed into the left stretching tensor "
                                        "$\\mathbf L$ (the finite strain we are interested in), and the "
                                        "rotation tensor $\\mathbf Q$. See the corresponding cookbook in "
                                        "the manual for more detailed information.")
    }
  }
}

