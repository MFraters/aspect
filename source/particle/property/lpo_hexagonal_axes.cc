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


        Particle::Property::LpoElasticTensor<dim>::load_particle_data(lpo_elastic_tensor_data_position,
                                                                      data,
                                                                      elastic_matrix);


        const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = compute_dilatation_stiffness_tensor(elastic_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_full = compute_voigt_stiffness_tensor(elastic_matrix);
        Tensor<2,3> SCC_full = compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);

        //double full_elastic_vector_norm_square = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix).norm_square();

        std::array<std::array<double,3>,7 > norms = compute_elastic_tensor_SCC_decompositions(SCC_full, elastic_matrix);

        // get max hexagonal element index, which is the same as the permutation index
        const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
        std::array<unsigned short int, 3> perumation = indexed_even_permutation(max_hexagonal_element_index);
        // reorder the SCC be the SCC permutation which yields the largest hexagonal vector (percentage of anisotropy)
        Tensor<2,3> hexa_permutated_SCC;
        for (size_t index = 0; index < 3; index++)
          {
            hexa_permutated_SCC[index] = SCC_full[perumation[index]];
          }

        data.push_back(SCC_full[0][0]);
        data.push_back(SCC_full[0][1]);
        data.push_back(SCC_full[0][2]);
        data.push_back(SCC_full[1][0]);
        data.push_back(SCC_full[1][1]);
        data.push_back(SCC_full[1][2]);
        data.push_back(SCC_full[2][0]);
        data.push_back(SCC_full[2][1]);
        data.push_back(SCC_full[2][2]);
        data.push_back(hexa_permutated_SCC[2][0]);
        data.push_back(hexa_permutated_SCC[2][1]);
        data.push_back(hexa_permutated_SCC[2][2]);
        data.push_back(norms[6][0]);
        data.push_back(norms[0][0]); // triclinic
        data.push_back(norms[0][1]); // triclinic
        data.push_back(norms[0][2]); // triclinic
        data.push_back(norms[1][0]); // monoclinic
        data.push_back(norms[1][1]); // monoclinic
        data.push_back(norms[1][2]); // monoclinic
        data.push_back(norms[2][0]); // orthorhomic
        data.push_back(norms[2][1]); // orthorhomic
        data.push_back(norms[2][2]); // orthorhomic
        data.push_back(norms[3][0]); // tetragonal
        data.push_back(norms[3][1]); // tetragonal
        data.push_back(norms[3][2]); // tetragonal
        data.push_back(norms[4][0]); // hexagonal
        data.push_back(norms[4][1]); // hexagonal
        data.push_back(norms[4][2]); // hexagonal
        data.push_back(norms[5][0]); // isotropic

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

        */
      }

      template <int dim>
      void
      LpoHexagonalAxes<dim>::update_one_particle_property(const unsigned int data_position,
                                                          const Point<dim> &,
                                                          const Vector<double> &,
                                                          const std::vector<Tensor<1,dim> > &,
                                                          const ArrayView<double> &data) const
      {
        SymmetricTensor<2,6> elastic_matrix;
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


        //elastic_matrix[0][0] = 1769.50;
        //elastic_matrix[0][1] = 873.50;
        //elastic_matrix[0][2] = 838.22;
        //elastic_matrix[0][3] = -17.68;
        //elastic_matrix[0][4] = -110.32;
        //elastic_matrix[0][5] = 144.92;
        //elastic_matrix[1][0] = 873.50;
        //elastic_matrix[1][1] = 1846.64;
        //elastic_matrix[1][2] = 836.66;
        //elastic_matrix[1][3] = -37.60;
        //elastic_matrix[1][4] = -32.32;
        //elastic_matrix[1][5] = 153.80;
        //elastic_matrix[2][0] = 838.22;
        //elastic_matrix[2][1] = 836.66;
        //elastic_matrix[2][2] = 1603.28;
        //elastic_matrix[2][3] = -29.68;
        //elastic_matrix[2][4] = -93.52;
        //elastic_matrix[2][5] = 22.40;
        //elastic_matrix[3][0] = -17.68;
        //elastic_matrix[3][1] = -37.60;
        //elastic_matrix[3][2] = -29.68;
        //elastic_matrix[3][3] = 438.77;
        //elastic_matrix[3][4] = 57.68;
        //elastic_matrix[3][5] = -50.50;
        //elastic_matrix[4][0] = -110.32;
        //elastic_matrix[4][1] = -32.32;
        //elastic_matrix[4][2] = -93.52;
        //elastic_matrix[4][3] = 57.68;
        //elastic_matrix[4][4] = 439.79;
        //elastic_matrix[4][5] = -34.78;
        //elastic_matrix[5][0] = 144.92;
        //elastic_matrix[5][1] = 153.80;
        //elastic_matrix[5][2] = 22.40;
        //elastic_matrix[5][3] = -50.50;
        //elastic_matrix[5][4] = -34.78;
        //elastic_matrix[5][5] = 501.80;

        //SymmetricTensor<2,6> full_elastic_matrix = elastic_matrix;
        //full_elastic_matrix[0][0] = 192;//225;//192.;
        //full_elastic_matrix[0][1] = 66;//54;//66.;
        //full_elastic_matrix[0][2] = 60;//72;//60.;
        //full_elastic_matrix[1][1] = 160;//214;//160.;
        //full_elastic_matrix[1][2] = 56;//53;//56.;
        //full_elastic_matrix[2][2] = 272;//178;//272.;
        //full_elastic_matrix[3][3] = 60;//78;//60.;
        //full_elastic_matrix[4][4] = 62;//82;//62.;
        //full_elastic_matrix[5][5] = 49;//76;//49.;

        //full_elastic_matrix[0][0] = 225;//192.;
        //full_elastic_matrix[0][1] = 54;//66.;
        //full_elastic_matrix[0][2] = 72;//60.;
        //full_elastic_matrix[1][1] = 214;//160.;
        //full_elastic_matrix[1][2] = 53;//56.;
        //full_elastic_matrix[2][2] = 178;//272.;
        //full_elastic_matrix[3][3] = 78;//60.;
        //full_elastic_matrix[4][4] = 82;//62.;
        //full_elastic_matrix[5][5] = 76;//49.;


        //full_elastic_matrix[0][0] = 192;//225;//192.;
        //full_elastic_matrix[0][1] = 66;//54;//66.;
        //full_elastic_matrix[0][2] = 56;//72;//60.;
        //full_elastic_matrix[1][1] = 192;//214;//160.;
        //full_elastic_matrix[1][2] = 56;//53;//56.;
        //full_elastic_matrix[2][2] = 272;//178;//272.;
        //full_elastic_matrix[3][3] = 60;//78;//60.;
        //full_elastic_matrix[4][4] = 60;//82;//62.;
        //full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);//76;//49.;

        // from solimechanics.org (Be)
        /*full_elastic_matrix[0][0] = 192.3;//225;//192.;
        full_elastic_matrix[0][1] = 26.7;//54;//66.;
        full_elastic_matrix[0][2] = 14;//72;//60.;
        full_elastic_matrix[1][1] = full_elastic_matrix[0][0];
        full_elastic_matrix[1][2] = full_elastic_matrix[0][2];//53;//56.;
        full_elastic_matrix[2][2] = 336.4;//178;//272.;
        full_elastic_matrix[3][3] = 162.5;//78;//60.;
        full_elastic_matrix[4][4] = full_elastic_matrix[3][3];//82;//62.;
        full_elastic_matrix[5][5] = 0.5*(full_elastic_matrix[0][0] + full_elastic_matrix[0][1]);//76;//49.;*/

        const SymmetricTensor<2,3> dilatation_stiffness_tensor_full = compute_dilatation_stiffness_tensor(elastic_matrix);
        const SymmetricTensor<2,3> voigt_stiffness_tensor_full = compute_voigt_stiffness_tensor(elastic_matrix);
        Tensor<2,3> SCC_full = compute_unpermutated_SCC(dilatation_stiffness_tensor_full, voigt_stiffness_tensor_full);

        //double full_elastic_vector_norm_square = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix).norm_square();
        //double full_elastic_vector_norm_square_rot = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_matrix,SCC_full)).norm_square();
        //std::cout << "full_elastic_vector_norm_square = " << full_elastic_vector_norm_square << ", rot = " << full_elastic_vector_norm_square_rot << std::endl;

        std::array<std::array<double,3>,7 > norms = compute_elastic_tensor_SCC_decompositions(SCC_full, elastic_matrix);

        // get max hexagonal element index, which is the same as the permutation index
        const size_t max_hexagonal_element_index = std::max_element(norms[4].begin(),norms[4].end())-norms[4].begin();
        std::array<unsigned short int, 3> perumation = indexed_even_permutation(max_hexagonal_element_index);
        // reorder the SCC be the SCC permutation which yields the largest hexagonal vector (percentage of anisotropy)
        Tensor<2,3> hexa_permutated_SCC;
        for (size_t index = 0; index < 3; index++)
          {
            hexa_permutated_SCC[index] = SCC_full[perumation[index]];
          }



        /*std::cout << " ------> %percentages (min-norm:perm;max-norm:perm): isotropic = " << (norms[5][0]/full_elastic_vector_norm_square)*100
                  << ", hexag = (" << (norms[4][min_max_norm_elemtents[4][0]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[4][0] << ";" << (norms[4][min_max_norm_elemtents[4][1]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[4][1] <<  ")"
                  << ", tetra = (" << (norms[3][min_max_norm_elemtents[3][0]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[3][0] << ";" << (norms[3][min_max_norm_elemtents[3][1]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[3][1] <<  ")"
                  << ", ortho = (" << (norms[2][min_max_norm_elemtents[2][0]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[2][0] << ";" << (norms[2][min_max_norm_elemtents[2][1]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[2][1] <<  ")"
                  << ", monoc = (" << (norms[1][min_max_norm_elemtents[1][0]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[1][0] << ";" << (norms[1][min_max_norm_elemtents[1][1]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[1][1] <<  ")"
                  << ", tricl = (" << (norms[0][min_max_norm_elemtents[0][0]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[0][0] << ";" << (norms[0][min_max_norm_elemtents[0][1]]/full_elastic_vector_norm_square)*100 << ":" << min_max_norm_elemtents[0][1] <<  ")"
                  << std::endl << std::endl;

        std::cout << " ------> %anisotropy = " << (anistropic_norm/full_elastic_vector_norm_square)*100. << ", (min-norm:perm;max-norm:perm):"
                  << ", hexag = (" << (norms[4][min_max_norm_elemtents[4][0]]/anistropic_norm)*100  << ";" << (norms[4][min_max_norm_elemtents[4][1]]/anistropic_norm)*100 <<  ")"
                  << ", tetra = (" << (norms[3][min_max_norm_elemtents[3][0]]/anistropic_norm)*100  << ";" << (norms[4][min_max_norm_elemtents[4][1]]/anistropic_norm)*100 <<  ")"
                  << ", ortho = (" << (norms[2][min_max_norm_elemtents[2][0]]/anistropic_norm)*100  << ";" << (norms[4][min_max_norm_elemtents[4][1]]/anistropic_norm)*100 <<  ")"
                  << ", monoc = (" << (norms[1][min_max_norm_elemtents[1][0]]/anistropic_norm)*100  << ";" << (norms[4][min_max_norm_elemtents[4][1]]/anistropic_norm)*100 <<  ")"
                  << ", tricl = (" << (norms[0][min_max_norm_elemtents[0][0]]/anistropic_norm)*100  << ";" << (norms[4][min_max_norm_elemtents[4][1]]/anistropic_norm)*100 <<  ")"
                  << std::endl;*/

        data[data_position]    = SCC_full[0][0];
        data[data_position+1]  = SCC_full[0][1];
        data[data_position+2]  = SCC_full[0][2];
        data[data_position+3]  = SCC_full[1][0];
        data[data_position+4]  = SCC_full[1][1];
        data[data_position+5]  = SCC_full[1][2];
        data[data_position+6]  = SCC_full[2][0];
        data[data_position+7]  = SCC_full[2][1];
        data[data_position+8]  = SCC_full[2][2];
        data[data_position+9]  = hexa_permutated_SCC[2][0];
        data[data_position+10] = hexa_permutated_SCC[2][1];
        data[data_position+11] = hexa_permutated_SCC[2][2];
        data[data_position+12] = norms[6][0];
        data[data_position+13] = norms[0][0]; // triclinic
        data[data_position+14] = norms[0][1]; // triclinic
        data[data_position+15] = norms[0][2]; // triclinic
        data[data_position+16] = norms[1][0]; // monoclinic
        data[data_position+17] = norms[1][1]; // monoclinic
        data[data_position+18] = norms[1][2]; // monoclinic
        data[data_position+19] = norms[2][0]; // orthorhomic
        data[data_position+20] = norms[2][1]; // orthorhomic
        data[data_position+21] = norms[2][2]; // orthorhomic
        data[data_position+22] = norms[3][0]; // tetragonal
        data[data_position+23] = norms[3][1]; // tetragonal
        data[data_position+24] = norms[3][2]; // tetragonal
        data[data_position+25] = norms[4][0]; // hexagonal
        data[data_position+26] = norms[4][1]; // hexagonal
        data[data_position+27] = norms[4][2]; // hexagonal
        data[data_position+28] = norms[5][0]; // isotropic


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
            /*case 3:
              return {0,2,1};
            case 4 :
              return {1,0,2};
            case 5:
              return {2,1,0};*/
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
      std::array<std::array<double,3>,7>
      LpoHexagonalAxes<dim>::compute_elastic_tensor_SCC_decompositions(
        const Tensor<2,3> &unpermutated_SCC,
        const SymmetricTensor<2,6> &elastic_matrix) const
      {
        /**
         * TODO: this is from the minimum hexagonal function as used in D-Rex, see if this is
         * still usefull for what I am doing here.
         * Try the different permutations to determine what is the best hexagonal projection.
         * This is based on Browaeys and Chevrot (2004), GJI (doi: 10.1111/j.1365-246X.2004.024115.x),
         * which states at the end of paragraph 3.3 that "... an important property of an orthogonal projection
         * is that the distance between a vector $X$ and its orthogonal projection $X_H = p(X)$ on a given
         * subspace is minimum. These two features ensure that the decomposition is optimal once a 3-D Cartesian
         * coordiante systeem is chosen.". The other property they talk about is that "The space of elastic
         * vectors has a finite dimension [...], i.e. using a differnt norm from eq. (2.3 will change disstances
         * but not the resulting decomposition.".
         */
        Tensor<2,3> permutated[3];
        SymmetricTensor<2,6> rotated_elastic_matrix[3];
        std::array<std::array<double,3>,7> norms;
        // the norms of the full tensor are only the same in the SCC axes.

        for (unsigned short int permutation_i = 0; permutation_i < 3; permutation_i++)
          {
            //std::cout << "permutation_i = " << permutation_i << std::endl;
            std::array<unsigned short int, 3> perumation = indexed_even_permutation(permutation_i);


            for (size_t j = 0; j < 3; j++)
              {
                permutated[permutation_i][j] = unpermutated_SCC[perumation[j]];
              }

            rotated_elastic_matrix[permutation_i] = LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_matrix,(permutated[permutation_i]));

            const Tensor<1,21> full_elastic_vector_rotated = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(rotated_elastic_matrix[permutation_i]);


            const double full_norm_square = full_elastic_vector_rotated.norm_square();
            norms[6][permutation_i] = full_norm_square;
            //std::cout << " ===> full: full_square_norm = " << full_norm_square << std::endl;
            auto mono_and_higher_vector = projection_matrix_tric_to_mono*full_elastic_vector_rotated;
            auto tric_vector = full_elastic_vector_rotated-mono_and_higher_vector;
            //auto tric = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tric_vector);
            norms[0][permutation_i] = tric_vector.norm_square();
            //std::cout << " ===> tric: norm = " << tric_vector.norm_square() << ", perc = " << (tric_vector.norm_square()/full_norm_square)*100 << std::endl;

            auto ortho_and_higher_vector = projection_matrix_mono_to_ortho*mono_and_higher_vector;
            auto mono_vector = mono_and_higher_vector-ortho_and_higher_vector;
            //auto mono = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(mono_vector);
            norms[1][permutation_i] = mono_vector.norm_square();
            //std::cout << " ===> mono: norm = " << mono_vector.norm_square() << ", perc = " << (mono_vector.norm_square()/full_norm_square)*100 << std::endl;


            auto tetra_and_higher_vector = projection_matrix_ortho_to_tetra*ortho_and_higher_vector;
            auto ortho_vector = ortho_and_higher_vector-tetra_and_higher_vector;
            //auto ortho = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(ortho_vector);
            norms[2][permutation_i] = ortho_vector.norm_square();
            //std::cout << " ===> ortho: norm = " << ortho_vector.norm_square() << ", perc = " << (ortho_vector.norm_square()/full_norm_square)*100 << std::endl;

            auto hexa_and_higher_vector = projection_matrix_tetra_to_hexa*tetra_and_higher_vector;
            auto tetra_vector = tetra_and_higher_vector-hexa_and_higher_vector;
            //auto tetra = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_vector);
            norms[3][permutation_i] = tetra_vector.norm_square();
            //std::cout << " hexa_and_higher_vector = " << hexa_and_higher_vector << ", tetra_vector = " << tetra_vector << ", tetra_and_higher_vector = " << tetra_and_higher_vector << std::endl;
            //std::cout << " ===> tetra: norm = " << tetra_vector.norm_square() << ", perc = " << (tetra_vector.norm_square()/full_norm_square)*100 << std::endl;//LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tetra_and_higher_vector-projection_matrix_tetra_to_hexa*tetra_and_higher_vector) << std::endl;


            auto iso_vector = projection_matrix_hexa_to_iso*hexa_and_higher_vector;
            auto hexa_vector = hexa_and_higher_vector-iso_vector;
            //auto hexa = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(hexa_vector);
            //auto iso = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(iso_vector);
            norms[4][permutation_i] = hexa_vector.norm_square();
            norms[5][permutation_i] = iso_vector.norm_square();
            //std::cout << " ===> hexa: norm = " << hexa_vector.norm_square() << ", perc = " << (hexa_vector.norm_square()/full_norm_square)*100  << std::endl;
            //std::cout << " ===> iso: norm = " << iso_vector.norm_square() << ", perc = " << (iso_vector.norm_square()/full_norm_square)*100 << std::endl;
            //std::cout << "inv perc ani = " << ((full_elastic_vector_rotated - iso_vector).norm_square()/full_norm_square)*100. << std::endl;//<< ", sqrt prec = " <<  ((full_elastic_vector_rotated - iso_vector).norm()/full_norm)*100.<<std::endl;
            //const double total_anisotropic = hexa_vector.norm_square() + tetra_vector.norm_square() + ortho_vector.norm_square() + mono_vector.norm_square() + tric_vector.norm_square();
            //std::cout << "%anisotropy = "  <<  ((full_elastic_vector_rotated - iso_vector).norm_square()/full_norm_square)*100.
            //          << ", %of anisotropic: hexa =  " << (hexa_vector.norm_square()/total_anisotropic)*100. << ", tetra = " << (tetra_vector.norm_square()/total_anisotropic)*100.
            //          << ", ortho = " << (ortho_vector.norm_square()/total_anisotropic)*100 << ", mono = " << (mono_vector.norm_square()/total_anisotropic)*100 << ", tric = " << (tric_vector.norm_square()/total_anisotropic)*100 << std::endl;


          }
        return norms;

      }


      template<int dim>
      std::pair<SymmetricTensor<2,6>,Tensor<2,3> >
      LpoHexagonalAxes<dim>::compute_minimum_hexagonal_projection(
        const Tensor<2,3> &unpermutated_SCC,
        const SymmetricTensor<2,6> &elastic_matrix,
        const double ) const
      {
        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_matrix);

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
        double lowest_norm_classic = elastic_vector.norm_square()*100;
        unsigned short int lowest_norm_permutation_classic = 99;

        Tensor<2,3> permutated[3];
        SymmetricTensor<2,6> rotated_elastic_matrix[3];
        for (unsigned short int permutation_i = 0; permutation_i < 3; permutation_i++)
          {
            std::array<unsigned short int, 3> perumation = indexed_even_permutation(permutation_i);

            for (size_t j = 0; j < 3; j++)
              {
                permutated[permutation_i][j] = unpermutated_SCC[perumation[j]];
              }

            rotated_elastic_matrix[permutation_i] = LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_matrix,(permutated[permutation_i]));

            const Tensor<1,21> full_elastic_vector_rotated = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(rotated_elastic_matrix[permutation_i]);
            const Tensor<1,9> hexagonal_elastic_vector = project_onto_hexagonal_symmetry(full_elastic_vector_rotated);

            Tensor<1,21> elastic_vector_tmp = full_elastic_vector_rotated;//elastic_vector;// full_elastic_vector_rotated;

            // now compute how much is left over in the origional elastic vector
            for (size_t substraction_i = 0; substraction_i < 9; substraction_i++)
              {
                elastic_vector_tmp[substraction_i] -= hexagonal_elastic_vector[substraction_i];
              }

            const double current_norm = elastic_vector_tmp.norm_square();

            if (current_norm < lowest_norm_classic)
              {
                lowest_norm_classic = current_norm;
                lowest_norm_permutation_classic = permutation_i;
              }

            //std::cout << "   => " << permutation_i << ": lowest_norm = " << lowest_norm_classic << ", current_norm = " << current_norm << ", lowest_norm_permutation = " << lowest_norm_permutation_classic << std::endl;
          }

        AssertThrow(lowest_norm_permutation_classic < 3,
                    ExcMessage("LPO Hexagonal axes plugin could not find a good hexagonal projection: " + std::to_string(lowest_norm_permutation_classic) + ", lowest_norm = " + std::to_string(lowest_norm_classic)));


        return std::make_pair(rotated_elastic_matrix[lowest_norm_permutation_classic],permutated[lowest_norm_permutation_classic]);
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

        property_information.push_back(std::make_pair("lpo elastic axis e1",3));
        property_information.push_back(std::make_pair("lpo elastic axis e2",3));
        property_information.push_back(std::make_pair("lpo elastic axis e3",3));
        property_information.push_back(std::make_pair("lpo elastic hexagonal axis",3));
        property_information.push_back(std::make_pair("lpo elastic vector norm square",1));
        property_information.push_back(std::make_pair("lpo elastic triclinic vector norm square p1",1));
        property_information.push_back(std::make_pair("lpo elastic triclinic vector norm square p2",1));
        property_information.push_back(std::make_pair("lpo elastic triclinic vector norm square p3",1));
        property_information.push_back(std::make_pair("lpo elastic monoclinic vector norm square p1",1));
        property_information.push_back(std::make_pair("lpo elastic monoclinic vector norm square p2",1));
        property_information.push_back(std::make_pair("lpo elastic monoclinic vector norm square p3",1));
        property_information.push_back(std::make_pair("lpo elastic orthorhombic vector norm square p1",1));
        property_information.push_back(std::make_pair("lpo elastic orthorhombic vector norm square p2",1));
        property_information.push_back(std::make_pair("lpo elastic orthorhombic vector norm square p3",1));
        property_information.push_back(std::make_pair("lpo elastic tetragonal vector norm square p1",1));
        property_information.push_back(std::make_pair("lpo elastic tetragonal vector norm square p2",1));
        property_information.push_back(std::make_pair("lpo elastic tetragonal vector norm square p3",1));
        property_information.push_back(std::make_pair("lpo elastic hexagonal vector norm square p1",1));
        property_information.push_back(std::make_pair("lpo elastic hexagonal vector norm square p2",1));
        property_information.push_back(std::make_pair("lpo elastic hexagonal vector norm square p3",1));
        property_information.push_back(std::make_pair("lpo elastic isotropic vector norm square",1));

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

