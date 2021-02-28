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
#include <aspect/particle/property/decompose_elastic_matrix.h>
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

      template <int dim> SymmetricTensor<2,21> DecomposeElasticMatrix<dim>::projection_matrix_tric_to_mono;
      template <int dim> SymmetricTensor<2,9> DecomposeElasticMatrix<dim>::projection_matrix_mono_to_ortho;
      template <int dim> SymmetricTensor<2,9> DecomposeElasticMatrix<dim>::projection_matrix_ortho_to_tetra;
      template <int dim> SymmetricTensor<2,9> DecomposeElasticMatrix<dim>::projection_matrix_tetra_to_hexa;
      template <int dim> SymmetricTensor<2,9> DecomposeElasticMatrix<dim>::projection_matrix_hexa_to_iso;
      template <int dim> Tensor<3,3> DecomposeElasticMatrix<dim>::permutation_operator_3d;


      template <int dim>
      DecomposeElasticMatrix<dim>::DecomposeElasticMatrix ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;


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

      template <int dim>
      void
      DecomposeElasticMatrix<dim>::initialize ()
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
        Assert(manager.plugin_name_exists("decompose elastic matrix"),
               ExcMessage("No hexagonal axes property plugin found."));

        //AssertThrow(manager.check_plugin_order("lpo","decompose elastic matrix"),
        //            ExcMessage("To use the decompose elastic matrix plugin, the lpo plugin need to be defined before this plugin."));

        AssertThrow(manager.check_plugin_order("lpo elastic tensor","decompose elastic matrix"),
                    ExcMessage("To use the decompose elastic matrix plugin, the lpo elastic tensor plugin need to be defined before this plugin."));

        //lpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo"));
        lpo_elastic_tensor_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo elastic tensor"));
//std::cout << "flag 1" << std::endl;
      }



      template <int dim>
      void
      DecomposeElasticMatrix<dim>::initialize_one_particle_property(const Point<dim> &,
                                                                    std::vector<double> &data) const
      {
        SymmetricTensor<2,6> elastic_matrix;
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

      }

      template <int dim>
      void
      DecomposeElasticMatrix<dim>::update_one_particle_property(const unsigned int data_position,
                                                                const Point<dim> &,
                                                                const Vector<double> &,
                                                                const std::vector<Tensor<1,dim> > &,
                                                                const ArrayView<double> &data) const
      {
        SymmetricTensor<2,6> elastic_matrix;
        Particle::Property::LpoElasticTensor<dim>::load_particle_data(lpo_elastic_tensor_data_position,
                                                                      data,
                                                                      elastic_matrix);


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
      }


      template<int dim>
      std::array<unsigned short int, 3>
      DecomposeElasticMatrix<dim>::indexed_even_permutation(const unsigned short int index)
      {
        // there are 6 permutations, but only the odd or even are needed. We use the even
        // permutation here.
        switch (index)
          {
            case 0 :
              return {{0,1,2}};
            case 1 :
              return {{1,2,0}};
            case 2:
              return {{2,0,1}};
            /*case 3:
              return {0,2,1};
            case 4 :
              return {1,0,2};
            case 5:
              return {2,1,0};*/
            default:
              AssertThrow(false,ExcMessage("Provided index larger then 2 (" + std::to_string(index)+ ")."));
              return {{0,0,0}};
          }

      }

      template<int dim>
      SymmetricTensor<2,3>
      DecomposeElasticMatrix<dim>::compute_voigt_stiffness_tensor(const SymmetricTensor<2,6> &elastic_matrix)
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
      DecomposeElasticMatrix<dim>::compute_dilatation_stiffness_tensor(const SymmetricTensor<2,6> &elastic_matrix)
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
      DecomposeElasticMatrix<dim>::compute_bulk_and_shear_moduli(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                                                 const SymmetricTensor<2,3> &voigt_stiffness_tensor)
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
      DecomposeElasticMatrix<dim>::compute_isotropic_approximation(const double bulk_modulus,
                                                                   const double shear_modulus)
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
      DecomposeElasticMatrix<dim>::compute_unpermutated_SCC(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                                            const SymmetricTensor<2,3> &voigt_stiffness_tensor)
      {
        // computing the eigenvector of the dilation and voigt stiffness matrices and then averaging them by bysection.
        const std::array<std::pair<double,Tensor<1,3,double> >, 3> voigt_eigenvectors_a = eigenvectors(voigt_stiffness_tensor, SymmetricTensorEigenvectorMethod::jacobi);
        const std::array<std::pair<double,Tensor<1,3,double> >, 3> dilatation_eigenvectors_a = eigenvectors(dilatation_stiffness_tensor, SymmetricTensorEigenvectorMethod::jacobi);

        //std::cout << "voigt_eigenvectors = " << std::endl;
        //for (size_t iii = 0; iii < 3; iii++)
        //{
        //  for (size_t jjj = 0; jjj < 3; jjj++)
        //  {
        //    std::cout << voigt_eigenvectors_a[iii].second[jjj] << " ";
        //  }
        //  std::cout << std::endl;
        //}
        //std::cout << std::endl;

        //std::cout << "dilatation_eigenvectors = " << std::endl;
        //for (size_t iii = 0; iii < 3; iii++)
        //{
        //  for (size_t jjj = 0; jjj < 3; jjj++)
        //  {
        //    std::cout << dilatation_eigenvectors_a[iii].second[jjj] << " ";
        //  }
        //  std::cout << std::endl;
        //}
        //std::cout << std::endl;


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
                double dv_dot_product = dilatation_eigenvectors_a[i1].second*voigt_eigenvectors_a[i2].second;
                // limit the dot product between 1 and -1 so we can use the arccos function safely.
                if (std::abs(dv_dot_product) >= 1.0)
                  dv_dot_product = std::copysign(1.0,dv_dot_product);
                // compute the angle bewteen the vectors and account for that vectors in the oposit
                // direction are the same. So limit them between 0 and 90 degrees such that it
                // represents the minimum angle between the two lines.
                double ADV = dv_dot_product < 0.0 ? std::acos(-1.0)-std::acos(dv_dot_product) : std::acos(dv_dot_product);
                // store this if the angle is smaller
                if (ADV < ADVC)
                  {
                    NDVC=std::copysign(1.0, dv_dot_product)*i2;
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
      DecomposeElasticMatrix<dim>::compute_elastic_tensor_SCC_decompositions(
        const Tensor<2,3> &unpermutated_SCC,
        const SymmetricTensor<2,6> &elastic_matrix)
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

        for (unsigned short int permutation_i = 0; permutation_i < 3; permutation_i++)
          {
            //std::cout << "permutation_i = " << permutation_i << std::endl;
            std::array<unsigned short int, 3> perumation = indexed_even_permutation(permutation_i);

            for (size_t j = 0; j < 3; j++)
              {
                permutated[permutation_i][j] = unpermutated_SCC[perumation[j]];
              }
            //std::cout << "permutated scc: " << permutated[permutation_i] << std::endl;

            rotated_elastic_matrix[permutation_i] = LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_matrix,(permutated[permutation_i]));

            const Tensor<1,21> full_elastic_vector_rotated = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(rotated_elastic_matrix[permutation_i]);


            const double full_norm_square = full_elastic_vector_rotated.norm_square();
            norms[6][permutation_i] = full_norm_square;

            // The following line would do the same as the lines below, but is is very slow. It has therefore been
            // replaced by the lines below.
            //auto mono_and_higher_vector = projection_matrix_tric_to_mono*full_elastic_vector_rotated;
            dealii::Tensor<1,21> mono_and_higher_vector;
            mono_and_higher_vector[0] = full_elastic_vector_rotated[0];
            mono_and_higher_vector[1] = full_elastic_vector_rotated[1];
            mono_and_higher_vector[2] = full_elastic_vector_rotated[2];
            mono_and_higher_vector[3] = full_elastic_vector_rotated[3];
            mono_and_higher_vector[4] = full_elastic_vector_rotated[4];
            mono_and_higher_vector[5] = full_elastic_vector_rotated[5];
            mono_and_higher_vector[6] = full_elastic_vector_rotated[6];
            mono_and_higher_vector[7] = full_elastic_vector_rotated[7];
            mono_and_higher_vector[8] = full_elastic_vector_rotated[8];
            mono_and_higher_vector[11] = full_elastic_vector_rotated[11];
            mono_and_higher_vector[14] = full_elastic_vector_rotated[14];
            mono_and_higher_vector[17] = full_elastic_vector_rotated[17];
            mono_and_higher_vector[20] = full_elastic_vector_rotated[20];

            auto tric_vector = full_elastic_vector_rotated-mono_and_higher_vector;
            //auto tric = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(tric_vector);
            norms[0][permutation_i] = tric_vector.norm_square();

            // The following line would do the same as the lines below, but is is slow. It has therefore been
            // replaced by the lines below.
            //auto ortho_and_higher_vector = projection_matrix_mono_to_ortho*mono_and_higher_vector;
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
            dealii::Tensor<1,9> ortho_and_higher_vector;
            ortho_and_higher_vector[0] = mono_and_higher_vector[0];
            ortho_and_higher_vector[1] = mono_and_higher_vector[1];
            ortho_and_higher_vector[2] = mono_and_higher_vector[2];
            ortho_and_higher_vector[3] = mono_and_higher_vector[3];
            ortho_and_higher_vector[4] = mono_and_higher_vector[4];
            ortho_and_higher_vector[5] = mono_and_higher_vector[5];
            ortho_and_higher_vector[6] = mono_and_higher_vector[6];
            ortho_and_higher_vector[7] = mono_and_higher_vector[7];
            ortho_and_higher_vector[8] = mono_and_higher_vector[8];
            auto mono_vector = mono_and_higher_vector_croped-ortho_and_higher_vector;
            //auto mono = LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(mono_vector);
            norms[1][permutation_i] = mono_vector.norm_square();


            auto tetra_and_higher_vector = projection_matrix_ortho_to_tetra*ortho_and_higher_vector;
            auto ortho_vector = ortho_and_higher_vector-tetra_and_higher_vector;
            norms[2][permutation_i] = ortho_vector.norm_square();

            auto hexa_and_higher_vector = projection_matrix_tetra_to_hexa*tetra_and_higher_vector;
            auto tetra_vector = tetra_and_higher_vector-hexa_and_higher_vector;
            norms[3][permutation_i] = tetra_vector.norm_square();

            auto iso_vector = projection_matrix_hexa_to_iso*hexa_and_higher_vector;
            auto hexa_vector = hexa_and_higher_vector-iso_vector;
            norms[4][permutation_i] = hexa_vector.norm_square();
            norms[5][permutation_i] = iso_vector.norm_square();

          }
        return norms;

      }


      template<int dim>
      std::pair<SymmetricTensor<2,6>,Tensor<2,3> >
      DecomposeElasticMatrix<dim>::compute_minimum_hexagonal_projection(
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
          }

        AssertThrow(lowest_norm_permutation_classic < 3,
                    ExcMessage("Decompose Elastic Matrix plugin could not find a good hexagonal projection: " + std::to_string(lowest_norm_permutation_classic) + ", lowest_norm = " + std::to_string(lowest_norm_classic)));


        return std::make_pair(rotated_elastic_matrix[lowest_norm_permutation_classic],permutated[lowest_norm_permutation_classic]);
      }


      template<int dim>
      Tensor<1,9>
      DecomposeElasticMatrix<dim>::project_onto_hexagonal_symmetry(const Tensor<1,21> &elastic_vector) const
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
      DecomposeElasticMatrix<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      DecomposeElasticMatrix<dim>::get_needed_update_flags () const
      {
        return update_default;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      DecomposeElasticMatrix<dim>::get_property_information() const
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
      DecomposeElasticMatrix<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("DecomposeElasticMatrix");
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
      DecomposeElasticMatrix<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("DecomposeElasticMatrix");
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(DecomposeElasticMatrix,
                                        "decompose elastic matrix",
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
