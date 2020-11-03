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
      LpoElasticTensor<dim>::LpoElasticTensor ()
      {
        permutation_operator_3d[0][1][2]  = 1;
        permutation_operator_3d[1][2][0]  = 1;
        permutation_operator_3d[2][0][1]  = 1;
        permutation_operator_3d[0][2][1]  = -1;
        permutation_operator_3d[1][0][2]  = -1;
        permutation_operator_3d[2][1][0]  = -1;

        // The following values are directly form D-Rex.
        // Todo: make them a input parameter
        // Stiffness matrix for Olivine (GigaPascals)
        stiffness_matrix_olivine[0][0] = 320.71;
        stiffness_matrix_olivine[0][1] = 69.84;
        stiffness_matrix_olivine[0][2] = 71.22;
        stiffness_matrix_olivine[1][1] = 197.25;
        stiffness_matrix_olivine[1][2] = 74.8;
        stiffness_matrix_olivine[2][2] = 234.32;
        stiffness_matrix_olivine[3][3] = 63.77;
        stiffness_matrix_olivine[4][4] = 77.67;
        stiffness_matrix_olivine[5][5] = 78.36;


        // Stiffness matrix for Enstatite (GPa)
        stiffness_matrix_enstatite[0][0] = 236.9;
        stiffness_matrix_enstatite[0][1] = 79.6;
        stiffness_matrix_enstatite[0][2] = 63.2;
        stiffness_matrix_enstatite[1][1] = 180.5;
        stiffness_matrix_enstatite[1][2] = 56.8;
        stiffness_matrix_enstatite[2][2] = 230.4;
        stiffness_matrix_enstatite[3][3] = 84.3;
        stiffness_matrix_enstatite[4][4] = 79.4;
        stiffness_matrix_enstatite[5][5] = 80.1;

        // tensors of indices
        indices_tensor[0][0] = 0;
        indices_tensor[0][1] = 5;
        indices_tensor[0][2] = 4;
        indices_tensor[1][0] = 5;
        indices_tensor[1][1] = 1;
        indices_tensor[1][2] = 3;
        indices_tensor[2][0] = 4;
        indices_tensor[2][1] = 3;
        indices_tensor[2][2] = 2;

        // vectors of indices
        indices_vector_1.resize(6);
        indices_vector_1[0] = 0;
        indices_vector_1[1] = 1;
        indices_vector_1[2] = 2;
        indices_vector_1[3] = 1;
        indices_vector_1[4] = 2;
        indices_vector_1[5] = 0;

        indices_vector_2.resize(6);
        indices_vector_2[0] = 0;
        indices_vector_2[1] = 1;
        indices_vector_2[2] = 2;
        indices_vector_2[3] = 2;
        indices_vector_2[4] = 0;
        indices_vector_2[5] = 1;
      }

      template <int dim>
      void
      LpoElasticTensor<dim>::initialize ()
      {
        // todo: check wheter this works correctly. Since the get_random_number function takes a reference
        // to the random_number_generator function, changing the function should mean that I have to update the
        // get_random_number function as well. But I will need to test this.
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);
        //std::cout << ">>> random_number_seed+my_rank = " << random_number_seed+my_rank << ", random_number_seed = " << random_number_seed << std::endl;

        const auto &manager = this->get_particle_world().get_property_manager();
        AssertThrow(manager.plugin_name_exists("lpo"),
                    ExcMessage("No lpo property plugin found."));
        Assert(manager.plugin_name_exists("lpo elastic tensor"),
               ExcMessage("No s wave anisotropy property plugin found."));

        AssertThrow(manager.check_plugin_order("lpo","lpo elastic tensor"),
                    ExcMessage("To use the lpo elastic tensor plugin, the lpo plugin need to be defined before this plugin."));

        lpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo"));


      }





      template <int dim>
      SymmetricTensor<2,6>
      LpoElasticTensor<dim>::compute_elastic_tensor (const double volume_fraction_olivine,
                                                     std::vector<double> &volume_fractions_olivine,
                                                     std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                                     std::vector<double> &volume_fractions_enstatite,
                                                     std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite) const
      {
        //std::cout << "volume_fraction_olivine = " << volume_fraction_olivine << std::endl;
        /** This implements the Voigt averaging as described in the equation at the
        * bottom of page 385 in Mainprice (1990):
        * $C^V_{ijkl} = \sum^S_l F_s \sum^{N_s}_l C_{ijkl}/N_s$, where $F_s$ is the
        * grain size, $N_s$ is the number of grains and $C_{ijkl}$ is the elastic
        * tensor. This elastic tensor is computed by the equation above in
        * Mainprice (1990): $C_{ijkl} = R_{ip} R_{jg} R_{kr} R_{is} C_{pgrs}$, where
        * R_{ij} is the lpo orientation matrix.
        */
        Tensor<4,3,double> Cav;
        Tensor<4,3,double> C0;
        // add olivine to the matrix
        // compute C0 for olivine
        //std::cout << "C0:" << std::endl;
        // Turn the olivine stiffness matrix into a rank 4 tensor
        for (size_t i = 0; i < 3; i++)
          {
            for (size_t j = 0; j < 3; j++)
              {
                for (size_t k = 0; k < 3; k++)
                  {
                    for (size_t l = 0; l < 3; l++)
                      {
                        C0[i][j][k][l] = stiffness_matrix_olivine[indices_tensor[i][j]][indices_tensor[k][l]];
                        //std::cout << "C0["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"] = "<<C0[i][j][k][l] << std::endl;
                      }
                  }
              }
          }

        // Do the Voigt average for olvine
        for (size_t grain_i = 0; grain_i < volume_fractions_olivine.size(); grain_i++)
          {
            Tensor<4,3,double> Cav2;

            Tensor<2,3> &acmo = a_cosine_matrices_olivine[grain_i];
            for (size_t i = 0; i < 3; i++)
              {
                for (size_t j = 0; j < 3; j++)
                  {
                    for (size_t k = 0; k < 3; k++)
                      {
                        for (size_t l = 0; l < 3; l++)
                          {
                            for (size_t p = 0; p < 3; p++)
                              {
                                for (size_t q = 0; q < 3; q++)
                                  {
                                    for (size_t r = 0; r < 3; r++)
                                      {
                                        for (size_t s = 0; s < 3; s++)
                                          {
                                            Cav2[i][j][k][l] += acmo[p][i]*acmo[q][j]*acmo[r][k]*acmo[s][l]*C0[p][q][r][s];
                                            //std::cout << "Cav2["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"] = "<<Cav2[i][j][k][l] <<" = " << acmo[p][i] << "..." << std::endl;
                                          }
                                      }
                                  }
                              }
                            Cav[i][j][k][l] += Cav2[i][j][k][l] *  volume_fractions_olivine[grain_i] * volume_fraction_olivine;
                            //std::cout << "Cav["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"] = "<<Cav[i][j][k][l] <<":" << Cav2[i][j][k][l] << std::endl;
                          }
                      }
                  }
              }
          }

        // add enstatite for the matrix
        // compute C0 for enstatite
        // Turn the enstatite stiffness matrix into a rank 4 tensor
        for (size_t i = 0; i < 3; i++)
          {
            for (size_t j = 0; j < 3; j++)
              {
                for (size_t k = 0; k < 3; k++)
                  {
                    for (size_t l = 0; l < 3; l++)
                      {
                        C0[i][j][k][l] = stiffness_matrix_enstatite[indices_tensor[i][j]][indices_tensor[k][l]];
                        //std::cout << "C0["<<i<<"]["<<j<<"]["<<k<<"]["<<l<<"] = "<<C0[i][j][k][l] << std::endl;
                      }
                  }
              }
          }
        // Add the Voigt average for enstatite
        for (size_t grain_i = 0; grain_i < volume_fractions_olivine.size(); grain_i++)
          {
            Tensor<4,3,double> Cav2;

            Tensor<2,3> &acme = a_cosine_matrices_enstatite[grain_i];
            for (size_t i = 0; i < 3; i++)
              {
                for (size_t j = 0; j < 3; j++)
                  {
                    for (size_t k = 0; k < 3; k++)
                      {
                        for (size_t l = 0; l < 3; l++)
                          {
                            for (size_t p = 0; p < 3; p++)
                              {
                                for (size_t q = 0; q < 3; q++)
                                  {
                                    for (size_t r = 0; r < 3; r++)
                                      {
                                        for (size_t s = 0; s < 3; s++)
                                          {
                                            Cav2[i][j][k][l] += acme[p][i]*acme[q][j]*acme[r][k]*acme[s][l]*C0[p][q][r][s];
                                          }
                                      }
                                  }
                              }
                            Cav[i][j][k][l] += Cav2[i][j][k][l] *  volume_fractions_enstatite[grain_i] * (1-volume_fraction_olivine);
                          }
                      }
                  }
              }
          }

        // turn rank 4 Voigt averaged stiffness tensor into a stiffness matrix
        // TODO: optimize loop for symmetric matrices
        SymmetricTensor<2,6> Sav;
        for (size_t i = 0; i < 6; i++)
          {
            for (size_t j = 0; j < 6; j++)
              {
                Sav[i][j] = Cav[indices_vector_1[i]][indices_vector_2[i]][indices_vector_1[j]][indices_vector_2[j]];
              }

          }

        return Sav;
      }



      template <int dim>
      void
      LpoElasticTensor<dim>::initialize_one_particle_property(const Point<dim> &,
                                                              std::vector<double> &data) const
      {

        double water_content = 0;
        double volume_fraction_olivine = 0;
        std::vector<double> volume_fractions_olivine(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
        std::vector<double> volume_fractions_enstatite(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);

        //std::cout << "lpo_data_position = " << lpo_data_position << ", n_grains = " << n_grains << std::endl;
        Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         n_grains,
                                                         water_content,
                                                         volume_fraction_olivine,
                                                         volume_fractions_olivine,
                                                         a_cosine_matrices_olivine,
                                                         volume_fractions_enstatite,
                                                         a_cosine_matrices_enstatite);

        SymmetricTensor<2,6> S_average = compute_elastic_tensor(volume_fraction_olivine,volume_fractions_olivine,
                                                                a_cosine_matrices_olivine,
                                                                volume_fractions_enstatite,
                                                                a_cosine_matrices_enstatite);

        /*for (size_t i = 0; i < 6; i++)
        {
          for (size_t i = 0; i < 6; i++)
          {
            data.push_back(S_average[SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
          }

        }*/

        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
#if DEAL_II_VERSION_GTE(9,3,0)
          data.push_back(S_average[SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
          {
            if (i < 6)
              {
                data.push_back(S_average[ {i,i}]);
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
                              data.push_back(S_average[ {d,e}]);
                              return;
                            }
                        }
                    }
                }();
              }
          }
#endif

      }

      template <int dim>
      void
      LpoElasticTensor<dim>::update_one_particle_property(const unsigned int data_position,
                                                          const Point<dim> &,
                                                          const Vector<double> &,
                                                          const std::vector<Tensor<1,dim> > &,
                                                          const ArrayView<double> &data) const
      {

        double water_content = 0;
        double volume_fraction_olivine = 0;
        std::vector<double> volume_fractions_olivine(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
        std::vector<double> volume_fractions_enstatite(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);

        Particle::Property::LPO<dim>::load_particle_data(lpo_data_position,
                                                         data,
                                                         n_grains,
                                                         water_content,
                                                         volume_fraction_olivine,
                                                         volume_fractions_olivine,
                                                         a_cosine_matrices_olivine,
                                                         volume_fractions_enstatite,
                                                         a_cosine_matrices_enstatite);

        SymmetricTensor<2,6> S_average = compute_elastic_tensor(volume_fraction_olivine,
                                                                volume_fractions_olivine,
                                                                a_cosine_matrices_olivine,
                                                                volume_fractions_enstatite,
                                                                a_cosine_matrices_enstatite);

        Particle::Property::LpoElasticTensor<dim>::store_particle_data(data_position,
                                                                       data,
                                                                       S_average);


      }


      template <int dim>
      void
      LpoElasticTensor<dim>::load_particle_data(unsigned int lpo_data_position,
                                                const ArrayView<double> &data,
                                                SymmetricTensor<2,6> &elastic_tensor)
      {

        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
#if DEAL_II_VERSION_GTE(9,3,0)
          data.push_back(S_average[SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
          {
            if (i < 6)
              {
                elastic_tensor[ {i,i}] = data[lpo_data_position + i];
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
                              elastic_tensor[ {d,e}] = data[lpo_data_position + i];
                              return;
                            }
                        }
                    }
                }();
              }
          }
#endif
        //for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
        //elastic_tensor[SymmetricTensor<2,6>::unrolled_to_component_indices(i)] = data[lpo_data_position + i];
      }


      template <int dim>
      void
      LpoElasticTensor<dim>::store_particle_data(unsigned int lpo_data_position,
                                                 const ArrayView<double> &data,
                                                 SymmetricTensor<2,6> &elastic_tensor)
      {
        // There is a bug up to dealii 9.3.0, so we have to work around it.
        for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
#if DEAL_II_VERSION_GTE(9,3,0)
          data.push_back(S_average[SymmetricTensor<2,6>::unrolled_to_component_indices(i)]);
#else
          {
            if (i < 6)
              {
                data[lpo_data_position + i] = elastic_tensor[ {i,i}];
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
                              data[lpo_data_position + i] = elastic_tensor[ {d,e}];
                              return;
                            }
                        }
                    }
                }();
              }
          }
#endif
        //for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
        //  data[lpo_data_position + i] = elastic_tensor[SymmetricTensor<2,6>::unrolled_to_component_indices(i)];
      }


      template<int dim>
      Tensor<4,3>
      LpoElasticTensor<dim>::rotate_4th_order_tensor(const Tensor<4,3> &input_tensor, const Tensor<2,3> &rotation_tensor)
      {
        Tensor<4,3> output;

        for (unsigned short int i1 = 0; i1 < 3; i1++)
          {
            for (unsigned short int i2 = 0; i2 < 3; i2++)
              {
                for (unsigned short int i3 = 0; i3 < 3; i3++)
                  {
                    for (unsigned short int i4 = 0; i4 < 3; i4++)
                      {
                        for (unsigned short int j1 = 0; j1 < 3; j1++)
                          {
                            for (unsigned short int j2 = 0; j2 < 3; j2++)
                              {
                                for (unsigned short int j3 = 0; j3 < 3; j3++)
                                  {
                                    for (unsigned short int j4 = 0; j4 < 3; j4++)
                                      {
                                        output[i1][i2][i3][i4] = output[i1][i2][i3][i4] + rotation_tensor[i1][j1]*rotation_tensor[i2][j2]*rotation_tensor[i3][j3]*rotation_tensor[i4][j4]*input_tensor[j1][j2][j3][j4];
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }

        return output;
      }

      template<int dim>
      SymmetricTensor<2,6>
      LpoElasticTensor<dim>::rotate_6x6_matrix(const Tensor<2,6> &input_tensor, const Tensor<2,3> &rotation_tensor)
      {
        // we can represent the roation of the 4th order tensor as a rotation in the voigt
        // notation by computing $C'=MCM^{-1}$. Because M is orhtogonal we can replace $M^{-1}$
        // with $M^T$ resutling in $C'=MCM^{T}$ (Carcione, J. M. (2007). Wave Fields in Real Media:
        // Wave Propagation in Anisotropic, Anelastic, Porous and Electromagnetic Media. Netherlands:
        // Elsevier Science. Pages 8-9).
        Tensor<2,6> rotation_matrix;
        // top left block
        rotation_matrix[0][0] = rotation_tensor[0][0] * rotation_tensor[0][0];
        rotation_matrix[1][0] = rotation_tensor[1][0] * rotation_tensor[1][0];
        rotation_matrix[2][0] = rotation_tensor[2][0] * rotation_tensor[2][0];
        rotation_matrix[0][1] = rotation_tensor[0][1] * rotation_tensor[0][1];
        rotation_matrix[1][1] = rotation_tensor[1][1] * rotation_tensor[1][1];
        rotation_matrix[2][1] = rotation_tensor[2][1] * rotation_tensor[2][1];
        rotation_matrix[0][2] = rotation_tensor[0][2] * rotation_tensor[0][2];
        rotation_matrix[1][2] = rotation_tensor[1][2] * rotation_tensor[1][2];
        rotation_matrix[2][2] = rotation_tensor[2][2] * rotation_tensor[2][2];

        // top right block
        rotation_matrix[0][3] = 2.0 * rotation_tensor[0][1] * rotation_tensor[0][2];
        rotation_matrix[1][3] = 2.0 * rotation_tensor[1][1] * rotation_tensor[1][2];
        rotation_matrix[2][3] = 2.0 * rotation_tensor[2][1] * rotation_tensor[2][2];
        rotation_matrix[0][4] = 2.0 * rotation_tensor[0][2] * rotation_tensor[0][0];
        rotation_matrix[1][4] = 2.0 * rotation_tensor[1][2] * rotation_tensor[1][0];
        rotation_matrix[2][4] = 2.0 * rotation_tensor[2][2] * rotation_tensor[2][0];
        rotation_matrix[0][5] = 2.0 * rotation_tensor[0][0] * rotation_tensor[0][1];
        rotation_matrix[1][5] = 2.0 * rotation_tensor[1][0] * rotation_tensor[1][1];
        rotation_matrix[2][5] = 2.0 * rotation_tensor[2][0] * rotation_tensor[2][1];

        // bottom left block
        rotation_matrix[3][0] = rotation_tensor[1][0] * rotation_tensor[2][0];
        rotation_matrix[4][0] = rotation_tensor[2][0] * rotation_tensor[0][0];
        rotation_matrix[5][0] = rotation_tensor[0][0] * rotation_tensor[1][0];
        rotation_matrix[3][1] = rotation_tensor[1][1] * rotation_tensor[2][1];
        rotation_matrix[4][1] = rotation_tensor[2][1] * rotation_tensor[0][1];
        rotation_matrix[5][1] = rotation_tensor[0][1] * rotation_tensor[1][1];
        rotation_matrix[3][2] = rotation_tensor[1][2] * rotation_tensor[2][2];
        rotation_matrix[4][2] = rotation_tensor[2][2] * rotation_tensor[0][2];
        rotation_matrix[5][2] = rotation_tensor[0][2] * rotation_tensor[1][2];

        // bottom right block
        rotation_matrix[3][3] = rotation_tensor[1][1] * rotation_tensor[2][2] + rotation_tensor[1][2] * rotation_tensor[2][1];
        rotation_matrix[4][3] = rotation_tensor[0][1] * rotation_tensor[2][2] + rotation_tensor[0][2] * rotation_tensor[2][1];
        rotation_matrix[5][3] = rotation_tensor[0][1] * rotation_tensor[1][2] + rotation_tensor[0][2] * rotation_tensor[1][1];
        rotation_matrix[3][4] = rotation_tensor[1][0] * rotation_tensor[2][2] + rotation_tensor[1][2] * rotation_tensor[2][0];
        rotation_matrix[4][4] = rotation_tensor[0][2] * rotation_tensor[2][0] + rotation_tensor[0][0] * rotation_tensor[2][2];
        rotation_matrix[5][4] = rotation_tensor[0][2] * rotation_tensor[1][0] + rotation_tensor[0][0] * rotation_tensor[1][2];
        rotation_matrix[3][5] = rotation_tensor[1][1] * rotation_tensor[2][0] + rotation_tensor[1][0] * rotation_tensor[2][1];
        rotation_matrix[4][5] = rotation_tensor[0][0] * rotation_tensor[2][1] + rotation_tensor[0][1] * rotation_tensor[2][0];
        rotation_matrix[5][5] = rotation_tensor[0][0] * rotation_tensor[1][1] + rotation_tensor[0][1] * rotation_tensor[1][0];

        Tensor<2,6> rotation_matrix_tranposed = transpose(rotation_matrix);

        return symmetrize((rotation_matrix*input_tensor)*rotation_matrix_tranposed);
      }



      template<int dim>
      SymmetricTensor<2,6>
      LpoElasticTensor<dim>::transform_4th_order_tensor_to_6x6_matrix(const Tensor<4,3> &input_tensor)
      {
        SymmetricTensor<2,6> output;

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][i] = input_tensor[i][i][i][i];
          }

        for (unsigned short int i = 1; i < 3; i++)
          {
            //std::cout << "i = " << i << ", tensor = " << input_tensor[0][0][i][i] << ":" << input_tensor[i][i][0][0] << std::endl;
            output[0][i] = 0.5*(input_tensor[0][0][i][i] + input_tensor[i][i][0][0]);
            //output[0][i] = output[i][0];
          }
        output[1][2]=0.5*(input_tensor[1][1][2][2]+input_tensor[2][2][1][1]);
        //output[2][1]=output[1][2];

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][3]=0.25*(input_tensor[i][i][1][2]+input_tensor[i][i][2][1]+ input_tensor[1][2][i][i]+input_tensor[2][1][i][i]);
            //output[3][i]=output[i][3];
          }

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][4]=0.25*(input_tensor[i][i][0][2]+input_tensor[i][i][2][0]+ input_tensor[0][2][i][i]+input_tensor[2][0][i][i]);
            //output[4][i]=output[i][4];
          }

        for (unsigned short int i = 0; i < 3; i++)
          {
            //std::cout << i << ":5 = " << input_tensor[i][i][0][1] << ":" << input_tensor[i][i][1][0] << ":" << input_tensor[0][1][i][i] << ":" << input_tensor[1][0][i][i] << std::endl;
            output[i][5]=0.25*(input_tensor[i][i][0][1]+input_tensor[i][i][1][0]+input_tensor[0][1][i][i]+input_tensor[1][0][i][i]);
            //output[5][i]=output[i][5];
          }

        output[3][3]=0.25*(input_tensor[1][2][1][2]+input_tensor[1][2][2][1]+input_tensor[2][1][1][2]+input_tensor[2][1][2][1]);
        output[4][4]=0.25*(input_tensor[0][2][0][2]+input_tensor[0][2][2][0]+input_tensor[2][0][0][2]+input_tensor[2][0][2][0]);
        output[5][5]=0.25*(input_tensor[1][0][1][0]+input_tensor[1][0][0][1]+input_tensor[0][1][1][0]+input_tensor[0][1][0][1]);

        output[3][4]=0.125*(input_tensor[1][2][0][2]+input_tensor[1][2][2][0]+input_tensor[2][1][0][2]+input_tensor[2][1][2][0]+input_tensor[0][2][1][2]+input_tensor[0][2][2][1]+input_tensor[2][0][1][2]+input_tensor[2][0][2][1]);
        //output[4][3]=output[3][4];
        output[3][5]=0.125*(input_tensor[1][2][0][1]+input_tensor[1][2][1][0]+input_tensor[2][1][0][1]+input_tensor[2][1][1][0]+input_tensor[0][1][1][2]+input_tensor[0][1][2][1]+input_tensor[1][0][1][2]+input_tensor[1][0][2][1]);
        //output[5][3]=output[3][5];
        output[4][5]=0.125*(input_tensor[0][2][0][1]+input_tensor[0][2][1][0]+input_tensor[2][0][0][1]+input_tensor[2][0][1][0]+input_tensor[0][1][0][2]+input_tensor[0][1][2][0]+input_tensor[1][0][0][2]+input_tensor[1][0][2][0]);
        //output[5][4]=output[4][5];

        return output;
      }

      template<int dim>
      Tensor<4,3>
      LpoElasticTensor<dim>::transform_6x6_matrix_to_4th_order_tensor(const SymmetricTensor<2,6> &input_tensor)
      {
        Tensor<4,3> output;

        for (unsigned short int i = 0; i < 3; i++)
          for (unsigned short int j = 0; j < 3; j++)
            for (unsigned short int k = 0; k < 3; k++)
              for (unsigned short int l = 0; l < 3; l++)
                {
                  // The first part of the inline if statment gets the diagonal.
                  // The second part is never higher then 5 (which is the limit of the tensor index)
                  // because to reach this part the variables need to be different, which results in
                  // at least a minus 1.
                  const unsigned short int p = (i == j ? i : 6 - i - j);
                  const unsigned short int q = (k == l ? k : 6 - k - l);
                  output[i][j][k][l] = input_tensor[p][q];
                }
        return output;
      }

      template<int dim>
      Tensor<1,21>
      LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(const SymmetricTensor<2,6> &input)
      {
        return Tensor<1,21,double> (
        {
          input[0][0],           // 0  // 1
          input[1][1],           // 1  // 2
          input[2][2],           // 2  // 3
          sqrt(2)*input[1][2],   // 3  // 4
          sqrt(2)*input[0][2],   // 4  // 5
          sqrt(2)*input[0][1],   // 5  // 6
          2*input[3][3],         // 6  // 7
          2*input[4][4],         // 7  // 8
          2*input[5][5],         // 8  // 9
          2*input[0][3],         // 9  // 10
          2*input[1][4],         // 10 // 11
          2*input[2][5],         // 11 // 12
          2*input[2][3],         // 12 // 13
          2*input[0][4],         // 13 // 14
          2*input[1][5],         // 14 // 15

          2*input[1][3],         // 15 // 16
          2*input[2][4],         // 16 // 17
          2*input[0][5],         // 17 // 18
          2*sqrt(2)*input[4][5], // 18 // 19
          2*sqrt(2)*input[3][5], // 19 // 20
          2*sqrt(2)*input[3][4]  // 20 // 21
        });

      }


      template<int dim>
      SymmetricTensor<2,6>
      LpoElasticTensor<dim>::transform_21D_vector_to_6x6_matrix(const Tensor<1,21> &input)
      {
        SymmetricTensor<2,6> result;

        constexpr double sqrt_2_inv = 1/sqrt(2);

        result[0][0] = input[0];
        result[1][1] = input[1];
        result[2][2] = input[2];
        result[1][2] = sqrt_2_inv * input[3];
        result[0][2] = sqrt_2_inv * input[4];
        result[0][1] = sqrt_2_inv * input[5];
        result[3][3] = 0.5 * input[6];
        result[4][4] = 0.5 * input[7];
        result[5][5] = 0.5 * input[8];
        result[0][3] = 0.5 * input[9];
        result[1][4] = 0.5 * input[10];
        result[2][5] = 0.5 * input[11];
        result[2][3] = 0.5 * input[12];
        result[0][4] = 0.5 * input[13];
        result[1][5] = 0.5 * input[14];
        result[1][3] = 0.5 * input[15];
        result[2][4] = 0.5 * input[16];
        result[0][5] = 0.5 * input[17];
        result[4][5] = 0.5 * sqrt_2_inv * input[18];
        result[3][5] = 0.5 * sqrt_2_inv * input[19];
        result[3][4] = 0.5 * sqrt_2_inv * input[20];

        return result;

      }


      template<int dim>
      Tensor<1,21>
      LpoElasticTensor<dim>::transform_4th_order_tensor_to_21D_vector(const Tensor<4,3> &input_tensor)
      {
        return Tensor<1,21,double> (
        {
          input_tensor[0][0][0][0],           // 0  // 1
          input_tensor[1][1][1][1],           // 1  // 2
          input_tensor[2][2][2][2],           // 2  // 3
          sqrt(2)*0.5*(input_tensor[1][1][2][2] + input_tensor[2][2][1][1]),   // 3  // 4
          sqrt(2)*0.5*(input_tensor[0][0][2][2] + input_tensor[2][2][0][0]),   // 4  // 5
          sqrt(2)*0.5*(input_tensor[0][0][1][1] + input_tensor[1][1][0][0]),   // 5  // 6
          0.5*(input_tensor[1][2][1][2]+input_tensor[1][2][2][1]+input_tensor[2][1][1][2]+input_tensor[2][1][2][1]),         // 6  // 7
          0.5*(input_tensor[0][2][0][2]+input_tensor[0][2][2][0]+input_tensor[2][0][0][2]+input_tensor[2][0][2][0]),         // 7  // 8
          0.5*(input_tensor[1][0][1][0]+input_tensor[1][0][0][1]+input_tensor[0][1][1][0]+input_tensor[0][1][0][1]),         // 8  // 9
          0.5*(input_tensor[0][0][1][2]+input_tensor[0][0][2][1]+input_tensor[1][2][0][0]+input_tensor[2][1][0][0]),         // 9  // 10
          0.5*(input_tensor[1][1][0][2]+input_tensor[1][1][2][0]+input_tensor[0][2][1][1]+input_tensor[2][0][1][1]),         // 10 // 11
          0.5*(input_tensor[2][2][0][1]+input_tensor[2][2][1][0]+input_tensor[0][1][2][2]+input_tensor[1][0][2][2]),         // 11 // 12
          0.5*(input_tensor[2][2][1][2]+input_tensor[2][2][2][1]+input_tensor[1][2][2][2]+input_tensor[2][1][2][2]),         // 12 // 13
          0.5*(input_tensor[0][0][0][2]+input_tensor[0][0][2][0]+input_tensor[0][2][0][0]+input_tensor[2][0][0][0]),         // 13 // 14
          0.5*(input_tensor[1][1][0][1]+input_tensor[1][1][1][0]+input_tensor[0][1][1][1]+input_tensor[1][0][1][1]),         // 14 // 15
          0.5*(input_tensor[1][1][1][2]+input_tensor[1][1][2][1]+input_tensor[1][2][1][1]+input_tensor[2][1][1][1]),         // 15 // 16
          0.5*(input_tensor[2][2][0][2]+input_tensor[2][2][2][0]+input_tensor[0][2][2][2]+input_tensor[2][0][2][2]),         // 16 // 17
          0.5*(input_tensor[0][0][0][1]+input_tensor[0][0][1][0]+input_tensor[0][1][0][0]+input_tensor[1][0][0][0]),         // 17 // 18
          sqrt(2)*0.25*(input_tensor[0][2][0][1]+input_tensor[0][2][1][0]+input_tensor[2][0][0][1]+input_tensor[2][0][1][0]+input_tensor[0][1][0][2]+input_tensor[0][1][2][0]+input_tensor[1][0][0][2]+input_tensor[1][0][2][0]), // 18 // 19
          sqrt(2)*0.25*(input_tensor[1][2][0][1]+input_tensor[1][2][1][0]+input_tensor[2][1][0][1]+input_tensor[2][1][1][0]+input_tensor[0][1][1][2]+input_tensor[0][1][2][1]+input_tensor[1][0][1][2]+input_tensor[1][0][2][1]), // 19 // 20
          sqrt(2)*0.25*(input_tensor[1][2][0][2]+input_tensor[1][2][2][0]+input_tensor[2][1][0][2]+input_tensor[2][1][2][0]+input_tensor[0][2][1][2]+input_tensor[0][2][2][1]+input_tensor[2][0][1][2]+input_tensor[2][0][2][1])  // 20 // 21
        });

      }

      /*
            template<int dim>
            std::array<std::array<double,3>,3>
            LpoElasticTensor<dim>::compute_s_wave_anisotropy(std::vector<Tensor<2,6> >& matrices) const
            {
              SymmetricTensor< 2, 3, double > sum_matrix_a;
              SymmetricTensor< 2, 3, double > sum_matrix_b;
              SymmetricTensor< 2, 3, double > sum_matrix_c;

              // extracting the a, b and c orientations from the olivine a matrix
              for (unsigned int i_grain = 0; i_grain < matrices.size(); i_grain++)
                {
                  sum_matrix_a[0][0] += matrices[i_grain][0][0] * matrices[i_grain][0][0]; // SUM(l^2)
                  sum_matrix_a[1][1] += matrices[i_grain][0][1] * matrices[i_grain][0][1]; // SUM(m^2)
                  sum_matrix_a[2][2] += matrices[i_grain][0][2] * matrices[i_grain][0][2]; // SUM(n^2)
                  sum_matrix_a[0][1] += matrices[i_grain][0][0] * matrices[i_grain][0][1]; // SUM(l*m)
                  sum_matrix_a[0][2] += matrices[i_grain][0][0] * matrices[i_grain][0][2]; // SUM(l*n)
                  sum_matrix_a[1][2] += matrices[i_grain][0][1] * matrices[i_grain][0][2]; // SUM(m*n)


                  sum_matrix_b[0][0] += matrices[i_grain][1][0] * matrices[i_grain][1][0]; // SUM(l^2)
                  sum_matrix_b[1][1] += matrices[i_grain][1][1] * matrices[i_grain][1][1]; // SUM(m^2)
                  sum_matrix_b[2][2] += matrices[i_grain][1][2] * matrices[i_grain][1][2]; // SUM(n^2)
                  sum_matrix_b[0][1] += matrices[i_grain][1][0] * matrices[i_grain][1][1]; // SUM(l*m)
                  sum_matrix_b[0][2] += matrices[i_grain][1][0] * matrices[i_grain][1][2]; // SUM(l*n)
                  sum_matrix_b[1][2] += matrices[i_grain][1][1] * matrices[i_grain][1][2]; // SUM(m*n)


                  sum_matrix_c[0][0] += matrices[i_grain][2][0] * matrices[i_grain][2][0]; // SUM(l^2)
                  sum_matrix_c[1][1] += matrices[i_grain][2][1] * matrices[i_grain][2][1]; // SUM(m^2)
                  sum_matrix_c[2][2] += matrices[i_grain][2][2] * matrices[i_grain][2][2]; // SUM(n^2)
                  sum_matrix_c[0][1] += matrices[i_grain][2][0] * matrices[i_grain][2][1]; // SUM(l*m)
                  sum_matrix_c[0][2] += matrices[i_grain][2][0] * matrices[i_grain][2][2]; // SUM(l*n)
                  sum_matrix_c[1][2] += matrices[i_grain][2][1] * matrices[i_grain][2][2]; // SUM(m*n)

                }
              const std::array<std::pair<double,Tensor<1,3,double> >, 3> eigenvectors_a = eigenvectors(sum_matrix_a, SymmetricTensorEigenvectorMethod::jacobi);
              const std::array<std::pair<double,Tensor<1,3,double> >, 3> eigenvectors_b = eigenvectors(sum_matrix_b, SymmetricTensorEigenvectorMethod::jacobi);
              const std::array<std::pair<double,Tensor<1,3,double> >, 3> eigenvectors_c = eigenvectors(sum_matrix_c, SymmetricTensorEigenvectorMethod::jacobi);

              / *
              std::cout << "old eigen_vector_array_a = ";
              for (size_t i = 0; i < eigenvectors_a.size(); i++)
              {
                std::cout << eigenvectors_a[0].second[i] << " ";
              }
              std::cout << std::endl;
              * /

              // create shorcuts
              const Tensor<1,3,double> &averaged_a = eigenvectors_a[0].second;
              const Tensor<1,3,double> &averaged_b = eigenvectors_b[0].second;
              const Tensor<1,3,double> &averaged_c = eigenvectors_c[0].second;


              // todo: find out why returning a {{averaged_a[0],...},{...},{...}} does not compile.
              std::array a = {averaged_a[0],averaged_a[1],averaged_a[2]};
              std::array b = {averaged_b[0],averaged_b[1],averaged_b[2]};
              std::array c = {averaged_c[0],averaged_c[1],averaged_c[2]};

              return {a,b,c};
            }
      */
      template<int dim>
      std::vector<Tensor<2,3> >
      LpoElasticTensor<dim>::random_draw_volume_weighting(std::vector<double> fv,
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
      LpoElasticTensor<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      LpoElasticTensor<dim>::get_needed_update_flags () const
      {
        return update_default;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      LpoElasticTensor<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int> > property_information;

        //for (unsigned int i = 0; i < SymmetricTensor<2,6>::n_independent_components ; ++i)
        property_information.push_back(std::make_pair("lpo_elastic_tensor",SymmetricTensor<2,6>::n_independent_components));

        return property_information;
      }

      template <int dim>
      void
      LpoElasticTensor<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoElasticTensor");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducable as long as the problem is run with the "
                                 "same amount of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");


              prm.declare_entry ("Volume fraction olivine", "0.5",
                                 Patterns::Double(0),
                                 "The volume fraction of the olivine phase (0 is no olivine, 1 is fully olivine). "
                                 "The rest of the volume fraction is set to be entstatite. "
                                 "Todo: if full olivine make not enstite grains and vice-versa.");

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
      LpoElasticTensor<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LpoElasticTensor");
            {

              random_number_seed = prm.get_integer ("Random number seed"); // 2
              n_grains = LPO<dim>::get_number_of_grains();
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(LpoElasticTensor,
                                        "lpo elastic tensor",
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

