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
        stiffness_matrix_olivine[1][0] = stiffness_matrix_olivine[0][1];
        stiffness_matrix_olivine[1][1] = 197.25;
        stiffness_matrix_olivine[1][2] = 74.8;
        stiffness_matrix_olivine[2][0] = stiffness_matrix_olivine[0][2];
        stiffness_matrix_olivine[2][1] = stiffness_matrix_olivine[1][2];
        stiffness_matrix_olivine[2][2] = 234.32;
        stiffness_matrix_olivine[3][3] = 63.77;
        stiffness_matrix_olivine[4][4] = 77.67;
        stiffness_matrix_olivine[5][5] = 78.36;

        // Stiffness matrix for Enstatite (GPa)
        stiffness_matrix_enstatite[0][0] = 236.9;
        stiffness_matrix_enstatite[0][1] = 79.6;
        stiffness_matrix_enstatite[0][2] = 63.2;
        stiffness_matrix_enstatite[1][0] = stiffness_matrix_enstatite[0][1];
        stiffness_matrix_enstatite[1][1] = 180.5;
        stiffness_matrix_enstatite[1][2] = 56.8;
        stiffness_matrix_enstatite[2][0] = stiffness_matrix_enstatite[0][2];
        stiffness_matrix_enstatite[2][1] = stiffness_matrix_enstatite[1][2];
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

        const Particle::Property::Manager<dim> &manager = this->get_particle_world().get_property_manager();
        AssertThrow(manager.plugin_name_exists("lpo"),
                    ExcMessage("No lpo property plugin found."));
        Assert(manager.plugin_name_exists("lpo elastic tensor"),
               ExcMessage("No s wave anisotropy property plugin found."));

        AssertThrow(manager.check_plugin_order("lpo","lpo elastic tensor"),
                    ExcMessage("To use the lpo elastic tensor plugin, the lpo plugin need to be defined before this plugin."));

        lpo_data_position = manager.get_data_info().get_position_by_plugin_index(manager.get_plugin_index_by_name("lpo"));


      }





      template <int dim>
      Tensor<2,6>
      LpoElasticTensor<dim>::compute_elastic_tensor (double volume_fraction_olivine,
                                                     std::vector<double> &volume_fractions_olivine,
                                                     std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                                     std::vector<double> &volume_fractions_enstatite,
                                                     std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite) const
      {


        Tensor<4,3,double> Cav;
        Tensor<4,3,double> C0;
        // add olivine to the matrix
        // compute C0 for olivine
        //std::cout << "C0:" << std::endl;
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

        // Average stiffness matrix
        Tensor<2,6> Sav;
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

        Tensor<2,6> S_average = compute_elastic_tensor(volume_fraction_olivine,volume_fractions_olivine,
                                                       a_cosine_matrices_olivine,
                                                       volume_fractions_enstatite,
                                                       a_cosine_matrices_enstatite);


        std::array<std::array<double,3>,3> s_wave_anisotropy_olivine = compute_s_wave_anisotropy(a_cosine_matrices_olivine);
        std::array<std::array<double,3>,3> s_wave_anisotropy_enstatite = compute_s_wave_anisotropy(a_cosine_matrices_enstatite);

        // olivine
        for (unsigned int i = 0; i < 3; i++)
          for (unsigned int j = 0; j < 3; j++)
            data.push_back(s_wave_anisotropy_olivine[i][j]);

        // enstatite
        for (unsigned int i = 0; i < 3; i++)
          for (unsigned int j = 0; j < 3; j++)
            data.push_back(s_wave_anisotropy_enstatite[i][j]);
      }

      template <int dim>
      void
      LpoElasticTensor<dim>::update_one_particle_property(const unsigned int data_position,
                                                          const Point<dim> &position,
                                                          const Vector<double> &solution,
                                                          const std::vector<Tensor<1,dim> > &gradients,
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


        //std::cout << "new: n_grains = " << n_grains << ", n_samples = " << n_samples << std::endl;
        std::vector<Tensor<2,3> > weighted_olivine_a_matrices = random_draw_volume_weighting(volume_fractions_olivine, a_cosine_matrices_olivine, n_samples);
        std::vector<Tensor<2,3> > weighted_enstatite_a_matrices = random_draw_volume_weighting(volume_fractions_enstatite, a_cosine_matrices_enstatite, n_samples);

        /*std::cout << "    new weighted_olivine_a_matrices:" << std::endl;
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
        std::array<std::array<double,3>,3> s_wave_anisotropy_olivine = compute_s_wave_anisotropy(weighted_olivine_a_matrices);
        std::array<std::array<double,3>,3> s_wave_anisotropy_enstatite = compute_s_wave_anisotropy(weighted_enstatite_a_matrices);

        unsigned int counter = 0;
        for (unsigned int i = 0; i < 3; i++)
          for (unsigned int j = 0; j < 3; j++)
            {
              //std::cout << ">>> bingham: " << i << ":" << j << " old = " << data[data_position + counter] << ", new = "<< s_wave_anisotropy_olivine[i][j] << std::endl;
              data[data_position + counter] = s_wave_anisotropy_olivine[i][j];
              counter++;
            }

        for (unsigned int i = 0; i < 3; i++)
          for (unsigned int j = 0; j < 3; j++)
            {
              //std::cout << ">>> bingham: "  << i << ":" << j << " old = " << data[data_position + counter] << ", new = " << s_wave_anisotropy_olivine[i][j] << std::endl;
              data[data_position + counter] = s_wave_anisotropy_enstatite[i][j];
              counter++;
            }


      }


      template<int dim>
      std::array<std::array<double,3>,3>
      LpoElasticTensor<dim>::compute_s_wave_anisotropy(std::vector<Tensor<2,3> > matrices) const
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

        /*
        std::cout << "old eigen_vector_array_a = ";
        for (size_t i = 0; i < eigenvectors_a.size(); i++)
        {
          std::cout << eigenvectors_a[0].second[i] << " ";
        }
        std::cout << std::endl;
        */

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

        property_information.push_back(std::make_pair("lpo_elastic_tensor average olivine a axis",3));
        property_information.push_back(std::make_pair("lpo_elastic_tensor average olivine b axis",3));
        property_information.push_back(std::make_pair("lpo_elastic_tensor average olivine c axis",3));

        property_information.push_back(std::make_pair("lpo_elastic_tensor average enstatite a axis",3));
        property_information.push_back(std::make_pair("lpo_elastic_tensor average enstatite b axis",3));
        property_information.push_back(std::make_pair("lpo_elastic_tensor average enstatite c axis",3));

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

              prm.declare_entry ("Nucliation efficientcy", "5",
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
              mobility = prm.get_double("Mobility"); //50;
              x_olivine = prm.get_double("Volume fraction olivine"); // 0.5;
              stress_exponent = prm.get_double("Stress exponents"); //3.5;
              exponent_p = prm.get_double("Exponents p"); //1.5;
              nucliation_efficientcy = prm.get_double("Nucliation efficientcy"); //5;
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

