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

      }



      template <int dim>
      void
      LpoHexagonalAxes<dim>::initialize_one_particle_property(const Point<dim> &,
                                                              std::vector<double> &data) const
      {


        /*double water_content = 0;
        double volume_fraction_olivine = 0;
        std::vector<double> volume_fractions_olivine(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
        std::vector<double> volume_fractions_enstatite(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);*/
        SymmetricTensor<2,6> elastic_tensor;

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
                                                                      elastic_tensor);



        // These eigenvectors uniquely define the symmetry cartesian coordiante system (SCCS),
        // but we need to find the order in which


        Tensor<2,3> unprojected_SCC = compute_unprojected_SCC(elastic_tensor);

        Tensor<2,3> minimum_SCC_projection = compute_minimum_SCC_projection(unprojected_SCC, elastic_tensor);


        //compute_hexagonal_axes(Tensor<2,6> &elastic_tensor);



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
                std::array<std::array<double,3>,3> hexagonal_axes = compute_hexagonal_axes(elastic_tensor);

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
        //std::array<std::array<double,3>,3> hexagonal_axes_olivine = compute_hexagonal_axes(weighted_olivine_a_matrices);
        //std::array<std::array<double,3>,3> hexagonal_axes_enstatite = compute_hexagonal_axes(weighted_enstatite_a_matrices);
        /*
                unsigned int counter = 0;
                for (unsigned int i = 0; i < 3; i++)
                  for (unsigned int j = 0; j < 3; j++)
                    {
                      //std::cout << ">>> bingham: " << i << ":" << j << " old = " << data[data_position + counter] << ", new = "<< hexagonal_axes_olivine[i][j] << std::endl;
                      data[data_position + counter] = hexagonal_axes_olivine[i][j];
                      counter++;
                    }

                for (unsigned int i = 0; i < 3; i++)
                  for (unsigned int j = 0; j < 3; j++)
                    {
                      //std::cout << ">>> bingham: "  << i << ":" << j << " old = " << data[data_position + counter] << ", new = " << hexagonal_axes_olivine[i][j] << std::endl;
                      data[data_position + counter] = hexagonal_axes_enstatite[i][j];
                      counter++;
                    }*/


      }

      template<int dim>
      std::array<unsigned short int, 3>
      LpoHexagonalAxes<dim>::indexed_permutation(const unsigned short int index) const
      {
        switch (index)
          {
            case 0 :
              return {0,1,2};
            case 1 :
              return {1,2,0};
            case 2 :
              return {2,0,1};
            default:
              AssertThrow(false,ExcMessage("Provided index larger then 2 (" + std::to_string(index)+ ")."));
              return {0,0,0};
          }

      }

      template<int dim>
      Tensor<4,3>
      LpoHexagonalAxes<dim>::rotate_4th_order_tensor(const Tensor<4,3> &input_tensor, const Tensor<2,3> &rotation_tensor) const
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
      LpoHexagonalAxes<dim>::transform_4th_order_tensor_to_6x6_matrix(const Tensor<4,3> &input_tensor) const
      {
        SymmetricTensor<2,6> output;

        for (unsigned short int i = 0; i < 3; i++)
          {
            output[i][i] = input_tensor[i][i][i][i];
          }

        for (unsigned short int i = 1; i < 3; i++)
          {
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
            output[i][5]=0.25*(input_tensor[i][i][0][1]+input_tensor[i][i][1][2]+input_tensor[0][1][i][i]+input_tensor[1][0][i][i]);
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
      LpoHexagonalAxes<dim>::transform_6x6_matrix_to_4th_order_tensor(const SymmetricTensor<2,6> &input_tensor) const
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

      }

      template<int dim>
      Tensor<1,21>
      LpoHexagonalAxes<dim>::transform_6x6_matrix_to_21D_vector(const SymmetricTensor<2,6> &input) const
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
      Tensor<1,21>
      LpoHexagonalAxes<dim>::transform_4th_order_tensor_to_21D_vector(const Tensor<4,3> &input_tensor) const
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
          0.5*(input_tensor[0][0][1][2]+input_tensor[0][0][2][1]+ input_tensor[1][2][0][0]+input_tensor[2][1][0][0]),         // 9  // 10
          0.5*(input_tensor[1][1][0][2]+input_tensor[1][1][2][0]+ input_tensor[0][2][1][1]+input_tensor[2][0][1][1]),         // 10 // 11
          0.5*(input_tensor[2][2][0][1]+input_tensor[2][2][1][2]+input_tensor[0][1][2][2]+input_tensor[1][0][2][2]),         // 11 // 12
          0.5*(input_tensor[2][2][1][2]+input_tensor[2][2][2][1]+ input_tensor[1][2][2][2]+input_tensor[2][1][2][2]),         // 12 // 13
          0.5*(input_tensor[0][0][0][2]+input_tensor[0][0][2][0]+ input_tensor[0][2][0][0]+input_tensor[2][0][0][0]),         // 13 // 14
          0.5*(input_tensor[1][1][0][1]+input_tensor[1][1][1][2]+input_tensor[0][1][1][1]+input_tensor[1][0][1][1]),         // 14 // 15
          0.5*(input_tensor[1][1][1][2]+input_tensor[1][1][2][1]+ input_tensor[1][2][1][1]+input_tensor[2][1][1][1]),         // 15 // 16
          0.5*(input_tensor[2][2][0][2]+input_tensor[2][2][2][0]+ input_tensor[0][2][2][2]+input_tensor[2][0][2][2]),         // 16 // 17
          0.5*(input_tensor[0][0][0][1]+input_tensor[0][0][1][2]+input_tensor[0][1][0][0]+input_tensor[1][0][0][0]),         // 17 // 18
          sqrt(2)*0.25*(input_tensor[0][2][0][1]+input_tensor[0][2][1][0]+input_tensor[2][0][0][1]+input_tensor[2][0][1][0]+input_tensor[0][1][0][2]+input_tensor[0][1][2][0]+input_tensor[1][0][0][2]+input_tensor[1][0][2][0]), // 18 // 19
          sqrt(2)*0.25*(input_tensor[1][2][0][1]+input_tensor[1][2][1][0]+input_tensor[2][1][0][1]+input_tensor[2][1][1][0]+input_tensor[0][1][1][2]+input_tensor[0][1][2][1]+input_tensor[1][0][1][2]+input_tensor[1][0][2][1]), // 19 // 20
          sqrt(2)*0.25*(input_tensor[1][2][0][2]+input_tensor[1][2][2][0]+input_tensor[2][1][0][2]+input_tensor[2][1][2][0]+input_tensor[0][2][1][2]+input_tensor[0][2][2][1]+input_tensor[2][0][1][2]+input_tensor[2][0][2][1])  // 20 // 21
        });

      }


      template<int dim>
      Tensor<2,3> 
      LpoHexagonalAxes<dim>::compute_unprojected_SCC(const SymmetricTensor<2,6> &elastic_tensor) const
      {
        /**
         * We are going to use the Voigt stiffness tensor in this case since it is slightly cheaper to compute.
         * We could also use the dilatational stiffness tensor, or the average eigenvectors
         * of both, but since with a Orthohombic symmetry assumption they should be the same.
         */
        SymmetricTensor<2,3> voigt_stiffness_tensor;


        voigt_stiffness_tensor[0][0]=elastic_tensor[0][0]+elastic_tensor[5][5]+elastic_tensor[4][4];
        voigt_stiffness_tensor[1][1]=elastic_tensor[5][5]+elastic_tensor[1][1]+elastic_tensor[3][3];
        voigt_stiffness_tensor[2][2]=elastic_tensor[4][4]+elastic_tensor[3][3]+elastic_tensor[2][2];
        voigt_stiffness_tensor[1][0]=elastic_tensor[0][5]+elastic_tensor[1][5]+elastic_tensor[3][4];
        voigt_stiffness_tensor[2][0]=elastic_tensor[0][4]+elastic_tensor[2][4]+elastic_tensor[3][5];
        voigt_stiffness_tensor[2][1]=elastic_tensor[1][3]+elastic_tensor[2][4]+elastic_tensor[4][5];

        // computing the eigenvector of this matrix
        const std::array<std::pair<double,Tensor<1,3,double> >, 3> eigenvectors_a = eigenvectors(voigt_stiffness_tensor, SymmetricTensorEigenvectorMethod::jacobi);

        return Tensor<2,3>({
          {eigenvectors_a[0].second[0],eigenvectors_a[0].second[1],eigenvectors_a[0].second[2]},
          {eigenvectors_a[1].second[1],eigenvectors_a[1].second[1],eigenvectors_a[1].second[2]},
          {eigenvectors_a[2].second[2],eigenvectors_a[2].second[1],eigenvectors_a[2].second[2]}});
      }


      template<int dim>
      Tensor<2,3> 
      LpoHexagonalAxes<dim>::compute_minimum_SCC_projection(const Tensor<2,3> &unprojected_SCC, const SymmetricTensor<2,6> &elastic_tensor) const
      {
        return unprojected_SCC;
      }


      template<int dim>
      std::array<std::array<double,3>,3>
      LpoHexagonalAxes<dim>::compute_hexagonal_axes(Tensor<2,6> &elastic_tensor) const
      {
        // todo: find out why returning a {{averaged_a[0],...},{...},{...}} does not compile.


        std::array a = {0,0,0};
        std::array b = {0,0,0};
        std::array c = {0,0,0};

        //return {a,b,c};

        return std::array<std::array<double,3>,3>();
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

        property_information.push_back(std::make_pair("lpo_hexagonal_axes average olivine a axis",3));
        property_information.push_back(std::make_pair("lpo_hexagonal_axes average olivine b axis",3));
        property_information.push_back(std::make_pair("lpo_hexagonal_axes average olivine c axis",3));

        property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite a axis",3));
        property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite b axis",3));
        property_information.push_back(std::make_pair("lpo_hexagonal_axes average enstatite c axis",3));

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

