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


        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_tensor);

        double elastic_vector_norm = elastic_vector.norm();

        Tensor<2,3> unprojected_SCC = compute_unprojected_SCC(elastic_tensor);

        std::pair<Tensor<2,6>,Tensor<2,3> > minimum_hexagonal_projection = compute_minimum_hexagonal_projection(unprojected_SCC, elastic_tensor, elastic_vector_norm);


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

        return Tensor<2,3>(
        {
          {eigenvectors_a[0].second[0],eigenvectors_a[0].second[1],eigenvectors_a[0].second[2]},
          {eigenvectors_a[1].second[1],eigenvectors_a[1].second[1],eigenvectors_a[1].second[2]},
          {eigenvectors_a[2].second[2],eigenvectors_a[2].second[1],eigenvectors_a[2].second[2]}
        });
      }


      template<int dim>
      std::pair<SymmetricTensor<2,6>,Tensor<2,3> >
      LpoHexagonalAxes<dim>::compute_minimum_hexagonal_projection(
        const Tensor<2,3> &unprojected_SCC,
        const SymmetricTensor<2,6> &elastic_tensor,
        const double elastic_vector_norm) const
      {
        Tensor<1,21> elastic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(elastic_tensor);
        double lowest_norm = elastic_vector_norm;
        unsigned short int lowest_norm_permutation = 99;


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
        SymmetricTensor<2,6> projected_elastic_matrix;
        for (unsigned short int i = 0; i < 3; i++)
          {
            std::array<unsigned short int, 3> perumation = indexed_permutation(i);


            for (size_t j = 0; j < 3; j++)
              {
                projected_SCC[i][j] = unprojected_SCC[perumation[j]];
              }

            projected_elastic_matrix = LpoElasticTensor<dim>::rotate_6x6_matrix(elastic_tensor,(projected_SCC[i]));

            const Tensor<1,21> projected_elatic_vector = LpoElasticTensor<dim>::transform_6x6_matrix_to_21D_vector(projected_elastic_matrix);

            const Tensor<1,9> hexagonal_elastic_vector = project_onto_hexagonal_symmetry(projected_elatic_vector);

            Tensor<1,21> elastic_vector_tmp = elastic_vector;

            // now compute how much is left over in the origional elastic vector
            for (size_t i = 0; i < 9; i++)
              {
                elastic_vector_tmp[i] - hexagonal_elastic_vector[i];
              }

            const double current_norm = elastic_vector_tmp.norm();

            if (current_norm < lowest_norm)
              {
                lowest_norm = current_norm;
                lowest_norm_permutation = i;
              }
          }

        AssertThrow(lowest_norm_permutation < 3,
                    ExcMessage("LPO Hexagonal axes plugin could not find a good hexagonal projection."));


        return std::make_pair(projected_elastic_matrix,projected_SCC[lowest_norm_permutation]);
      }


      template<int dim>
      Tensor<1,9>
      LpoHexagonalAxes<dim>::project_onto_hexagonal_symmetry(const Tensor<1,21> &elastic_vector) const
      {
        return Tensor<1,9>(
        {
          0.375 * (elastic_vector[0] + elastic_vector[1]) + std::sqrt(2) * 0.25 * elastic_vector[5] + 0.25 * elastic_vector[8],              // 0 // 1
          0.375 * (elastic_vector[0] + elastic_vector[1]) + std::sqrt(2) * 0.25 * elastic_vector[5] + 0.25 * elastic_vector[8],              // 1 // 2
          elastic_vector[2],                                                                                                                 // 2 // 3
          0.5 * (elastic_vector[3] + elastic_vector[4]),                                                                                     // 3 // 4
          0.5 * (elastic_vector[3] + elastic_vector[4]),                                                                                     // 4 // 5
          std::sqrt(2) * 0.25 * (elastic_vector[0] + elastic_vector[1]) + 0.75 * elastic_vector[5] - std::sqrt(2) * 0.5 * elastic_vector[8], // 5 // 6
          0.5 * (elastic_vector[6] + elastic_vector[8]),                                                                                     // 6 // 7
          0.5 * (elastic_vector[6] + elastic_vector[8]),                                                                                     // 7 // 8
          0.25 * (elastic_vector[0] + elastic_vector[1]) - std::sqrt(2) * 0.5 * elastic_vector[5] + 0.5 * elastic_vector[8]                  // 8 // 9
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

