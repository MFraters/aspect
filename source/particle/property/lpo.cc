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
#include <aspect/particle/property/lpo.h>

#include <aspect/utilities.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      template <int dim>
      unsigned int LPO<dim>::n_grains = 0;

      template <int dim>
      LPO<dim>::LPO ()
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
      LPO<dim>::initialize ()
      {
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);
        // todo: check wheter this works correctly. Since the get_random_number function takes a reference
        // to the random_number_generator function, changing the function should mean that I have to update the
        // get_random_number function as well. But I will need to test this.
      }

      template <int dim>
      void
      LPO<dim>::load_lpo_particle_data(unsigned int lpo_data_position,
                                       const ArrayView<double> &data,
                                       unsigned int n_grains,
                                       std::vector<double> &volume_fractions_olivine,
                                       std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                       std::vector<double> &volume_fractions_enstatite,
                                       std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite)
      {
        // resize the vectors to fit n_grains
        volume_fractions_olivine.resize(n_grains);
        a_cosine_matrices_olivine.resize(n_grains);
        volume_fractions_enstatite.resize(n_grains);
        a_cosine_matrices_enstatite.resize(n_grains);

        // loop over grain retrieve from data from each grain
        unsigned int data_grain_i = 0;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // retrieve volume fraction for olvine grains
            volume_fractions_olivine[grain_i] = data[lpo_data_position + data_grain_i *
                                                     (Tensor<2,3>::n_independent_components + 1) + 1];

            // retrieve a_{ij} for olvine grains
            //Tensor<2,dim> a_cosine_matrices_olivine;
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                a_cosine_matrices_olivine[grain_i][index] = data[lpo_data_position + data_grain_i *
                                                                 (Tensor<2,3>::n_independent_components + 1) + 2 + i];
              }

            // retrieve volume fraction for enstatite grains
            volume_fractions_enstatite[grain_i] = data[lpo_data_position + (data_grain_i+1) *
                                                       (Tensor<2,3>::n_independent_components + 1) + 1];

            // retrieve a_{ij} for enstatite grains
            //Tensor<2,dim> a_cosine_matrices;
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                a_cosine_matrices_enstatite[grain_i][index] = data[lpo_data_position + (data_grain_i+1) *
                                                                   (Tensor<2,3>::n_independent_components + 1) + 2 + i];
              }
            data_grain_i = data_grain_i + 2;
          }
      }


      template <int dim>
      void
      LPO<dim>::store_lpo_particle_data(unsigned int lpo_data_position,
                                        const ArrayView<double> &data,
                                        unsigned int n_grains,
                                        std::vector<double> &volume_fractions_olivine,
                                        std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                        std::vector<double> &volume_fractions_enstatite,
                                        std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite)
      {
        Assert(volume_fractions_olivine.size() == n_grains, ExcMessage("Internal error: volume_fractions_olivine is not the same as n_grains."));
        Assert(a_cosine_matrices_olivine.size() == n_grains, ExcMessage("Internal error: a_cosine_matrices_olivine is not the same as n_grains."));
        Assert(volume_fractions_enstatite.size() == n_grains, ExcMessage("Internal error: volume_fractions_enstatite is not the same as n_grains."));
        Assert(a_cosine_matrices_enstatite.size() == n_grains, ExcMessage("Internal error: a_cosine_matrices_enstatite is not the same as n_grains."));

        // loop over grains to store the data of each grain
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // store volume fraction for olvine grains
            data[lpo_data_position + grain_i * 2.0 * (Tensor<2,3>::n_independent_components + 1) + 1] = volume_fractions_olivine[grain_i];

            // store a_{ij} for olvine grains
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                data[lpo_data_position + grain_i * 2.0 * (Tensor<2,3>::n_independent_components + 1) + 2 + i] = a_cosine_matrices_olivine[grain_i][index];
              }

            // store volume fraction for enstatite grains
            data[lpo_data_position + grain_i * 2.0 * (Tensor<2,3>::n_independent_components + 1) + 11] = volume_fractions_enstatite[grain_i];

            // store a_{ij} for enstatite grains
            for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
              {
                const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                data[lpo_data_position + grain_i * 2.0 * (Tensor<2,3>::n_independent_components + 1) + 12 + i] = a_cosine_matrices_enstatite[grain_i][index];
              }
          }
      }

      template <int dim>
      void
      LPO<dim>::initialize_one_particle_property(const Point<dim> &,
                                                 std::vector<double> &data) const
      {
        // the layout of the data vector per perticle is the following (note that for this plugin the following dim's are always 3):
        // 1. water content -> 1 double, always at location data_position -> i = 0;
        // 2. N grains times:
        //    2.1. volume fraction olivine -> 1 double, at location:
        //                                    data_position + i_grain * 20 + 1, or
        //                                    data_position + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + 1
        //    2.2. a_cosine_matrix olivine -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
        //                                    data_position + i_grain * 20 + 2, or
        //                                    data_position + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + 2
        //    2.3. volume fraction enstatite -> 1 double, at location:
        //                                      data_position + i_grain * 20 + 11, or
        //                                      data_position + i_grain * (Tensor<2,3>::n_independent_components + 2)  + 11
        //    2.4. a_cosine_matrix enstatite -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
        //                                      data_position + i_grain * 20 + 12, or
        //                                      data_position + i_grain * (Tensor<2,3>::n_independent_components + 2) + 12
        // We store it this way because this is also the order in which it is read, so this should
        // theoretically minimize chache misses. Note: It has not been tested wheter this is faster then storing it in another way.
        //
        // An other note is that we store exactly the same amount of olivine and enstatite grain, although
        // the volume may not be the same. This has to do with that we need a minimum amount of grains
        // per tracer to perform reliable statistics on it. This miminum is the same for both olivine and
        // enstatite.

        // set water content
        // for now no water. Later it might be composition dependent or set by a function.
        data.push_back(0.0);

        // set volume fraction
        const double initial_volume_fraction = 1.0/n_grains;
        //for (unsigned int i = 0; i < n_grains ; ++i)
        //data.push_back(initial_volume_fraction);

        boost::random::uniform_real_distribution<double> uniform_distribution(0,1);
        double two_pi = 2.0 * M_PI;
        std::vector<Tensor<2,3>> a_cosine_matrix(2*n_grains,Tensor<2,3>());

        // it is 2 times the amount of grains because we have to compute for olivine and enstatite.
        for (unsigned int i_grain = 0; i_grain < 2 * n_grains ; ++i_grain)
          {
            // set volume fraction
            data.push_back(initial_volume_fraction);

            // set a uniform random a_cosine_matrix per grain
            // This function is based on an article in Graphic Gems III, written by James Arvo, Cornell University (p 116-120).
            // The original code can be found on  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
            // andis licenend accourding to this website with the following licence:
            //
            // "The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your own and
            // resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial. Giving credit
            // is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems with any Gems
            // code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held responsible. Basically,
            // don't be a jerk, and remember that anything free comes with no guarantee.""
            //
            // The book saids in the preface the following: "As in the first two volumes, all of the C and C++ code in this book is in
            // the public domain, and is yours to study, modify, and use."

            // first generate three random numbers between 0 and 1 and multiply them with 2 PI or 2 for z. Note that these are not the same as phi_1, theta and phi_2.
            double theta = two_pi * uniform_distribution(this->random_number_generator); // Rotation about the pole (Z)
            double phi = two_pi * uniform_distribution(this->random_number_generator); // For direction of pole deflection.
            double z = 2.0* uniform_distribution(this->random_number_generator); //For magnitude of pole deflection.

            // Compute a vector V used for distributing points over the sphere
            // via the reflection I - V Transpose(V).  This formulation of V
            // will guarantee that if x[1] and x[2] are uniformly distributed,
            // the reflected points will be uniform on the sphere.  Note that V
            // has length sqrt(2) to eliminate the 2 in the Householder matrix.

            double r  = std::sqrt( z );
            double Vx = std::sin( phi ) * r;
            double Vy = std::cos( phi ) * r;
            double Vz = std::sqrt( 2.f - z );

            // Compute the row vector S = Transpose(V) * R, where R is a simple
            // rotation by theta about the z-axis.  No need to compute Sz since
            // it's just Vz.

            double st = std::sin( theta );
            double ct = std::cos( theta );
            double Sx = Vx * ct - Vy * st;
            double Sy = Vx * st + Vy * ct;

            // Construct the rotation matrix  ( V Transpose(V) - I ) R, which
            // is equivalent to V S - R.

            a_cosine_matrix[i_grain][0][0] = Vx * Sx - ct;
            a_cosine_matrix[i_grain][0][1] = Vx * Sy - st;
            a_cosine_matrix[i_grain][0][2] = Vx * Vz;

            a_cosine_matrix[i_grain][1][0] = Vy * Sx + st;
            a_cosine_matrix[i_grain][1][1] = Vy * Sy - ct;
            a_cosine_matrix[i_grain][1][2] = Vy * Vz;

            a_cosine_matrix[i_grain][2][0] = Vz * Sx;
            a_cosine_matrix[i_grain][2][1] = Vz * Sy;
            a_cosine_matrix[i_grain][2][2] = 1.0 - z;   // This equals Vz * Vz - 1.0


            for (unsigned int i = 0; i < Tensor<2,dim>::n_independent_components ; ++i)
              data.push_back(a_cosine_matrix[i_grain][Tensor<2,dim>::unrolled_to_component_indices(i)]);
          }
      }

      template<int dim>
      std::vector<Tensor<2,3> >
      LPO<dim>::random_draw_volume_weighting(std::vector<double> fv,
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
      void
      LPO<dim>::update_one_particle_property(const unsigned int data_position,
                                             const Point<dim> &position,
                                             const Vector<double> &solution,
                                             const std::vector<Tensor<1,dim> > &gradients,
                                             const ArrayView<double> &data) const
      {
        // Load data which is needed

        // need access to the pressure, viscosity,
        // get velocity
        Tensor<1,dim> velocity;
        for (unsigned int i = 0; i < dim; ++i)
          velocity[i] = solution[this->introspection().component_indices.velocities[i]];

        // if the velocity is zero, we do not need to update because
        // no LPO is formed.
        if (velocity.norm() < 1e-15)
          return;


        // get velocity gradient tensor.
        Tensor<2,dim> grad_u;
        for (unsigned int d=0; d<dim; ++d)
          grad_u[d] = gradients[d];


        // Calculate strain rate from velocity gradients
        const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);

        // compute local stress tensor
        const SymmetricTensor<2,dim> compressible_strain_rate
          = (this->get_material_model().is_compressible()
             ?
             strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
             :
             strain_rate);


        // To determine the deformation type of grains based on figure 4 of Karato et al.,
        // 2008 (Geodynamic Significance of seismic anisotropy o fthe Upper Mantle: New
        // insights from laboratory studies), we need to know the stress and water content.
        // The water content is stored on every particle and the stress is computed here.
        // To compute the stress we need the pressure, the compressible_strain_rate and the
        // viscosity at the location of the particle.

        double pressure = solution[this->introspection().component_indices.pressure];
        double temperature = solution[this->introspection().component_indices.temperature];
        double water_content = data[data_position];
        const double dt = this->get_timestep();
        const double strain_rate_second_invariant = std::sqrt(std::abs(second_invariant(strain_rate)));
        Assert(!std::isnan(strain_rate_second_invariant), ExcMessage("The second invariant of the strain rate is not a number."));


        // get the composition of the particle
        std::vector<double> compositions;
        for (unsigned int i = 0; i < this->n_compositional_fields(); i++)
          {
            const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
            compositions.push_back(solution[solution_component]);
          }

        // construct the material model inputs and outputs
        // Since this function is only evaluating one particle,
        // we use 1 for the amount of quadrature points.
        MaterialModel::MaterialModelInputs<dim> material_model_inputs(1,this->n_compositional_fields());
        material_model_inputs.position[0] = position;
        material_model_inputs.temperature[0] = temperature;
        material_model_inputs.pressure[0] = pressure;
        material_model_inputs.velocity[0] = velocity;
        material_model_inputs.composition[0] = compositions;
        material_model_inputs.strain_rate[0] = strain_rate;

        MaterialModel::MaterialModelOutputs<dim> material_model_outputs(1,this->n_compositional_fields());
        this->get_material_model().evaluate(material_model_inputs, material_model_outputs);
        double eta = material_model_outputs.viscosities[0];

        const SymmetricTensor<2,dim> stress = 2*eta*compressible_strain_rate +
                                              pressure * unit_symmetric_tensor<dim>();

        // Now compute what type of deformation takes place. For now just use type A
        DeformationType deformation_type = DeformationType::A_type;

        const std::array<double,4> ref_resolved_shear_stress = reference_resolved_shear_stress_from_deformation_type(deformation_type);

        std::vector<double> volume_fractions_olivine(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
        std::vector<double> volume_fractions_enstatite(n_grains);
        std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);

        //std::cout << "data_position = " << data_position << ", n_grains" << n_grains << std::endl;
        load_lpo_particle_data(data_position,
                               data,
                               n_grains,
                               volume_fractions_olivine,
                               a_cosine_matrices_olivine,
                               volume_fractions_enstatite,
                               a_cosine_matrices_enstatite);

        // Make the strain-rate and velocity gradient tensor non-dimensional
        // by dividing it through the second invariant
        const SymmetricTensor<2,dim> strain_rate_nondimensional = strain_rate / strain_rate_second_invariant;
        const Tensor<2,dim> velocity_gradient_nondimensional = grad_u / strain_rate_second_invariant;

        // Now we have loaded all the data and can do the actual computation.
        //
        // Use a 4th order R-K integration to calculate the change in the
        // direction cosines matrix and the volume (call function to calculate
        // derivatives 4 times). dt is the time-step, w_i is the solution from
        // the previous time-setp (or the inital condition).
        // new solution w_{i+1} = w_i + (1/6)*(k1 + 2.0 * k2 + 2.0 * k3 + k4)
        const double sum_volume_olivine = this->compute_runge_kutta(volume_fractions_olivine, a_cosine_matrices_olivine,
                                                                    strain_rate_nondimensional, velocity_gradient_nondimensional,
                                                                    deformation_type, ref_resolved_shear_stress,
                                                                    strain_rate_second_invariant, dt);

        const double sum_volume_enstatite = this->compute_runge_kutta(volume_fractions_enstatite, a_cosine_matrices_enstatite,
                                                                      strain_rate_nondimensional, velocity_gradient_nondimensional,
                                                                      deformation_type, ref_resolved_shear_stress,
                                                                      strain_rate_second_invariant, dt);

        // normalize both the olivine and enstatite volume fractions
        const double inv_sum_volume_olivine = 1/sum_volume_olivine;
        const double inv_sum_volume_enstatite = 1/sum_volume_enstatite;

        Assert(std::isfinite(inv_sum_volume_olivine),
               ExcMessage("inv_sum_volume_olivine is not finite. sum_volume_enstatite = "
                          + std::to_string(sum_volume_olivine)));
        Assert(std::isfinite(inv_sum_volume_enstatite),
               ExcMessage("inv_sum_volume_enstatite is not finite. sum_volume_enstatite = "
                          + std::to_string(sum_volume_enstatite)));

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            volume_fractions_olivine[grain_i] *= inv_sum_volume_olivine;
            Assert(isfinite(volume_fractions_olivine[grain_i]),
                   ExcMessage("volume_fractions_olivine[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(volume_fractions_olivine[grain_i]) + ", inv_sum_volume_olivine = "
                              + std::to_string(inv_sum_volume_olivine) + "."));

            volume_fractions_enstatite[grain_i] *= inv_sum_volume_enstatite;
            Assert(isfinite(volume_fractions_enstatite[grain_i]),
                   ExcMessage("volume_fractions_enstatite[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(volume_fractions_enstatite[grain_i]) + ", inv_sum_volume_enstatite = "
                              + std::to_string(inv_sum_volume_enstatite) + "."));
          }

        /**
         * Correct direction cosine matrices numerical error (orthnormality) after integration
         * Follows same method as in matlab version from Thissen of finding the nearest orthonormal
         * matrix using the SVD
         */
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // make tolerance variable
            double tolerance = 1e-10;
            orthogonalize_matrix(a_cosine_matrices_olivine[grain_i], tolerance);
            orthogonalize_matrix(a_cosine_matrices_enstatite[grain_i], tolerance);
          }

        store_lpo_particle_data(data_position,
                                data,
                                n_grains,
                                volume_fractions_olivine,
                                a_cosine_matrices_olivine,
                                volume_fractions_enstatite,
                                a_cosine_matrices_enstatite);



      }

      template<int dim>
      void
      LPO<dim>::orthogonalize_matrix(dealii::Tensor<2, 3> &tensor,
                                     double tolerance) const
      {
        if (std::abs(determinant(tensor) - 1.0) > tolerance)
          {
            LAPACKFullMatrix<double> identity_matrix(3);
            for (size_t i = 0; i < 3; i++)
              {
                identity_matrix.set(i,i,1);
              }
              
            FullMatrix<double> matrix_olivine(3);
            LAPACKFullMatrix<double> lapack_matrix_olivine(3);
            LAPACKFullMatrix<double> result(3);
            LAPACKFullMatrix<double> result2(3);

            // todo: find or add dealii functionallity to copy in one step.
            matrix_olivine.copy_from(tensor);
            lapack_matrix_olivine.copy_from(matrix_olivine);

            // now compute the svd of the matrices
            lapack_matrix_olivine.compute_svd();

            // Use the SVD results to orthogonalize: ((U*I)*V^T)^T
            lapack_matrix_olivine.get_svd_u().mmult(result,identity_matrix);
            result.mmult(result2,(lapack_matrix_olivine.get_svd_vt()));

            // todo: find or add dealii functionallity to copy in one step.
            matrix_olivine = result2;
            matrix_olivine.copy_to(tensor);
          }
      }

      template<int dim>
      std::array<double,4>
      LPO<dim>::reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type, double max_value) const
      {
        std::array<double,4> ref_resolved_shear_stress;
        switch (deformation_type)
          {
            // from Kaminski and Ribe, GJI 2004.
            case DeformationType::A_type :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 3;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004.
            case DeformationType::B_type :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 1;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004.
            case DeformationType::C_type :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = 2;
              ref_resolved_shear_stress[3] = 1;
              break;

            // from Kaminski and Ribe, GRL 2002.
            case DeformationType::D_type :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 3;
              break;

            // using this form the matlab code, not sure where it comes from
            case DeformationType::E_type :
              ref_resolved_shear_stress[0] = 2;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 3;
              break;

            // from Kaminski and Ribe, GJI 2004.
            // Todo: this one is not used in practice, since there is an optimalisation in
            // the code. So maybe remove it in the future.
            case DeformationType::enstatite :
              ref_resolved_shear_stress[0] = max_value;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 1;
              break;

            default:
              break;
          }
        return ref_resolved_shear_stress;
      }


      template<int dim>
      std::vector<std::vector<double> >
      LPO<dim>::volume_weighting(std::vector<double> fv, std::vector<std::vector<double>> angles) const
      {
        // Get volume weighted euler angles, using random draws to convert odf
        // to a discrete number of orientations, weighted by volume
        // 1a. Get index that would sort volume fractions AND
        //ix = np.argsort(fv[q,:]);
        // 1b. Get the sorted volume and angle arrays
        std::vector<double> fv_to_sort = fv;
        std::vector<double> fv_sorted = fv;
        std::vector<std::vector<double>> angles_sorted = angles;

        unsigned int n_grain = fv_to_sort.size();


        /**
         * ...
         */
        for (int i = n_grain-1; i >= 0; --i)
          {
            unsigned int index_max_fv = std::distance(fv_to_sort.begin(),max_element(fv_to_sort.begin(), fv_to_sort.end()));

            fv_sorted[i] = fv_to_sort[index_max_fv];
            angles_sorted[i] = angles[index_max_fv];
            Assert(angles[index_max_fv].size() == 3, ExcMessage("angles vector (size = " + std::to_string(angles[index_max_fv].size()) +
                                                                ") should have size 3."));
            Assert(angles_sorted[i].size() == 3, ExcMessage("angles_sorted vector (size = " + std::to_string(angles_sorted[i].size()) +
                                                            ") should have size 3."));
            fv_to_sort[index_max_fv] = -1;
          }


        // 2. Get cumulative weight for volume fraction
        std::vector<double> cum_weight(n_grains);
        std::partial_sum(fv_sorted.begin(),fv_sorted.end(),cum_weight.begin());
        // 3. Generate random indices
        boost::random::uniform_real_distribution<> dist(0, 1);
        std::vector<double> idxgrain(n_grains);
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            idxgrain[grain_i] = dist(this->random_number_generator); //random.rand(ngrains,1);
          }

        // 4. Find the maximum cum_weight that is less than the random value.
        // the euler angle index is +1. For example, if the idxGrain(g) < cumWeight(1),
        // the index should be 1 not zero)
        std::vector<std::vector<double>> angles_out(n_grains,std::vector<double>(3));
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            unsigned int counter = 0;
            for (unsigned int grain_j = 0; grain_j < n_grains-1; ++grain_j)
              {
                if (cum_weight[grain_j] < idxgrain[grain_i])
                  {
                    counter++;
                  }

                Assert(angles_sorted[counter].size() == 3, ExcMessage("angles_sorted vector (size = " + std::to_string(angles_sorted[counter].size()) +
                                                                      ") should have size 3."));
                angles_out[grain_i] = angles_sorted[counter];
                Assert(angles_out[counter].size() == 3, ExcMessage("angles_out vector (size = " + std::to_string(angles_out[counter].size()) +
                                                                   ") should have size 3."));
              }
          }
        return angles_out;
      }


      template <int dim>
      double
      LPO<dim>::wrap_angle(const double angle) const
      {
        return angle - 360.0*std::floor(angle/360.0);
      }


      template <int dim>
      std::vector<double>
      LPO<dim>::extract_euler_angles_from_dcm(const Tensor<2,3> &rotation_matrix) const
      {
        std::vector<double> euler_angles(3);
        const double s2 = std::sqrt(rotation_matrix[2][1] * rotation_matrix[2][1] + rotation_matrix[2][0] * rotation_matrix[2][0]);
        const double phi1  = std::atan2(rotation_matrix[2][0],-rotation_matrix[2][1]) * rad_to_degree;
        const double theta = std::acos(rotation_matrix[2][2]) * rad_to_degree;
        const double phi2  = std::atan2(rotation_matrix[0][2],rotation_matrix[1][2]) * rad_to_degree;

        euler_angles[0] = wrap_angle(phi1);
        euler_angles[1] = wrap_angle(theta);
        euler_angles[2] = wrap_angle(phi2);

        return euler_angles;
      }

      template <int dim>
      Tensor<2,3>
      LPO<dim>::dir_cos_matrix2(double phi1, double theta, double phi2) const
      {
        Tensor<2,3> dcm;


        dcm[0][0] = cos(phi2)*cos(phi1) - cos(theta)*sin(phi1)*sin(phi2);
        dcm[0][1] = cos(phi2)*sin(phi1) + cos(theta)*cos(phi1)*sin(phi2);
        dcm[0][2] = sin(phi2)*sin(theta);

        dcm[1][0] = -1.0*sin(phi2)*cos(phi1) - cos(theta)*sin(phi1)*cos(phi2);
        dcm[1][1] = -1.0*sin(phi2)*sin(phi1) + cos(theta)*cos(phi1)*cos(phi2);
        dcm[1][2] = cos(phi2)*sin(theta);

        dcm[2][0] = sin(theta)*sin(phi1);
        dcm[2][1] = -1.0*sin(theta)*cos(phi1);
        dcm[2][2] = cos(theta);
        return dcm;
      }

      template <int dim>
      UpdateTimeFlags
      LPO<dim>::need_update() const
      {
        return update_time_step;
      }

      template <int dim>
      UpdateFlags
      LPO<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      LPO<dim>::get_property_information() const
      {
        const unsigned int n_components = Tensor<2,dim>::n_independent_components;
        std::vector<std::pair<std::string,unsigned int> > property_information (1,std::make_pair("lpo water content",1));
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            property_information.push_back(std::make_pair("lpo grain " + std::to_string(grain_i) + " volume fraction olivine",1));

            for (unsigned int index = 0; index < Tensor<2,dim>::n_independent_components; index++)
              {
                property_information.push_back(std::make_pair("lpo grain " + std::to_string(grain_i) + " a_cosine_matrix olivine " + std::to_string(index),1));
              }

            property_information.push_back(std::make_pair("lpo grain " + std::to_string(grain_i) + " volume fraction enstatite",1));

            for (unsigned int index = 0; index < Tensor<2,dim>::n_independent_components; index++)
              {
                property_information.push_back(std::make_pair("lpo grain " + std::to_string(grain_i) + " a_cosine_matrix enstatite " + std::to_string(index),1));
              }
          }

        return property_information;
      }

      template <int dim>
      double
      LPO<dim>::compute_runge_kutta(std::vector<double> &volume_fractions,
                                    std::vector<Tensor<2,3> > &a_cosine_matrices,
                                    const SymmetricTensor<2,dim> &strain_rate_nondimensional,
                                    const Tensor<2,dim> &velocity_gradient_nondimensional,
                                    const DeformationType deformation_type,
                                    const std::array<double,4> &ref_resolved_shear_stress,
                                    const double strain_rate_second_invariant,
                                    const double dt) const
      {

        // RK step 1
        std::vector<double> k_volume_fractions_zero = volume_fractions;
        std::vector<Tensor<2,3> > a_cosine_matrices_zero = a_cosine_matrices;
        std::pair<std::vector<double>, std::vector<Tensor<2,3> > > derivatives;
        derivatives = this->compute_derivatives(k_volume_fractions_zero,
                                                a_cosine_matrices_zero,
                                                strain_rate_nondimensional,
                                                velocity_gradient_nondimensional,
                                                deformation_type,
                                                ref_resolved_shear_stress);

        std::vector<double> k_volume_fractions_one(n_grains);
        std::vector<Tensor<2,3> > k_a_cosine_matrices_one(n_grains);
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_one[grain_i] = strain_rate_second_invariant * dt * derivatives.first[grain_i];

            Assert(isfinite(k_volume_fractions_one[grain_i]),
                   ExcMessage("RK1: k_volume_fractions_one[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(k_volume_fractions_one[grain_i])));
            // this will be zero in the zeroth timestep (dt = 0)
            //Assert(k_volume_fractions_one[grain_i] > 0,
            //       ExcMessage("RK1: k_volume_fractions_one[ " + std::to_string(grain_i) + "] is smaller or equal to zero: "
            //                  + std::to_string(k_volume_fractions_one[grain_i])));

            k_a_cosine_matrices_one[grain_i] = strain_rate_second_invariant * dt * derivatives.second[grain_i];
          }

        // RK step 2: t+0.5*dt, wi+0.5k1
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_zero[grain_i] = volume_fractions[grain_i] + 0.5* k_volume_fractions_one[grain_i];

            Assert(isfinite(k_volume_fractions_zero[grain_i]),
                   ExcMessage("RK2.1: k_volume_fractions_zero[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(k_volume_fractions_zero[grain_i])));
            //Assert(k_volume_fractions_zero[grain_i] > 0,
            //       ExcMessage("RK2.1: k_volume_fractions_zero[ " + std::to_string(grain_i) + "] is smaller or equal to zero: "
            //                  + std::to_string(k_volume_fractions_zero[grain_i])));

            a_cosine_matrices_zero[grain_i] = a_cosine_matrices[grain_i] + 0.5 * k_a_cosine_matrices_one[grain_i];
          }

        // Todo: in the python code it has some commented out code which talks about
        // normalization of the volume fractions zero. Figure out what this is about.

        derivatives = this->compute_derivatives(k_volume_fractions_zero,
                                                a_cosine_matrices_zero,
                                                strain_rate_nondimensional,
                                                velocity_gradient_nondimensional,
                                                deformation_type,
                                                ref_resolved_shear_stress);


        std::vector<double> k_volume_fractions_two(n_grains);
        std::vector<Tensor<2,3> > k_a_cosine_matrices_two(n_grains);
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_two[grain_i] = strain_rate_second_invariant * dt * derivatives.first[grain_i];

            Assert(isfinite(k_volume_fractions_two[grain_i]),
                   ExcMessage("RK2.2: k_volume_fractions_two[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(k_volume_fractions_two[grain_i])));

            // this will be zero in the zeroth timestep (dt = 0)
            //Assert(k_volume_fractions_two[grain_i] > 0,
            //       ExcMessage("RK2.2: k_volume_fractions_two[ " + std::to_string(grain_i) + "] is smaller or equal to zero: "
            //                  + std::to_string(k_volume_fractions_two[grain_i])));

            k_a_cosine_matrices_two[grain_i] = strain_rate_second_invariant * dt * derivatives.second[grain_i];
          }

        // RK step 3: t+0.5*dt, wi+0.5k2
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_zero[grain_i] = volume_fractions[grain_i] + 0.5* k_volume_fractions_two[grain_i];
            Assert(isfinite(k_volume_fractions_zero[grain_i]),
                   ExcMessage("RK3.1: k_volume_fractions_zero[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(k_volume_fractions_zero[grain_i])));
            //Assert(k_volume_fractions_zero[grain_i] > 0,
            //       ExcMessage("RK3.1: k_volume_fractions_zero[ " + std::to_string(grain_i) + "] is smaller or equal to zero: " + std::to_string(k_volume_fractions_zero[grain_i])));
            a_cosine_matrices_zero[grain_i] = a_cosine_matrices[grain_i] + 0.5 * k_a_cosine_matrices_two[grain_i];
          }

        // Todo: in the python code it has some commented out code which talks about
        // normalization of the volume fractions zero. Figure out what this is about.
        derivatives = this->compute_derivatives(k_volume_fractions_zero,
                                                a_cosine_matrices_zero,
                                                strain_rate_nondimensional,
                                                velocity_gradient_nondimensional,
                                                deformation_type,
                                                ref_resolved_shear_stress);


        std::vector<double> k_volume_fractions_three(n_grains);
        std::vector<Tensor<2,3> > k_a_cosine_matrices_three(n_grains);
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_three[grain_i] = strain_rate_second_invariant * dt * derivatives.first[grain_i];
            Assert(isfinite(k_volume_fractions_three[grain_i]),
                   ExcMessage("RK3.2: k_volume_fractions_three[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(k_volume_fractions_three[grain_i])));

            // this will be zero in the zeroth timestep (dt = 0)
            //Assert(k_volume_fractions_three[grain_i] > 0,
            //       ExcMessage("RK3.2: k_volume_fractions_three[ " + std::to_string(grain_i) + "] is smaller or equal to zero: "
            //                  + std::to_string(k_volume_fractions_three[grain_i])));
            k_a_cosine_matrices_three[grain_i] = strain_rate_second_invariant * dt * derivatives.second[grain_i];
          }

        // RK step 4: t+0.5*dt, wi+0.5k3
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_zero[grain_i] = volume_fractions[grain_i] + 0.5* k_volume_fractions_three[grain_i];
            Assert(isfinite(k_volume_fractions_zero[grain_i]),
                   ExcMessage("RK4: k_volume_fractions_zero[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(k_volume_fractions_zero[grain_i])));
            //Assert(k_volume_fractions_zero[grain_i] > 0,
            //       ExcMessage("RK4: k_volume_fractions_zero[ " + std::to_string(grain_i) + "] is smaller or equal to zero: " + std::to_string(k_volume_fractions_zero[grain_i])));
            a_cosine_matrices_zero[grain_i] = a_cosine_matrices[grain_i] + 0.5 * k_a_cosine_matrices_three[grain_i];
          }

        // Todo: in the python code it has some commented out code which talks about
        // normalization of the volume fractions zero. Figure out what this is about.
        derivatives = this->compute_derivatives(k_volume_fractions_zero,
                                                a_cosine_matrices_zero,
                                                strain_rate_nondimensional,
                                                velocity_gradient_nondimensional,
                                                deformation_type,
                                                ref_resolved_shear_stress);


        std::vector<double> k_volume_fractions_four(n_grains);
        std::vector<Tensor<2,3> > k_a_cosine_matrices_four(n_grains);
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            k_volume_fractions_four[grain_i] = strain_rate_second_invariant * dt * derivatives.first[grain_i];

            Assert(isfinite(k_volume_fractions_four[grain_i]),
                   ExcMessage("RK5: k_volume_fractions_four[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(k_volume_fractions_four[grain_i])));

            // this will be zero in the zeroth timestep (dt = 0)
            //Assert(k_volume_fractions_four[grain_i] > 0,
            //       ExcMessage("RK5: k_volume_fractions_four[ " + std::to_string(grain_i) + "] is smaller or equal to zero: "
            //+ std::to_string(k_volume_fractions_four[grain_i])));
            k_a_cosine_matrices_four[grain_i] = strain_rate_second_invariant * dt * derivatives.second[grain_i];
          }

        // Now combine all the intermediate RK steps:
        // python: acs = acs + (kacs1 + 2.0*kacs2 + 2.0*kacs3 + kacs4)/6.0
        std::vector<double> grain_rotations(n_grains);
        double sum_volume_fractions = 0;
        for (unsigned int i = 0; i < n_grains; ++i)
          {
            Assert(!isnan(volume_fractions[i]), ExcMessage("Volume fraction of grain " + std::to_string(i) + " is not a number: " + std::to_string(volume_fractions[i]) + "."))
            Assert(isfinite(volume_fractions[i]), ExcMessage("Volume fraction of grain " + std::to_string(i) + " is not a number: " + std::to_string(volume_fractions[i]) + "."))
            volume_fractions[i] = volume_fractions[i] +
                                  (k_volume_fractions_one[i] + 2.0 * k_volume_fractions_two[i] +
                                   2.0 * k_volume_fractions_three[i] + k_volume_fractions_four[i])/6.0;


            Assert(volume_fractions[i] > 0, ExcMessage("volulme_fractions[ " + std::to_string(i) + "] is smaller or equal to zero: " + std::to_string(volume_fractions[i])));
            sum_volume_fractions += volume_fractions[i];

            Assert(!isnan(sum_volume_fractions), ExcMessage("When adding volume fraction of grain " + std::to_string(i) + ", sum_volume_fractions is not a number: " + std::to_string(sum_volume_fractions) + ", volume_fractions[i] = " + std::to_string(volume_fractions[i]) + "."))
            a_cosine_matrices[i] = a_cosine_matrices[i] +
                                   (k_a_cosine_matrices_one[i] + 2.0 * k_a_cosine_matrices_two[i] +
                                    2.0 * k_a_cosine_matrices_three[i] + k_a_cosine_matrices_four[i])/6.0;
          }

        Assert(sum_volume_fractions != 0, ExcMessage("sum_volume_fractions is zero, which should never happen."));


        return sum_volume_fractions;
      }

      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3> > >
      LPO<dim>::compute_derivatives(const std::vector<double> &volume_fractions,
                                    const std::vector<Tensor<2,3> > &a_cosine_matrices,
                                    const SymmetricTensor<2,dim> &strain_rate_nondimensional,
                                    const Tensor<2,dim> &velocity_gradient_tensor_nondimensional,
                                    const DeformationType deformation_type,
                                    const std::array<double,4> &ref_resolved_shear_stress) const
      {
        // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
        // adding an extra dimension which is zero.
        // Todo: for now we just add a thrid row and column
        // and make them zero. We have to check whether that is correct.
        SymmetricTensor<2,3> strain_rate_nondim_3d;
        strain_rate_nondim_3d[0][0] = strain_rate_nondimensional[0][0];
        strain_rate_nondim_3d[0][1] = strain_rate_nondimensional[0][1];
        //sym: strain_rate_nondim_3d[0][0] = strain_rate_nondimensional[1][0];
        strain_rate_nondim_3d[1][1] = strain_rate_nondimensional[1][1];

        if (dim == 3)
          {
            strain_rate_nondim_3d[0][2] = strain_rate_nondimensional[0][2];
            strain_rate_nondim_3d[1][2] = strain_rate_nondimensional[1][2];
            //sym: strain_rate_nondim_3d[0][0] = strain_rate_nondimensional[2][0];
            //sym: strain_rate_nondim_3d[0][1] = strain_rate_nondimensional[2][1];
            strain_rate_nondim_3d[2][2] = strain_rate_nondimensional[2][2];
          }
        Tensor<2,3> velocity_gradient_tensor_nondim_3d;
        velocity_gradient_tensor_nondim_3d[0][0] = velocity_gradient_tensor_nondimensional[0][0];
        velocity_gradient_tensor_nondim_3d[0][1] = velocity_gradient_tensor_nondimensional[0][1];
        velocity_gradient_tensor_nondim_3d[1][0] = velocity_gradient_tensor_nondimensional[1][0];
        velocity_gradient_tensor_nondim_3d[1][1] = velocity_gradient_tensor_nondimensional[1][1];
        if (dim == 3)
          {
            velocity_gradient_tensor_nondim_3d[0][2] = velocity_gradient_tensor_nondimensional[0][2];
            velocity_gradient_tensor_nondim_3d[1][2] = velocity_gradient_tensor_nondimensional[1][2];
            velocity_gradient_tensor_nondim_3d[2][0] = velocity_gradient_tensor_nondimensional[2][0];
            velocity_gradient_tensor_nondim_3d[2][1] = velocity_gradient_tensor_nondimensional[2][1];
            velocity_gradient_tensor_nondim_3d[2][2] = velocity_gradient_tensor_nondimensional[2][2];
          }

        // create output variables
        std::vector<double> deriv_volume_fractions(n_grains);
        std::vector<Tensor<2,3> > deriv_a_cosine_matrices(n_grains);

        // create shorcuts
        const std::array<double, 4> &tau = ref_resolved_shear_stress;

        double Xm = deformation_type == DeformationType::enstatite
                    ?
                    x_olivine
                    :
                    1 - x_olivine;


        std::vector<double> strain_energy(n_grains);
        double mean_strain_energy = 0;

        // loop over grains
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Compute the Schmidt tensor for this grain (nu), s is the slip system.
            // We first compute beta_s,nu (equation 5, Kaminski & Ribe, 2001)
            // Then we use the beta to calculate the Schmidt tensor G_{ij} (Eq. 5, Kaminski & Ribe, 2001)
            Tensor<2,3> G;
            Tensor<1,3> w;
            Tensor<1,4> beta({1.0, 1.0, 1.0, 1.0});

            // these are variables we only need for olivine, but we need them for both
            // within this if bock and the next ones
            // todo: initialize to dealii uninitialized value
            unsigned int index_max_q = 0;
            unsigned int index_intermediate_q = 0;
            unsigned int index_min_q = 0;
            unsigned int index_inactive_q = 0;

            // compute G and beta for enstatite or olivine.
            if (deformation_type == DeformationType::enstatite)
              {
                beta = 1.0;
                for (unsigned int i = 0; i < 3; i++)
                  {
                    for (unsigned int j = 0; j < 3; j++)
                      {
                        G[i][j] = G[i][j] + 2.0 * a_cosine_matrices[grain_i][2][i] * a_cosine_matrices[grain_i][0][j];
                      }
                  }
              }
            else
              {
                const unsigned int n_slip = 4;

                Tensor<1,4> bigI;
                // this should be equal to a_cosine_matrices[grain_i]*a_cosine_matrices[grain_i]?
                // todo: check and maybe replace?
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    for (unsigned int j = 0; j < 3; ++j)
                      {
                        //strain_rate_nondimensional is in 2d not big enough
                        bigI[0] = bigI[0] + strain_rate_nondim_3d[i][j] * a_cosine_matrices[grain_i][0][i] * a_cosine_matrices[grain_i][1][j];

                        bigI[1] = bigI[1] + strain_rate_nondim_3d[i][j] * a_cosine_matrices[grain_i][0][i] * a_cosine_matrices[grain_i][2][j];

                        bigI[2] = bigI[2] + strain_rate_nondim_3d[i][j] * a_cosine_matrices[grain_i][2][i] * a_cosine_matrices[grain_i][1][j];

                        bigI[3] = bigI[3] + strain_rate_nondim_3d[i][j] * a_cosine_matrices[grain_i][2][i] * a_cosine_matrices[grain_i][0][j];
                      }
                  }

                // compute the element wise absolute value of the element wise
                // division of BigI by tau (tau = ref_resolved_shear_stress).
                std::vector<double> q_abs(4);
                for (unsigned int i = 0; i < 4; i++)
                  {
                    q_abs[i] = std::abs(bigI[i] / tau[i]);
                  }

                // here we find the indices starting at the largest value and ending at the smallest value
                // and assign them to special variables. Because all the variables are absolute values,
                // we can set them to a negative value to ignore them. This should be faster then deleting
                // the element, which would require allocation. (not tested)
                index_max_q = std::distance(q_abs.begin(),max_element(q_abs.begin(), q_abs.end()));

                q_abs[index_max_q] = -1;

                index_intermediate_q = std::distance(q_abs.begin(),max_element(q_abs.begin(), q_abs.end()));

                q_abs[index_intermediate_q] = -1;

                index_min_q = std::distance(q_abs.begin(),max_element(q_abs.begin(), q_abs.end()));

                q_abs[index_min_q] = -1;

                index_inactive_q = std::distance(q_abs.begin(),max_element(q_abs.begin(), q_abs.end()));

                // todo: explain
                double ratio = tau[index_max_q]/bigI[index_max_q];

                double q_intermediate = ratio * (bigI[index_intermediate_q]/tau[index_intermediate_q]);

                double q_min = ratio * (bigI[index_min_q]/tau[index_min_q]);

                // todo: explain
                beta[index_max_q] = 1.0; // max q_abs, weak system (most deformation) "s=1"
                beta[index_intermediate_q] = q_intermediate * std::pow(std::abs(q_intermediate), stress_exponent-1);
                beta[index_min_q] = q_min * std::pow(std::abs(q_min), stress_exponent-1);
                beta[index_inactive_q] = 0.0;


                // todo: explain
                for (unsigned int i = 0; i < 3; i++)
                  {
                    for (unsigned int j = 0; j < 3; j++)
                      {
                        G[i][j] = 2.0 * (beta[0] * a_cosine_matrices[grain_i][0][i] * a_cosine_matrices[grain_i][1][j]
                                         + beta[1] * a_cosine_matrices[grain_i][0][i] * a_cosine_matrices[grain_i][2][j]
                                         + beta[2] * a_cosine_matrices[grain_i][2][i] * a_cosine_matrices[grain_i][1][j]
                                         + beta[3] * a_cosine_matrices[grain_i][2][i] * a_cosine_matrices[grain_i][0][j]);
                      }
                  }
              }


            // Now calculate the analytic solution to the deformation minimization problem
            // compute gamma (equation 7, Kaminiski & Ribe, 2001)
            // todo: expand
            double top = 0;
            double bottom = 0;
            for (unsigned int i = 0; i < 3; ++i)
              {
                // Following the Drex code, which differs from EPSL paper,
                // which says gamma_nu depends on i+1: actually uses i+2
                unsigned int ip2 = i + 2;
                if (ip2 > 2)
                  ip2 = ip2-3;

                top = top - (velocity_gradient_tensor_nondim_3d[i][ip2]-velocity_gradient_tensor_nondim_3d[ip2][i])*(G[i][ip2]-G[ip2][i]);
                bottom = bottom - (G[i][ip2]-G[ip2][i])*(G[i][ip2]-G[ip2][i]);

                for (unsigned int j = 0; j < 3; ++j)
                  {
                    top = top + 2.0 * G[i][j]*velocity_gradient_tensor_nondim_3d[i][j];
                    bottom = bottom + 2.0* G[i][j] * G[i][j];
                  }
              }
            double gamma = top/bottom;

            // compute w (equation 8, Kaminiski & Ribe, 2001)
            // todo: explain what w is
            // todo: there was a loop around this in the phyton code, discuss/check
            w[0] = 0.5*(velocity_gradient_tensor_nondim_3d[2][1]-velocity_gradient_tensor_nondim_3d[1][2]) - 0.5*(G[2][1]-G[1][2])*gamma;
            w[1] = 0.5*(velocity_gradient_tensor_nondim_3d[0][2]-velocity_gradient_tensor_nondim_3d[2][0]) - 0.5*(G[0][2]-G[2][0])*gamma;
            w[2] = 0.5*(velocity_gradient_tensor_nondim_3d[1][0]-velocity_gradient_tensor_nondim_3d[0][1]) - 0.5*(G[1][0]-G[0][1])*gamma;

            // Compute strain energy for this grain (abrivated Estr)
            // For olivine: DREX only sums over 1-3. But Thissen's matlab code corrected
            // this and writes each term using the indices created when calculating bigI.
            // Note tau = RRSS = (tau_m^s/tau_o), this why we get tau^(p-n)
            if (deformation_type == DeformationType::enstatite)
              {
                // todo: check beta is alright
                const double rhos = std::pow(tau[3],(exponent_p-stress_exponent)) *
                                    std::pow(std::abs(gamma*beta[0]),exponent_p/stress_exponent);
                strain_energy[grain_i] = rhos * std::exp(-nucliation_efficientcy*rhos*rhos);
              }
            else
              {
                const double rhos1 = std::pow(tau[index_max_q],exponent_p-stress_exponent) *
                                     std::pow(std::abs(gamma*beta[index_max_q]),exponent_p/stress_exponent);

                const double rhos2 = std::pow(tau[index_intermediate_q],exponent_p-stress_exponent) *
                                     std::pow(std::abs(gamma*beta[index_intermediate_q]),exponent_p/stress_exponent);

                const double rhos3 = std::pow(tau[index_min_q],exponent_p-stress_exponent) *
                                     std::pow(std::abs(gamma*beta[index_min_q]),exponent_p/stress_exponent);

                const double rhos4 = std::pow(tau[index_inactive_q],exponent_p-stress_exponent) *
                                     std::pow(std::abs(gamma*beta[index_inactive_q]),exponent_p/stress_exponent);

                strain_energy[grain_i] = (rhos1 * exp(-nucliation_efficientcy * rhos1 * rhos1)
                                          + rhos2 * exp(-nucliation_efficientcy * rhos2 * rhos2)
                                          + rhos3 * exp(-nucliation_efficientcy * rhos3 * rhos3)
                                          + rhos4 * exp(-nucliation_efficientcy * rhos4 * rhos4));

                Assert(isfinite(strain_energy[grain_i]), ExcMessage("strain_energy[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(strain_energy[grain_i])
                                                                    + ", rhos1 = " + std::to_string(rhos1) + ", rhos2 = " + std::to_string(rhos2) + ", rhos3 = " + std::to_string(rhos3) + ", rhos4= " + std::to_string(rhos4) + ", nucliation_efficientcy = " + std::to_string(nucliation_efficientcy) + "."));
              }

            // compute the derivative of the cosine matrix a: \frac{\partial a_{ij}}{\partial t}
            // (Eq. 9, Kaminski & Ribe 2001)
            deriv_a_cosine_matrices[grain_i] = 0;
            if (volume_fractions[grain_i] > threshold_GBS/n_grains)
              {
                for (unsigned int i = 0; i < 3; ++i)
                  for (unsigned int j = 0; j < 3; ++j)
                    for (unsigned int k = 0; k < 3; ++k)
                      for (unsigned int l = 0; l < 3; ++l)
                        {
                          deriv_a_cosine_matrices[grain_i][i][j] = deriv_a_cosine_matrices[grain_i][i][j] + permutation_operator_3d[j][k][l] * a_cosine_matrices[grain_i][i][l]*w[k];
                        }


                mean_strain_energy += volume_fractions[grain_i] * strain_energy[grain_i];

                Assert(isfinite(mean_strain_energy), ExcMessage("mean_strain_energy when adding grain " + std::to_string(grain_i) + " is not finite: " + std::to_string(mean_strain_energy)
                                                                + ", volume_fractions[grain_i] = " + std::to_string(volume_fractions[grain_i]) + ", strain_energy[grain_i] = " + std::to_string(strain_energy[grain_i]) + "."));
              }
            else
              {
                strain_energy[grain_i] = 0;
              }

          }

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            deriv_volume_fractions[grain_i] = Xm * mobility * volume_fractions[grain_i] * (mean_strain_energy - strain_energy[grain_i]);

            Assert(isfinite(deriv_volume_fractions[grain_i]), ExcMessage("deriv_volume_fractions[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(deriv_volume_fractions[grain_i])));
          }

        return std::pair<std::vector<double>, std::vector<Tensor<2,3> > >(deriv_volume_fractions, deriv_a_cosine_matrices);
      }

      template<int dim>
      unsigned int
      LPO<dim>::get_number_of_grains()
      {
        return n_grains;
      }


      template <int dim>
      void
      LPO<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LPO");
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
      LPO<dim>::parse_parameters (ParameterHandler &prm)
      {

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("LPO");
            {

              random_number_seed = prm.get_integer ("Random number seed"); // 2
              n_grains = prm.get_integer("Number of grains per praticle"); //10000;
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(LPO,
                                        "lpo",
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

