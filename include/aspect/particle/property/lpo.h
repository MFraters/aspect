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

#ifndef _aspect_particle_property_lpo_h
#define _aspect_particle_property_lpo_h

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>
#include <array>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <boost/random.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      enum class DeformationType
      {
        A_type, B_type, C_type, D_type, E_type, enstatite
      };
      /**
       * Todo: write what this plugin does.
       *
       * The layout of the data vector per perticle is the following (note that for this plugin the following dim's are always 3):
       * 1. water content -> 1 double, always at location data_position -> i = 0;
       * 2. N grains times:
       *    2.1. volume fraction olivine -> 1 double, at location:
       *                                    data_position + i_grain * 20 + 1, or
       *                                    data_position + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + 1
       *    2.2. a_cosine_matrix olivine -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
       *                                    data_position + i_grain * 20 + 2, or
       *                                    data_position + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + 2
       *    2.3. volume fraction enstatite -> 1 double, at location:
       *                                      data_position + i_grain * 20 + 11, or
       *                                      data_position + i_grain * (Tensor<2,3>::n_independent_components + 2)  + 11
       *    2.4. a_cosine_matrix enstatite -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
       *                                      data_position + i_grain * 20 + 12, or
       *                                      data_position + i_grain * (Tensor<2,3>::n_independent_components + 2) + 12
       * We store it this way because this is also the order in which it is read, so this should
       * theoretically minimize chache misses. Note: It has not been tested wheter this is faster then storing it in another way.
       *
       * An other note is that we store exactly the same amount of olivine and enstatite grain, although
       * the volume may not be the same. This has to do with that we need a minimum amount of grains
       * per tracer to perform reliable statistics on it. This miminum is the same for both olivine and
       * enstatite.
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class LPO : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * constructor
           */
          LPO();
          /**
           * Initialization function. This function is called once at the
           * beginning of the program after parse_parameters is run.
           */
          virtual
          void
          initialize ();

          /**
           * Initialization function. This function is called once at the
           * creation of every particle for every property to initialize its
           * value.
           *
           * @param [in] position The current particle position.
           * @param [in,out] particle_properties The properties of the particle
           * that is initialized within the call of this function. The purpose
           * of this function should be to extend this vector by a number of
           * properties.
           */
          virtual
          void
          initialize_one_particle_property (const Point<dim> &position,
                                            std::vector<double> &particle_properties) const;

          /**
           * Update function. This function is called every time an update is
           * request by need_update() for every particle for every property.
           *
           * @param [in] data_position An unsigned integer that denotes which
           * component of the particle property vector is associated with the
           * current property. For properties that own several components it
           * denotes the first component of this property, all other components
           * fill consecutive entries in the @p particle_properties vector.
           *
           * @param [in] position The current particle position.
           *
           * @param [in] solution The values of the solution variables at the
           * current particle position.
           *
           * @param [in] gradients The gradients of the solution variables at
           * the current particle position.
           *
           * @param [in,out] particle_properties The properties of the particle
           * that is updated within the call of this function.
           */
          virtual
          void
          update_one_particle_property (const unsigned int data_position,
                                        const Point<dim> &position,
                                        const Vector<double> &solution,
                                        const std::vector<Tensor<1,dim> > &gradients,
                                        const ArrayView<double> &particle_properties) const;

          /**
           * This implementation tells the particle manager that
           * we need to update particle properties every time step.
           */
          UpdateTimeFlags
          need_update () const;

          /**
           * Return which data has to be provided to update the property.
           * The integrated strains needs the gradients of the velocity.
           */
          virtual
          UpdateFlags
          get_needed_update_flags () const;

          /**
           * Set up the information about the names and number of components
           * this property requires.
           *
           * @return A vector that contains pairs of the property names and the
           * number of components this property plugin defines.
           */
          virtual
          std::vector<std::pair<std::string, unsigned int> >
          get_property_information() const;

          static
          void
          load_lpo_particle_data(unsigned int lpo_index,
                                 const ArrayView<double> &data,
                                 unsigned int n_grains,
                                 std::vector<double> &volume_fractions_olivine,
                                 std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                 std::vector<double> &volume_fractions_enstatite,
                                 std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite);

          static
          void
          store_lpo_particle_data(unsigned int lpo_data_position,
                                  const ArrayView<double> &data,
                                  unsigned int n_grains,
                                  std::vector<double> &volume_fractions_olivine,
                                  std::vector<Tensor<2,3> > &a_cosine_matrices_olivine,
                                  std::vector<double> &volume_fractions_enstatite,
                                  std::vector<Tensor<2,3> > &a_cosine_matrices_enstatite);

          /**
           * Find nearest orthogonal matrix using a SVD if the
           * deteriminant is more than a tolerance away from one.
           */
          void
          orthogonalize_matrix(dealii::Tensor<2, 3> &tensor,
                               double tolerance) const;

          /**
           * Return resolved shear stress based on deformation type.
           * @param max_value is set by default to 1e60 instead of infinity or
           * std::numeric_limits<double>::max() because it needs to be able to
           * be squared without becomming infinite. The default is based on
           * Fortran D-Rex uses 1e60
           */
          std::array<double,4>
          reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type,
                                                                double max_value = 1e60) const;

          std::vector<Tensor<2,3> >
          random_draw_volume_weighting(std::vector<double> fv,
                                       std::vector<Tensor<2,3>> matrices,
                                       unsigned int n_output_grains) const;

          /**
           * derivatives: Todo
           */
          double
          compute_runge_kutta(std::vector<double> &volume_fractions,
                              std::vector<Tensor<2,3> > &a_cosine_matrices,
                              const SymmetricTensor<2,dim> &strain_rate,
                              const Tensor<2,dim> &velocity_gradient_tensor,
                              const DeformationType deformation_type,
                              const std::array<double,4> &ref_resolved_shear_stress,
                              const double strain_rate_second_invariant,
                              const double dt) const;

          /**
           * derivatives: Todo
           */
          std::pair<std::vector<double>, std::vector<Tensor<2,3> > >
          compute_derivatives(const std::vector<double> &volume_fractions,
                              const std::vector<Tensor<2,3> > &a_cosine_matrices,
                              const SymmetricTensor<2,dim> &strain_rate_nondimensional,
                              const Tensor<2,dim> &velocity_gradient_tensor_nondimensional,
                              const DeformationType deformation_type,
                              const std::array<double,4> &ref_resolved_shear_stress) const;

          std::vector<std::vector<double> >
          volume_weighting(std::vector<double> fv, std::vector<std::vector<double> > angles) const;

          double
          wrap_angle(const double angle) const;


          std::vector<double>
          extract_euler_angles_from_dcm(const Tensor<2,3> &rotation_matrix) const;

          Tensor<2,3>
          dir_cos_matrix2(double phi1, double theta, double phi2) const;

          /**
           * Return the number of grains per particle
           */
          static
          unsigned int
          get_number_of_grains();

          /**
           * Todo, rewrite.
           * Declare the parameters this class takes through input files.
           * Derived classes should overload this function if they actually do
           * take parameters; this class declares a fall-back function that
           * does nothing, so that property classes that do not take any
           * parameters do not have to do anything at all.
           *
           * This function is static (and needs to be static in derived
           * classes) so that it can be called without creating actual objects
           * (because declaring parameters happens before we read the input
           * file and thus at a time when we don't even know yet which
           * property objects we need).
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Todo: rewrite.
           * Read the parameters this class declares from the parameter file.
           * The default implementation in this class does nothing, so that
           * derived classes that do not need any parameters do not need to
           * implement it.
           */
          virtual
          void
          parse_parameters (ParameterHandler &prm);

        private:

          double rad_to_degree = 180.0/M_PI;
          double degree_to_rad = M_PI/180.0;
          /**
           * Todo: rewrite
           * Random number generator. For reproducibility of tests it is
           * initialized in the constructor with a constant.
           */
          mutable boost::lagged_fibonacci44497            random_number_generator;
          //boost::variate_generator<boost::lagged_fibonacci44497&, boost::random::uniform_real_distribution<double> > get_random_number;
          unsigned int random_number_seed;

          static
          unsigned int n_grains;

          // when doing the random draw volume weighting, this sets how many samples are taken.
          unsigned int n_samples;

          double x_olivine;

          double stress_exponent;

          /**
           * efficientcy of nucliation parameter.
           * lamda_m in equation 8 of Kamisnki et al. (2004, Geophys. J. Int)
           */
          double nucliation_efficientcy;

          /**
           * An exponent described in equation 10 of Kaminsty and Ribe (2001, EPSL)
           */
          double exponent_p;

          /**
           * todo
           */
          double threshold_GBS;

          /**
           * todo
           */
          Tensor<3,3> permutation_operator_3d;

          /**
           * grain boundery mobility
           */
          double mobility;



      };
    }
  }
}

#endif

