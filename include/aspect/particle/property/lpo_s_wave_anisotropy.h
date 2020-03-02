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

#ifndef _aspect_particle_property_lpo_s_wave_anisotropy_h
#define _aspect_particle_property_lpo_s_wave_anisotropy_h

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

      /**
       * Todo: write what this plugin does.
       *
       * A class that integrates the finite strain that a particle has
       * experienced.
       * The implementation of this property is equivalent to the implementation
       * for compositional fields that is described in the cookbook
       * finite_strain <code>cookbooks/finite_strain/finite_strain.cc</code>.
       *
       * The layout of the data vector per partcle is the following (note that for this plugin the following dim's are always 3):
       * 1 averaged a axis of olivine -> 3 (dim) doubles, starts at:
       *                                   data_position + 1,
       * 2 averaged b axis of olivine -> 3 (dim) doubles, starts at:
       *                                   data_position + 4
       * 3 averaged c axis of olivine -> 3 (dim) doubles, starts at:
       *                                   data_position + 7
       * 4 averaged a axis of enstatite -> 3 (dim) doubles, starts at:
       *                                    data_position + 10
       * 5 averaged b axis of enstatite -> 3 (dim) doubles, starts at:
       *                                    data_position + 13
       * 6 averaged c axis of enstatite -> 3 (dim) doubles, starts at:
       *                                    data_position + 16
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class LpoSWaveAnisotropy : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * constructor
           */
          LpoSWaveAnisotropy();

          /**
           * Initialization function. This function is called once at the
           * beginning of the program after parse_parameters is run.
           */
          virtual
          void
          initialize ();

          /**
           * Initialization function. This function is called once directly
           * after the initialize function when the property_information is
           * set.
           */
          virtual
          void
          post_initialize ();

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

          /**
           * todo
           */
          std::array<std::array<double,3>,3> compute_s_wave_anisotropy(std::vector<Tensor<2,3> > matrices) const;

          /**
           * todo
           */
          std::vector<Tensor<2,3> >
          random_draw_volume_weighting(std::vector<double> fv,
                                       std::vector<Tensor<2,3>> matrices,
                                       unsigned int n_output_grains) const;


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
          unsigned int lpo_data_position;

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

