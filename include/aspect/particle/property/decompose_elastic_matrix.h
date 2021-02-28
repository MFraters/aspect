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

#ifndef _aspect_particle_property_decompose_elastic_matrix_h
#define _aspect_particle_property_decompose_elastic_matrix_h

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
      class DecomposeElasticMatrix : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * constructor
           */
          DecomposeElasticMatrix();

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
           * Return a specific permutation based on an index
           */
          static
          std::array<unsigned short int, 3>
          indexed_even_permutation(const unsigned short int index);


          /**
           * Computes the voigt stiffness tensor from the elastic tensor
           */
          static
          SymmetricTensor<2,3>
          compute_voigt_stiffness_tensor(const SymmetricTensor<2,6> &elastic_tensor);

          /**
           * Computes the dilatation stiffness tensor from the elastic tensor
           */
          static
          SymmetricTensor<2,3>
          compute_dilatation_stiffness_tensor(const SymmetricTensor<2,6> &elastic_tensor);

          /**
           * computes the bulk and shear moduli from the voigt and dilatation stiffness tensors
           */
          static
          std::pair<double,double>
          compute_bulk_and_shear_moduli(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                        const SymmetricTensor<2,3> &voigt_stiffness_tensor);

          static
          Tensor<1,9>
          compute_isotropic_approximation(const double bulk_modulus,
                                          const double shear_modulus);

          /**
           * todo
           */
          static
          Tensor<2,3> compute_unpermutated_SCC(const SymmetricTensor<2,3> &dilatation_stiffness_tensor,
                                               const SymmetricTensor<2,3> &voigt_stiffness_tensor);

          /**
           * todo
           *
           * depricated, use compute_elastic_tensor_SCC_decompositions instead.
           */
          std::pair<SymmetricTensor<2,6>,Tensor<2,3>> compute_minimum_hexagonal_projection(const Tensor<2,3> &unpermutated_SCC,
                                                   const SymmetricTensor<2,6> &elastic_tensor,
                                                   const double elastic_vector_norm) const;

          /**
           * todo
           */
          static
          std::array<std::array<double,3>,7>
          compute_elastic_tensor_SCC_decompositions(
            const Tensor<2,3> &unpermutated_SCC,
            const SymmetricTensor<2,6> &elastic_matrix);


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
           * project elastc vector onto hexagonal symmetry as explained in the appendix A4 of
           * Browaeys and Chevrot (2004), GJI (doi: 10.1111/j.1365-246X.2004.024115.x).
           * depricated, needs update and added for all projections or be removed.
           */
          Tensor<1,9> project_onto_hexagonal_symmetry(const Tensor<1,21> &elastic_vector) const;


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


          static SymmetricTensor<2,21> projection_matrix_tric_to_mono;
          static SymmetricTensor<2,9> projection_matrix_mono_to_ortho;
          static SymmetricTensor<2,9> projection_matrix_ortho_to_tetra;
          static SymmetricTensor<2,9> projection_matrix_tetra_to_hexa;
          static SymmetricTensor<2,9> projection_matrix_hexa_to_iso;

        private:
          unsigned int lpo_data_position;
          unsigned int lpo_elastic_tensor_data_position;

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
           * efficientcy of nucleation parameter.
           * lamda_m in equation 8 of Kamisnki et al. (2004, Geophys. J. Int)
           */
          double nucleation_efficientcy;

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
          static
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
