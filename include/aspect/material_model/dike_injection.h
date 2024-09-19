/*
  Copyright (C) 2024 by the authors of the ASPECT code.
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
#ifndef _aspect_material_model_dike_injection_h
#define _aspect_material_model_dike_injection_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/signaling_nan.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <aspect/particle/integrator/interface.h>

#include <aspect/solution_evaluator.h>

#include <tuple>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

class MPIChain{
    // Uses a chained MPI message (T) to coordinate serial execution of code (the content of the message is irrelevant).
    private:
        int message_out; // The messages aren't really used here
        int message_in;
        int size;
        int rank;

    public:
        void next(){
            // Send message to next core (if there is one)
            if(rank + 1 < size) {
            // MPI_Send - Performs a standard-mode blocking send.
            MPI_Send(& message_out, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }

        void wait(int & msg_count) {
            // Waits for message to arrive. Message is well-formed if msg_count = 1
            MPI_Status status;

            // MPI_Probe - Blocking test for a message.
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, & status);
            // MPI_Get_count - Gets the number of top level elements.
            MPI_Get_count(& status, MPI_INT, & msg_count);

            if(msg_count == 1) {
                // MPI_Recv - Performs a standard-mode blocking receive.
                MPI_Recv(& message_in, msg_count, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, & status);
            }
        }

        MPIChain(int message_init, int c_rank, int c_size): message_out(message_init), size(c_size), rank(c_rank) {}

        int get_rank() const { return rank;}
        int get_size() const { return size;}
};
    /**
     * This dike injection function defines material injection throug a narrow
     * dike by prescribing a dilation term applied to the mass equation, a
     * deviatoric strain rate correction term in the momentum equation, and
     * a injection-related heating term in the temperature equation.
     * Since the direction of the dike opening is assumed to be the same as
     * the direction of plate spreading, there should be no volumetric
     * deformation in the dike, i.e., dike injection has no effect on the
     * deviatroic strain rate.
     *
     * @ingroup MaterialModels
     */

    template <int dim>
    class DikeInjection : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Initialize the model at the beginning of the run.
         */
        void initialize() override;

        /**
         * Update the base model and dilation function at the beginning
         * of each timestep.
         */
        void update() override;

        /**
         * Function to compute the material properties in @p out given
         * the inputs in @p in.
         */
        void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                      MaterialModel::MaterialModelOutputs<dim> &out) const override;
        /**
         * Declare the parameters through input files.
         */
        static void
        declare_parameters (ParameterHandler &prm);

        /**
         * Parse parameters through the input file
         */
        void
        parse_parameters (ParameterHandler &prm) override;

        /**
         * Indicate whether material is compressible only based on
         * the base model.
         */
        bool is_compressible () const override;

        /**
         * Method to calculate reference viscosity. Not used anymore.
         */
        // double reference_viscosity () const override;

        void
        create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const override;

      private:
        std::vector<std::vector<Point<dim>>> dike_locations;
        std::unique_ptr<Particles::ParticleHandler<dim>> particle_handler;
        std::unique_ptr<aspect::Particle::Integrator::Interface<dim>> particle_integrator;
        //bool particle_lost;
        // stores the 0: the index, 1 wether they are active (0), lost(1) or lost and processed (2) and 2: the location if lost.
        std::vector<std::tuple<unsigned int,unsigned int,Point<dim>>> particle_statuses;
        //unsigned int particle_lost_index;
        //Point<dim> particle_lost_location;

        void set_particle_lost(const typename Particles::ParticleIterator<dim> &particle,
                               const typename Triangulation<dim>::active_cell_iterator &cell);

        /**
         * @brief computes the velocity field based on the povided solution vector. By default it computes the stress largesest eigenvector
         *
         * @param cells
         * @param positions
         * @param reference_positions
         * @param input_solution
         * @return std::vector<Tensor<1,dim>>
         */
        std::vector<Tensor<1,dim>> compute_velocity_field(std::vector<typename DoFHandler<dim>::active_cell_iterator> &cells,
                                                           std::vector<Point<dim>> &positions,
                                                           std::vector<Point<dim>> &reference_positions,
                                                           std::vector<unsigned int> &particle_map,
                                                           const LinearAlgebra::BlockVector &input_solution);
        /**
         * Parsed function that specifies the region and amount of
         * material that is injected into the model.
         */
        Functions::ParsedFunction<dim> injection_function;

        /**
         * Amount of new injected material from the dike
         */
        double dike_material_injection_fraction;

        double dike_visosity_multiply_factor;

        double min_dike_viscosity;

        double max_dike_viscosity;

        unsigned dikes_created;

        double dike_width;

        double dike_dilation_velocity;

        /**
         * Temperature at the bottom of the generated dike.
         * It usually equals to the temperature at the
         * intersection of the brittle-ductile transition
         * zone and the dike.
         */
        double T_bottom_dike;

        /**
         * Whether using the random dike generation or not.
         */
        bool enable_random_dike_generation;

        /**
         * X_coordinate of the center of the dike generation zone.
         */
        double x_center_dike_generation_zone;

        /**
         * Width of the center of the dike generation zone.
         */
        double width_dike_generation_zone;

        /**
         * x_coordinates of the randomly generated dike
         */
        double x_dike_left_boundary;
        double x_dike_right_boundary;

        /**
         * Reference top depth of the randomly generated dike.
         */
        double ref_top_depth_random_dike;

        /**
         * full range of dike depth change.
         */
        double range_depth_change_random_dike;

        /**
         * randomly genetrated dike top depth.
         */
        double top_depth_random_dike;

        /**
         * Width of the randomly generated dike.
         */
        //double width_random_dike;

        /**
         * Seed for the random number generator
         */
        double seed;

        /**
         * The total refinment levels in the model, which equals to
         * the sum of global refinement levels and adpative refinement
         * levels. This is used for calcuting the dike location.
         */
        double total_refinement_levels;

        /**
         * Pointer to the material model used as the base model.
         */
        std::shared_ptr<MaterialModel::Interface<dim>> base_model;

        /**
         * We cache the evaluators that are necessary to evaluate
         * compositions. By caching the evaluator, we can avoid
         * recreating them every time we need it.
         */
        mutable std::vector<std::unique_ptr<FEPointEvaluation<1, dim>>> composition_evaluators;


        /**
         * Parameters for anhydrous melting of peridotite after Katz, 2003
         */

        double melt_fraction_threshold;

        mutable std::mt19937 random_number_generator;

        // for the solidus temperature
        double A1;   // °C
        double A2; // °C/Pa
        double A3; // °C/(Pa^2)

        // for the lherzolite liquidus temperature
        double B1;   // °C
        double B2;   // °C/Pa
        double B3; // °C/(Pa^2)

        // for the liquidus temperature
        double C1;   // °C
        double C2;  // °C/Pa
        double C3; // °C/(Pa^2)

        // for the reaction coefficient of pyroxene
        double r1;     // cpx/melt
        double r2;     // cpx/melt/GPa
        double M_cpx;  // mass fraction of pyroxenite

        // melt fraction exponent
        double beta;

        /**
         * Parameters for melting of pyroxenite after Sobolev et al., 2011
         */

        // for the melting temperature
        double D1;    // °C
        double D2;  // °C/Pa
        double D3; // °C/(Pa^2)

        // for the melt-fraction dependence of productivity
        double E1;
        double E2;
    };
  }
}

#endif
