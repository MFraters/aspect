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

#include <algorithm>
#include <vector>
#include <random>

#include "aspect/material_model/dike_injection.h"
#include <aspect/geometry_model/box.h>
#include <aspect/mesh_deformation/free_surface.h>
#include <aspect/utilities.h>
#include <aspect/parameters.h>
#include <aspect/solution_evaluator.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/particle/integrator/rk_4.h>

#include <aspect/simulator.h>


#include <deal.II/grid/grid_tools.h>
namespace aspect
{

// Global variables (to be set by parameters)
  unsigned int clear_composition_field_index;
  namespace MaterialModel
  {
    template <int dim>
    void clear_compositional_field (const SimulatorAccess<dim> &simulator_access)
    {
     //std::cout << "clear_compositional_field signal" << std::endl;
      simulator_access.get_pcout() << "Signal clear_compositional_field triggered, clearing field " << clear_composition_field_index << "!" << std::endl;
      const typename Simulator<dim>::AdvectionField adv_field (Simulator<dim>::AdvectionField::composition(clear_composition_field_index));
      //std::cout << "before: " << const_cast<LinearAlgebra::BlockVector &>(simulator_access.get_solution()).block(adv_field.block_index(simulator_access.introspection()))[0] << std::endl;
      const_cast<LinearAlgebra::BlockVector &>(simulator_access.get_solution()).block(adv_field.block_index(simulator_access.introspection())) = 0;
      const_cast<LinearAlgebra::BlockVector &>(simulator_access.get_old_solution()).block(adv_field.block_index(simulator_access.introspection())) = 0;
      const_cast<LinearAlgebra::BlockVector &>(simulator_access.get_current_linearization_point()).block(adv_field.block_index(simulator_access.introspection())) = 0;
      //std::cout << "after: " << const_cast<LinearAlgebra::BlockVector &>(simulator_access.get_solution()).block(adv_field.block_index(simulator_access.introspection()))[0] << std::endl;
    }
    template <int dim>
    void
    DikeInjection<dim>::initialize()
    {
      base_model->initialize();
      this->get_signals().start_timestep.connect(&clear_compositional_field<dim>);
    }

    template <int dim>
    void
    DikeInjection<dim>::set_particle_lost(const typename Particles::ParticleIterator<dim> &particle, const typename Triangulation<dim>::active_cell_iterator &/*cell*/)
    {
     //std::cout << "lost particle here!!!!: " << particle->get_location() << std::endl;
      //particle_lost = true;
      //particle_lost_location = particle->get_location();
      //particle_lost_index = particle->index();
      bool found = false;
      for (auto &particle_status : particle_statuses)
        {
         //std::cout << "particle_status = " << std::get<0>(particle_status) << ";" << std::get<1>(particle_status) << "; " << (std::get<2>(particle_status))[0] << ":" << (std::get<2>(particle_status))[1]
                    //<< ", particle = " << particle->get_id() << "; " << particle->get_location()[0] << ":" << particle->get_location()[1] << std::endl;
          if (std::get<0>(particle_status) == particle->get_id())
            {
              found = true;
              std::get<1>(particle_status) = 1;
              std::get<2>(particle_status) = particle->get_location();
            }
        }
      AssertThrow(found, ExcMessage("could not find which particle was lost..."));
    }


    template <int dim>
    std::vector<Tensor<1,dim>>
    DikeInjection<dim>::compute_stress_largest_eigenvector(//std::vector<std::unique_ptr<SolutionEvaluator<dim>>>& evaluators,
      std::vector<typename DoFHandler<dim>::active_cell_iterator> &cells,
      std::vector<Point<dim>> &positions,
      std::vector<Point<dim>> &reference_positions,
      const LinearAlgebra::BlockVector &input_solution)
    {



      const UpdateFlags update_flags = update_values | update_gradients;
      std::unique_ptr<SolutionEvaluator<dim>> evaluator = construct_solution_evaluator(*this,update_flags);

      // Todo: assert that vectors of cells, positions and reference positino are the same size

      // FEPointEvaluation uses different evaluation flags than the common UpdateFlags.
      // Translate between the two.
      std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags (this->introspection().n_components, EvaluationFlags::nothing);

      for (unsigned int i=0; i<this->introspection().n_components; ++i)
        {
          evaluation_flags[i] |= EvaluationFlags::values;
          evaluation_flags[i] |= EvaluationFlags::gradients;
        }

      std::vector<Tensor< 1, dim, double >> stress_largest_eigenvectors;

      for (unsigned int cell_i = 0; cell_i < cells.size(); ++cell_i)
        {
          //std::cout << "flag 10" << std::endl;
          Assert(cells[cell_i].state() == IteratorState::valid,ExcMessage("Cell state is not valid."));
          //std::cout << "flag 11" << std::endl;
          small_vector<double,50> solution_values(this->get_fe().dofs_per_cell);
          cells[cell_i]->get_dof_values(input_solution,
                                        solution_values.begin(),
                                        solution_values.end());
          //std::cout << "flag 12" << std::endl;

          std::vector<std::vector<double>> solution(this->get_fe().dofs_per_cell);
          //std::cout << "flag 13" << std::endl;
          solution.resize(1,std::vector<double>(evaluator->n_components(), numbers::signaling_nan<double>()));
          //std::cout << "flag 14 evaluator->n_components() = " << evaluator->n_components() << ", solution.size() = " << solution.size() << std::endl;
          //std::cout << "solution[0].size() = " << solution[0].size() << std::endl;
          solution[0] = std::vector<double>(evaluator->n_components(), numbers::signaling_nan<double>());
          //std::cout << "solution[0].size() = " << solution[0].size() << std::endl;

          std::vector<std::vector<Tensor<1,dim>>> gradients(this->get_fe().dofs_per_cell);
          //std::cout << "flag 15" << std::endl;
          gradients.resize(1,std::vector<Tensor<1,dim>>(evaluator->n_components(), numbers::signaling_nan<Tensor<1,dim>>()));
          gradients[0]=std::vector<Tensor<1,dim>>(evaluator->n_components(), numbers::signaling_nan<Tensor<1,dim>>());
          //std::cout << "flag 16" << std::endl;

          evaluator->reinit(cells[cell_i], reference_positions);
          //std::cout << "flag 17" << std::endl;
          evaluator->evaluate({solution_values.data(),solution_values.size()},evaluation_flags);
          //std::cout << "flag 45"<< std::endl;
          //std::cout << "solution.size() = " << solution.size() << std::endl;
          //std::cout << "&solution[0] = " << &solution[0] << std::endl;
          //std::cout << "solution[0].size() = " << solution[0].size() << std::endl;
          //std::cout << "&solution[0][0] = " << &solution[0][0] << std::endl;
//           //std::cout << "solution[0][0] = " << solution[0][0] << std::endl;
          evaluator->get_solution(0, {&solution[0][0],solution[0].size()}, evaluation_flags);
          //std::cout << "flag 46"<< std::endl;
          //std::cout << "&solution[0] = " << &solution[0] << std::endl;
          //std::cout << "solution[0].size() = " << solution[0].size() << std::endl;
          //std::cout << "&solution[0][0] = " << &solution[0][0] << std::endl;
          //std::cout << "solution[0][0] = " << solution[0][0] << std::endl;
          evaluator->get_gradients(0, {&gradients[0][0],gradients[0].size()}, evaluation_flags);
          //std::cout << "flag 47"<< std::endl;
          //std::cout << "&solution[0] = " << &solution[0] << std::endl;
          //std::cout << "solution[0].size() = " << solution[0].size() << std::endl;
          //std::cout << "&solution[0][0] = " << &solution[0][0] << std::endl;
          //std::cout << "solution[0][0] = " << solution[0][0] << std::endl;

          Tensor<1,dim> velocity;

          for (unsigned int i = 0; i < dim; ++i)
            velocity[i] = solution_values[this->introspection().component_indices.velocities[i]];

          //std::cout << "flag 48"<< std::endl;
          // get velocity gradient tensor.
          Tensor<2,dim> velocity_gradient;
          for (unsigned int i = 0; i < dim; ++i)
            velocity_gradient[i] = gradients[0][this->introspection().component_indices.velocities[i]];

          //std::cout << "flag 49"<< std::endl;
          // Calculate strain rate from velocity gradients
          const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);
          const SymmetricTensor<2,dim> deviatoric_strain_rate
            = (this->get_material_model().is_compressible()
               ?
               strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
               :
               strain_rate);

          const double pressure = solution[0][this->introspection().component_indices.pressure];

          const double temperature =solution[0][this->introspection().component_indices.temperature];

          //std::cout << "flag 50"<< std::endl;
          //std::cout << "temperature = " << temperature << ", pressure = " << pressure << ",solution[0][0] = " << solution[0][0] << ",1:" << solution[0][1] << ",2:" << solution[0][2] << ",3:" << solution[0][3] << ", old: " << solution[0][this->introspection().component_indices.temperature] << ", this->introspection().component_indices.temperature = " << this->introspection().component_indices.temperature << ", pres index = " << this->introspection().component_indices.pressure << ", solution_values = " << solution_values[0] << ":" << solution_values[1] << ":" << solution_values[2] << ":" << solution_values[3] << std::endl;

          // get the composition of the particle
          std::vector<double> compositions;
          for (unsigned int i = 0; i < this->n_compositional_fields(); ++i)
            {
              const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
              compositions.push_back(solution[0][solution_component]);
            }


          // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
          // adding an extra dimension which is zero.
          SymmetricTensor<2,3> strain_rate_3d;
          strain_rate_3d[0][0] = strain_rate[0][0];
          strain_rate_3d[0][1] = strain_rate[0][1];
          //sym: strain_rate_3d[1][0] = strain_rate[1][0];
          strain_rate_3d[1][1] = strain_rate[1][1];

          if (dim == 3)
            {
              strain_rate_3d[0][2] = strain_rate[0][2];
              strain_rate_3d[1][2] = strain_rate[1][2];
              //sym: strain_rate_3d[2][0] = strain_rate[0][2];
              //sym: strain_rate_3d[2][1] = strain_rate[1][2];
              strain_rate_3d[2][2] = strain_rate[2][2];
            }
          Tensor<2,3> velocity_gradient_3d;
          velocity_gradient_3d[0][0] = velocity_gradient[0][0];
          velocity_gradient_3d[0][1] = velocity_gradient[0][1];
          velocity_gradient_3d[1][0] = velocity_gradient[1][0];
          velocity_gradient_3d[1][1] = velocity_gradient[1][1];
          if (dim == 3)
            {
              velocity_gradient_3d[0][2] = velocity_gradient[0][2];
              velocity_gradient_3d[1][2] = velocity_gradient[1][2];
              velocity_gradient_3d[2][0] = velocity_gradient[2][0];
              velocity_gradient_3d[2][1] = velocity_gradient[2][1];
              velocity_gradient_3d[2][2] = velocity_gradient[2][2];
            }

          //std::cout << "flag 51"<< std::endl;
          // compute the viscosity
          MaterialModel::MaterialModelInputs<dim> material_model_inputs(1,this->n_compositional_fields());
          material_model_inputs.position[0] = positions[0];
          material_model_inputs.temperature[0] = temperature;
          material_model_inputs.pressure[0] = pressure;
          material_model_inputs.velocity[0] = velocity;
          material_model_inputs.composition[0] = compositions;
          material_model_inputs.strain_rate[0] = strain_rate;
          material_model_inputs.current_cell = cells[cell_i];
          //std::cout << "flag 52"<< std::endl;

          MaterialModel::MaterialModelOutputs<dim> material_model_outputs(1,this->n_compositional_fields());
          this->get_material_model().evaluate(material_model_inputs, material_model_outputs);
          double eta = material_model_outputs.viscosities[0];

          const SymmetricTensor<2,dim>  stress = -2. * eta * deviatoric_strain_rate;
          //const std::array< std::pair< double, Tensor< 1, dim, double >>, std::integral_constant< int, dim >::value > stress_eigenvectors = dealii::eigenvectors(stress);
          stress_largest_eigenvectors.emplace_back(dealii::eigenvectors(stress)[0].second);

          //std::cout << "flag 53"<< std::endl;
          //std::cout << "size eigenvectors = " <<  dealii::eigenvectors(stress)[0].first
          //          << ", " <<dealii::eigenvectors(stress)[1].first << std::endl;

          // now we have the largest stress eigenvector. We need to deterine what is up.
          Tensor<1,dim> gravity_vector = this->get_gravity_model().gravity_vector(positions[0])/this->get_gravity_model().gravity_vector(positions[0]).norm();

          double angle = stress_largest_eigenvectors.back() * gravity_vector;
          //std::cout << "positions[0] = " << positions[0] << ", gravity_vector = " << gravity_vector << ", angle = " << angle << ":" << angle*180./numbers::PI
          //          << ", stress_largest_eigenvectors = " << stress_largest_eigenvectors << ",eta = " << eta << ", deviatoric_strain_rate = " << deviatoric_strain_rate << std::endl;
          if (std::fabs(angle) < 0.5*numbers::PI)
            {
              stress_largest_eigenvectors.back() *= -1;
            }
          //std::cout << "flag 60"<< std::endl;

        }

      //std::vector<EvaluationFlags::EvaluationFlags> evaluation_flags (1, EvaluationFlags::nothing);
      //evaluation_flags[0] |= EvaluationFlags::values;
      //evaluation_flags[0] |= EvaluationFlags::gradients;
      //evaluator->reinit(cell, positions, {solution_values.data(), solution_values.size()}, update_flags);
      //std::cout << "ifcsle flag 1: positions.size() = " << positions.size() << ", this->introspection().n_components = " << this->introspection().n_components << std::endl;
      //Assert(cell.state() == IteratorState::valid,ExcMessage("Cell state is not valid."));

      //std::vector<Vector<double>> solution(this->get_fe().dofs_per_cell);
      //solution.resize(1,Vector<double>(this->introspection().n_components));
      //small_vector<double,50> solution_values(this->get_fe().dofs_per_cell);
      //cell->get_dof_values(this->get_solution(),
      //                     solution_values.begin(),
      //                     solution_values.end());
      //solution_values.resize(1,small_vector<double,50>(evaluator.n_components(), numbers::signaling_nan<double>()));
      //std::vector<small_vector<double,50>> solution(this->get_fe().dofs_per_cell);
      //solution.resize(1,small_vector<double,50>(evaluator->n_components(), numbers::signaling_nan<double>()));

      //std::vector<std::vector<Tensor<1,dim>>> gradients;
      //gradients.resize(1,std::vector<Tensor<1,dim>>(this->introspection().n_components));
      //small_vector<small_vector<Tensor<1,dim>,50>> gradients(this->get_fe().dofs_per_cell);
      //gradients.resize(1,small_vector<Tensor<1,dim>,50>(evaluator->n_components(), numbers::signaling_nan<Tensor<1,dim>>()));

      //std::cout << "ifcsle flag 5" << std::endl;
      //for (unsigned int i = 0; i<1; ++i)
      //{
      //  //evaluator->evaluate({&solution[0][0],solution[0].size()}, evaluation_flags);
      //  evaluator->evaluate({solution_values.data(),solution_values.size()},evaluation_flags);
      //  // Evaluate the solution, but only if it is requested in the update_flags
      //  //if (update_flags & update_values)
      //  evaluator->get_solution(0, {&solution[0][0],solution[0].size()}, evaluation_flags);
      //
      //  //std::cout << "ifcsle flag 6" << std::endl;
      //  // Evaluate the gradients, but only if they are requested in the update_flags
      //  //if (update_flags & update_gradients)
      //  evaluator->get_gradients(0, {&gradients[0][0],gradients[0].size()}, evaluation_flags);
      //}

      //std::cout << "ifcsle flag 7" << std::endl;
      // get presure, temp, etc

      // need access to the pressure, viscosity,
      // get velocity

      /*Tensor<1,dim> velocity;

      for (unsigned int i = 0; i < dim; ++i)
        velocity[i] = solution_values[this->introspection().component_indices.velocities[i]];

      // get velocity gradient tensor.
      Tensor<2,dim> velocity_gradient;
      for (unsigned int i = 0; i < dim; ++i)
        velocity_gradient[i] = gradients[0][this->introspection().component_indices.velocities[i]];

      // Calculate strain rate from velocity gradients
      const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);
      const SymmetricTensor<2,dim> deviatoric_strain_rate
        = (this->get_material_model().is_compressible()
           ?
           strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
           :
           strain_rate);

      const double pressure = solution[0][this->introspection().component_indices.pressure];

      //std::vector<double> temperature_values = {1};
      //fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(), temperature_values);
      const double temperature =solution[0][this->introspection().component_indices.temperature];

      //std::cout << "temperature = " << temperature << ", pressure = " << pressure << ",solution[0][0] = " << solution[0][0] << ",1:" << solution[0][1] << ",2:" << solution[0][2] << ",3:" << solution[0][3] << ", old: " << solution[0][this->introspection().component_indices.temperature] << ", this->introspection().component_indices.temperature = " << this->introspection().component_indices.temperature << ", pres index = " << this->introspection().component_indices.pressure << ", solution_values = " << solution_values[0] << ":" << solution_values[1] << ":" << solution_values[2] << ":" << solution_values[3] << std::endl;

      // get the composition of the particle
      std::vector<double> compositions;
      for (unsigned int i = 0; i < this->n_compositional_fields(); ++i)
        {
          const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
          compositions.push_back(solution[0][solution_component]);
        }

      //const double dt = this->get_timestep();

      // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
      // adding an extra dimension which is zero.
      SymmetricTensor<2,3> strain_rate_3d;
      strain_rate_3d[0][0] = strain_rate[0][0];
      strain_rate_3d[0][1] = strain_rate[0][1];
      //sym: strain_rate_3d[1][0] = strain_rate[1][0];
      strain_rate_3d[1][1] = strain_rate[1][1];

      if (dim == 3)
        {
          strain_rate_3d[0][2] = strain_rate[0][2];
          strain_rate_3d[1][2] = strain_rate[1][2];
          //sym: strain_rate_3d[2][0] = strain_rate[0][2];
          //sym: strain_rate_3d[2][1] = strain_rate[1][2];
          strain_rate_3d[2][2] = strain_rate[2][2];
        }
      Tensor<2,3> velocity_gradient_3d;
      velocity_gradient_3d[0][0] = velocity_gradient[0][0];
      velocity_gradient_3d[0][1] = velocity_gradient[0][1];
      velocity_gradient_3d[1][0] = velocity_gradient[1][0];
      velocity_gradient_3d[1][1] = velocity_gradient[1][1];
      if (dim == 3)
        {
          velocity_gradient_3d[0][2] = velocity_gradient[0][2];
          velocity_gradient_3d[1][2] = velocity_gradient[1][2];
          velocity_gradient_3d[2][0] = velocity_gradient[2][0];
          velocity_gradient_3d[2][1] = velocity_gradient[2][1];
          velocity_gradient_3d[2][2] = velocity_gradient[2][2];
        }

      // compute the viscosity
      MaterialModel::MaterialModelInputs<dim> material_model_inputs(1,this->n_compositional_fields());
      material_model_inputs.position[0] = positions[0];
      material_model_inputs.temperature[0] = temperature;
      material_model_inputs.pressure[0] = pressure;
      material_model_inputs.velocity[0] = velocity;
      material_model_inputs.composition[0] = compositions;
      material_model_inputs.strain_rate[0] = strain_rate;
      material_model_inputs.current_cell = cell;
      //std::cout << "position = " << positions[0] << ", temperature = " << temperature << ", pressure = " << pressure
      //<< ", velocity = " << velocity << ", strain_rate = " << strain_rate << std::endl;

      MaterialModel::MaterialModelOutputs<dim> material_model_outputs(1,this->n_compositional_fields());
      this->get_material_model().evaluate(material_model_inputs, material_model_outputs);
      double eta = material_model_outputs.viscosities[0];

      //const SymmetricTensor<2,dim> stress = 2*eta*deviatoric_strain_rate +
      //                                      pressure * unit_symmetric_tensor<dim>();

      //                 const SymmetricTensor<2,dim> deviatoric_strain_rate
      // = (this->get_material_model().is_compressible()
      //    ?
      //    strain_rate - 1./3. * trace(strain_rate) * unit_symmetric_tensor<dim>()
      //    :
      //    strain_rate);

      // Compressive stress is positive in geoscience applications
      const SymmetricTensor<2,dim>  stress = -2. * eta * deviatoric_strain_rate;
      //const std::array< std::pair< double, Tensor< 1, dim, double >>, std::integral_constant< int, dim >::value > stress_eigenvectors = dealii::eigenvectors(stress);
      Tensor< 1, dim, double > stress_largest_eigenvectors = dealii::eigenvectors(stress)[0].second;

      //std::cout << "size eigenvectors = " <<  dealii::eigenvectors(stress)[0].first
      //          << ", " <<dealii::eigenvectors(stress)[1].first << std::endl;

      // now we have the largest stress eigenvector. We need to deterine what is up.
      Tensor<1,dim> gravity_vector = this->get_gravity_model().gravity_vector(positions[0])/this->get_gravity_model().gravity_vector(positions[0]).norm();

      double angle = stress_largest_eigenvectors * gravity_vector;
      //std::cout << "positions[0] = " << positions[0] << ", gravity_vector = " << gravity_vector << ", angle = " << angle << ":" << angle*180./numbers::PI
      //          << ", stress_largest_eigenvectors = " << stress_largest_eigenvectors << ",eta = " << eta << ", deviatoric_strain_rate = " << deviatoric_strain_rate << std::endl;
      if (std::fabs(angle) < 0.5*numbers::PI)
        {
          stress_largest_eigenvectors *= -1;
        }*/

      //std::cout << "ifcsle flag end" << std::endl;
      return stress_largest_eigenvectors;

    }

    template <int dim>
    void
    DikeInjection<dim>::update()
    {
      base_model->update();

      if (!particle_handler)
        {
          particle_handler = std::make_unique<Particles::ParticleHandler<dim>>(this->get_triangulation(), this->get_mapping(),Particle::Integrator::RK4<dim>::n_integrator_properties);
          particle_handler->signals.particle_lost.connect([&] (const typename Particles::ParticleIterator<dim> &particle, const typename Triangulation<dim>::active_cell_iterator &cell)
          {
            this->set_particle_lost(particle, cell);
          });
        }

      // find_active_cell_around_point(mapping, tria, point);
      // dealii tests: particle_handler_03 -> inefficient particle inser
      // use instead: insert particle(posiiton, reference_positin, particle_index, cell)
      //                                        ^ don't care, can be anything, ^ need to be unique if there are multiple particles at the same time
      //                                                                       ^get_next_free_particle_index(), call updated chashed_numbers() after

      // advection: local_integrate_step(b,e,solution, currently_lin_point, dt)
      // call sort_particles_into_subdomains_and_cells() afterwards

      // connect to signal if particle leaves the domain: signals.particle_lost
      // let the function change a member variable of the class

      // Maybe move to MaterialModel Utilities

      // Use dealii mpi functions


      // we get time passed as seconds (always) but may want
      // to reinterpret it in years
      //if (this->convert_output_to_years())
      //  injection_function.set_time (this->get_time() / year_in_seconds);
      //else
      //  injection_function.set_time (this->get_time());

      //bool enable_diking = true;

      // Set mpi variables
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      //particle_lost = false;
      dike_locations.resize(2);
      dike_locations[0].resize(0);
      dike_locations[1].resize(0);

      // TODO: To know if we need diking, we need to compute whether or not we have melting.
      if (dim == 2)
        {
          dike_locations[0].emplace_back(Point<dim>(-1370.4997314869,40489.36393586183));//(0,50225));
          dike_locations[1].emplace_back(Point<dim>(-20000,40489.36393586183));//(0,50225));
        }
      else
        {
          dike_locations[0].emplace_back(Point<dim>(-1370.4997314869,50e3,40489.36393586183));//(0,50225));
        }
      // If we found the correct cell on this MPI process, we have found the right cell.
      //Assert(cell_it.first.state() == IteratorState::valid && cell_it.first->is_locally_owned(), ExcMessage("Internal error: could not find cell to place initial point."));

      //std::cout << "world_rank = " << world_rank << "/" << world_size << ": Flag 0: enable_random_dike_generation = " << enable_random_dike_generation
      //<< ", this->get_timestep_number() = " << this->get_timestep_number() << ", cell_it.first.state() = " << cell_it.first.state()
      ////<< ", cell_it.first->is_locally_owned() =" << cell_it.first->is_locally_owned()
      //<< std::endl;

      if (enable_random_dike_generation && this->get_timestep_number() > 0)// && cell_it.first.state() == IteratorState::valid)// && cell_it.first->is_locally_owned())
        {
          //std::cout << "dike_locations.size() = " << dike_locations.size() << std::endl;
          for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
            {
              std::pair<const typename parallel::distributed::Triangulation<dim>::active_cell_iterator,Point<dim>> cell_it_start = GridTools::find_active_cell_around_point<>(this->get_mapping(), this->get_triangulation(), dike_locations[dike_i].back());


              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 1" << std::endl;

              unsigned int next_free_id = 0;
              if (cell_it_start.first.state() == IteratorState::valid && cell_it_start.first->is_locally_owned())
                {
                  next_free_id = particle_handler->get_next_free_particle_index();
                  unsigned int next_free_id_sum = Utilities::MPI::sum(next_free_id,this->get_mpi_communicator());
                  Assert(next_free_id == next_free_id_sum, ExcMessage("mpi internal error"));
                  particle_statuses.emplace_back(std::tuple<unsigned int,unsigned int,Point<dim>> {next_free_id, 0,Point<dim>()});
                  particle_handler->insert_particle(dike_locations[dike_i].back(),cell_it_start.second,next_free_id, cell_it_start.first);
                  //std::cout << "next_free_id = " << next_free_id << std::endl;
                  particle_handler->update_cached_numbers();
                }
              else
                {
                  next_free_id = Utilities::MPI::sum(next_free_id,this->get_mpi_communicator());
                  particle_statuses.emplace_back(std::tuple<unsigned int,unsigned int,Point<dim>> {next_free_id, 0,Point<dim>()});
                  particle_handler->update_cached_numbers();
                }

              // TODO: Is this safe in parallel? Do I need to call  update_cached_numbers()?
              //       Add an Assert(next_free_id==particle_handler->get_next_free_particle_index(),ExcMessage(...)) before update_cached_numbers() to check if the number has been updated in between?

            }
          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 1.5" << std::endl;
          //particle_handler->update_cached_numbers();
          particle_handler->sort_particles_into_subdomains_and_cells();
          // get the stress at the point
          // get the solutions and gradients
          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 2" << std::endl;

          const UpdateFlags update_flags = update_values | update_gradients;//property_manager->get_needed_update_flags();

          //std::unique_ptr<SolutionEvaluator<dim>> evaluator = construct_solution_evaluator(*this,
          //                                                     update_flags);


          //const Quadrature<dim> quadrature_formula (std::vector<Point<dim>>(1,particle_handler->begin()->get_reference_location()));

          //const unsigned int n_q_points =  quadrature_formula.size();
          //FEValues<dim> fe_values (this->get_mapping(), this->get_fe(),  quadrature_formula,
          //                         update_flags);

          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3" << std::endl;
          // loop untill point is no longer in any cell;
          // todo: or max number?
          int iteration = 0;
          unsigned int n_active_particles = dike_locations.size();
         //std::cout << "n_active_particles = " << n_active_particles << std::endl;
          while (n_active_particles > 0)
            {
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.1" << std::endl;
              iteration++;
              if (!(iteration < 5000))
                {
                  std::string concat = "";
                 //std::cout << "Failing at iteration " << iteration << ", current dike path: ";
                  for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
                    {
                     //std::cout << std::endl << "dike " << dike_i << ": ";
                      for (auto coords : dike_locations[dike_i])
                        {
                          //concat += std::to_string(coords);
                         //std::cout << coords << ", ";
                        }
                    }
                  AssertThrow(iteration < 5000, ExcMessage ("too many iterations for the dike to reach the surface. rank: " + std::to_string(world_rank)));
                }
              //std::vector<Point<dim>> positions = {dim == 3 ? Point<dim>(0,0,0) : Point<dim>(0,0)};
              //std::vector<Point<dim>> reference_positions = {dim == 3 ? Point<dim>(0,0,0) : Point<dim>(0,0)};

              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.2, cell_it.first.state() = " << cell_it.first.state() << ", IteratorState::valid = " << IteratorState::valid << std::endl;
              //if (particle_handler->n_locally_owned_particles() > 0) //        cell_it.first.state() == IteratorState::valid)
              //  {
              //    //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.3, cell_it.first.state() = " << cell_it.first.state() << ", IteratorState::valid = " << IteratorState::valid << ", particle_handler->begin() = " << particle_handler->begin()->get_surrounding_cell().state() << std::endl;
              //    positions[0] = particle_handler->begin()->get_location();
              //    reference_positions[0] = particle_handler->begin()->get_reference_location();
              //    //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.4" << std::endl;
              //  } //? {{particle_handler->begin()->get_reference_location()}} : {};

              //std::cout << " old position: " << particle_handler->begin()->get_location() << std::endl;


              std::vector<Point<dim>> new_dike_points(particle_statuses.size(),Point<dim>());
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 4" << std::endl;
              size_t iter2 = 0;
              do
                {
                  iter2++;
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 4.5" << std::endl;
                  //std::cout << iteration << ":" << iter2 << std::endl;//"(1): particle lost = " << particle_lost << std::endl;
                  particle_handler->sort_particles_into_subdomains_and_cells();
                  //std::cout << iteration << ":" << iter2 << "(2): parwhileticle lost = " << particle_lost << std::endl;
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 5" << std::endl;
                  //unsigned int particle_lost_int = (unsigned int)particle_lost;
                  //std::cout << iteration << ":" << iter2 << "(3): parwhileticle lost = " << particle_lost_int << std::endl;

                  // recmpute active particles
                  n_active_particles = 0;
                  //std::cout << "particle_statuses = " << particle_statuses.size() << std::endl;
                  for (auto &particle_status : particle_statuses)
                    {
                      //std::cout << "a particle" << std::endl;
                      //if (std::get<1>(particle_status) == 0 || std::get<1>(particle_status) == 1)
                      {
                        // check whether this is still active on all processes (0 is active, so if sum is not zero, it is inactive)
                        //std::cout << "std::get<0:1>(particle_status) = " << std::get<0>(particle_status) << ":" << std::get<1>(particle_status) << std::endl;
                        if (Utilities::MPI::sum(std::get<1>(particle_status),this->get_mpi_communicator()))
                          {
                            // particle lost on some processor, so set it to 1 on all processors
                            //std::cout << "not active anymore: " << Utilities::MPI::sum(std::get<1>(particle_status),this->get_mpi_communicator()) << std::endl;
                            std::get<1>(particle_status) = 1;
                          }
                        else
                          {
                            //std::cout << "an active particle!" << std::endl;
                            n_active_particles++;
                          }
                      }
                    }
                  //std::cout << "new active particles = " << n_active_particles << std::endl;

                  unsigned int particle_lost = 0;
                  particle_lost = Utilities::MPI::sum((unsigned int)particle_lost,this->get_mpi_communicator());
                  //std::cout << iteration << ":" << iter2 << "(4): parwhileticle lost = " << particle_lost_int << std::endl;
                  if (n_active_particles == 0)
                    {
                      //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 6" << std::endl;

                      //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7" << std::endl;
                      //particle_handler->sort_particles_into_subdomains_and_cells();
                      ////std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7.1" << std::endl;
                      //Utilities::MPI::sum(particle_lost_int,this->get_mpi_communicator());
                      ////std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7.2" << std::endl;
                      do
                        {
                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7.3" << std::endl;
                          //particle_handler->sort_particles_into_subdomains_and_cells();
                          ////std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7.1" << std::endl;
                          //Utilities::MPI::sum(particle_lost_int,this->get_mpi_communicator());
                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 7.2" << std::endl;
                          //keep iterating to make sure the iteratino step  is back at 0
                          // TODO: create a function to rest the integration step.
                        }
                      while (particle_integrator->new_integration_step());
                      break;
                    }
                  //if (particle_handler->n_locally_owned_particles() == 0)
                  //  {
                  //    continue;
                  //  }

                  //std::cout << iteration << "(3): particle lost = " << particle_lost << std::endl;

                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << std::endl;//": Flag 8, positions.size() = " << positions.size() << std::endl;

                  std::vector<Point<dim>> positions;// = {dim == 3 ? Point<dim>(0,0,0) : Point<dim>(0,0)};
                  std::vector<Point<dim>> reference_positions;// = {dim == 3 ? Point<dim>(0,0,0) : Point<dim>(0,0)};
                  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
                  //std::vector<small_vector<double>> solution_values;
                  //std::vector<std::unique_ptr<SolutionEvaluator<dim>>> evaluators;

                  for (auto particle_it = particle_handler->begin(); particle_it != particle_handler->end(); ++particle_it)
                    {
                      if (particle_it->get_surrounding_cell().state() == IteratorState::valid)
                        {

                          cells.emplace_back(typename DoFHandler<dim>::active_cell_iterator(*particle_handler->begin()->get_surrounding_cell(),&(this->get_dof_handler())));

                          //std::cout << iteration << ": ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9, positions.size() = " << positions.size()
                          //<< ", cell_it.first.state() = " << cell->state() << ":" << IteratorState::valid << std::endl;

                          //Assert(positions.size() == 1, ExcMessage("Internal error."));
                          //Assert(reference_positions.size() == 1, ExcMessage("Internal error."));
                          positions.emplace_back(particle_it->get_location());
                          reference_positions.emplace_back(particle_it->get_reference_location());
                          //Assert(cell->state() == IteratorState::valid, ExcMessage("internal error"));

                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.5" << std::endl;
                          //solution_values.emplace_back(this->get_fe().dofs_per_cell);

                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.6" << std::endl;
                          //cells->end()->get_dof_values(this->get_solution(),
                          //                     solution_values.begin(),
                          //                     solution_values.end());

                          //evaluators.emplace_back(construct_solution_evaluator(*this,
                          //                                       update_flags));
                          //evaluators.back()->reinit(cells.back(), reference_positions.back());
                        }
                    }
                  //{



                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.7" << std::endl;

                  //fe_values.reinit(cell);
                  //evaluator->reinit(cell, reference_positions);


                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 10" << std::endl;
                  // function here
                  //Tensor<1,dim> solution_stress =
                  if (cells.size() > 0)
                    {
                      std::vector<Tensor<1,dim>> solution_stress = compute_stress_largest_eigenvector(cells,positions,reference_positions,this->get_solution());

                      //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 11" << std::endl;
                      //cell->get_dof_values(this->get_current_linearization_point(),
                      //                     solution_values.begin(),
                      //                     solution_values.end());
                      //
                      //evaluator->reinit(cell, reference_positions);

                      std::vector<Tensor<1,dim>> current_linerization_point_stress = compute_stress_largest_eigenvector(cells,positions,reference_positions,this->get_current_linearization_point());

                      // set the new point at half the cell size away from the current point and check if that is still in the domain.
                      const double distance = 613.181;//cell->minimum_vertex_distance()*this->get_parameters().CFL_number;

                      //auto old_position = particle_it->get_location();
                      //}

                     //std::cout << iteration << ": world_rank = " << world_rank << "/" << world_size << ", old position = " << particle_handler->begin()->get_location() << std::endl;
                      particle_integrator->local_integrate_step(particle_handler->begin(),particle_handler->end(),solution_stress, current_linerization_point_stress, distance);
                    }
                  //std::cout << iteration << ": world_rank = " << world_rank << "/" << world_size << ", solution_stress = " << solution_stress[0] << ", current_linerization_point_stress = " << current_linerization_point_stress[0]
                  //          << ", new position: " << particle_handler->begin()->get_location() << ", distance = " << distance << ", actual distance = " << (old_position-particle_handler->begin()->get_location()).norm() << std::endl;

                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12" << std::endl;
                  //}
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12.25" << std::endl;
                  //  }
                  //}

                  /*if (particle_handler->n_locally_owned_particles() > 0 && particle_handler->begin()->get_surrounding_cell().state() == IteratorState::valid)
                    {
                      typename DoFHandler<dim>::active_cell_iterator cell = typename DoFHandler<dim>::active_cell_iterator(*particle_handler->begin()->get_surrounding_cell(),&(this->get_dof_handler()));

                      //std::cout << iteration << ": ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9, positions.size() = " << positions.size()
                      //<< ", cell_it.first.state() = " << cell->state() << ":" << IteratorState::valid << std::endl;

                      Assert(positions.size() == 1, ExcMessage("Internal error."));
                      Assert(reference_positions.size() == 1, ExcMessage("Internal error."));
                      positions[0] = particle_handler->begin()->get_location();
                      reference_positions[0] = particle_handler->begin()->get_reference_location();
                      Assert(cell->state() == IteratorState::valid, ExcMessage("internal error"));

                      {
                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.5" << std::endl;
                        small_vector<double> solution_values(this->get_fe().dofs_per_cell);

                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.6" << std::endl;
                        cell->get_dof_values(this->get_solution(),
                                             solution_values.begin(),
                                             solution_values.end());


                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9.7" << std::endl;

                        //fe_values.reinit(cell);
                        evaluator->reinit(cell, reference_positions);


                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 10" << std::endl;
                        // function here
                        //Tensor<1,dim> solution_stress =
                        std::vector<Tensor<1,dim>> solution_stress = compute_stress_largest_eigenvector(evaluator,cell,positions,solution_values);;

                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 11" << std::endl;
                        cell->get_dof_values(this->get_current_linearization_point(),
                                             solution_values.begin(),
                                             solution_values.end());

                        evaluator->reinit(cell, reference_positions);

                        std::vector<Tensor<1,dim>> current_linerization_point_stress = compute_stress_largest_eigenvector(evaluator,cell,positions,solution_values);

                        // set the new point at half the cell size away from the current point and check if that is still in the domain.
                        const double distance = 613.181;//cell->minimum_vertex_distance()*this->get_parameters().CFL_number;

                        auto old_position = particle_handler->begin()->get_location();

                        //std::cout << iteration << ": world_rank = " << world_rank << "/" << world_size << ", old position = " << particle_handler->begin()->get_location() << std::endl;
                        particle_integrator->local_integrate_step(particle_handler->begin(),particle_handler->end(),solution_stress, current_linerization_point_stress, distance);

                        //std::cout << iteration << ": world_rank = " << world_rank << "/" << world_size << ", solution_stress = " << solution_stress[0] << ", current_linerization_point_stress = " << current_linerization_point_stress[0]
                        //          << ", new position: " << particle_handler->begin()->get_location() << ", distance = " << distance << ", actual distance = " << (old_position-particle_handler->begin()->get_location()).norm() << std::endl;

                        //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12" << std::endl;
                      }
                      //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12.25" << std::endl;
                    }*/
                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12.5" << std::endl;
                }
              while (particle_integrator->new_integration_step());

             //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 13: " << std::endl; //particle_lost = " << particle_lost << ", cell_it.first.state() = " << cell_it.first.state() << std::endl;
              //if (particle_handler->n_locally_owned_particles() > 0) //cell_it.first.state() == IteratorState::valid) {
              //  {
              //    //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 13.5: particle_lost = " << particle_lost << std::endl;
              //    for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
              //    {
              //      new_dike_point = particle_lost ? particle_lost_location : particle_handler->begin()->get_location();
              //    }
              //    //int world_rank;
              //    //MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
              //    //Utilities::MPI::broadcast(this->get_mpi_communicator(),n_active_particles,world_rank);
              //  }
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 30" << std::endl;
              /*for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
                {
                  Utilities::MPI::sum(get<1>(particle_statuses[dike_i]),this->get_mpi_communicator(),get<1>(particle_statuses[dike_i]));

                  if (get<1>(particle_statuses[dike_i] == 1))
                    {
                      get<2>(particle_statuses[dike_i]) = 2;
                    }

                  // if particle is not lost add a new point to the dike
                  if (get<1>(particle_statuses[dike_i]) == 0)
                    dike_locations[dike_i].emplace_back(new_dike_point);
                }*/



              // recmpute active particles
              n_active_particles = 0;
              for (unsigned int dike_i = 0; dike_i < particle_statuses.size(); ++dike_i)
                {
                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ":Flag 14: dike = " << dike_i << std::endl;
                  //if(std::get<1>(particle_statuses[dike_i]) == 1){
                  //}
                  if (std::get<1>(particle_statuses[dike_i]) == 0 || std::get<1>(particle_statuses[dike_i]) == 1)
                    {
                     //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ":Flag 15: dike = " << dike_i << ", particle_statuses[dike_i] = " << std::get<1>(particle_statuses[dike_i]) << std::endl;
                      // check whether this is still active on all processes (0 is active, so if sum is not zero, it is inactive)
                      if (Utilities::MPI::sum(std::get<1>(particle_statuses[dike_i]),this->get_mpi_communicator()))
                        {
                         //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ":Flag 16: dike = " << dike_i << std::endl;
                          // particle lost on some processor, so set it to 1 on all processors
                          Point<dim> new_dike_location = std::get<2>(particle_statuses[dike_i]);
                          for (unsigned int dim_i = 0; dim_i < dim; ++dim_i)
                            {
                              new_dike_location[dim_i] = Utilities::MPI::sum(new_dike_location[dim_i],this->get_mpi_communicator());
                            }
                          dike_locations[dike_i].emplace_back(new_dike_location);
                          std::get<1>(particle_statuses[dike_i]) = 2;

                         //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 19: dike = " << dike_i << std::endl;
                        }
                      else
                        {
                         //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 20: dike = " << dike_i << std::endl;
                          n_active_particles++;

                          //new_dike_points[dike_i] = particle_lost ? particle_lost_location : particle_handler->[dike_i]->get_location();
                          Point<dim> new_dike_location = Point<dim>();
                          for (auto it = particle_handler->begin(); it != particle_handler->end(); ++it)
                            {
                             //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 21: dike = " << dike_i << std::endl;
                              // if the indexes are equal we found a match
                              //std::cout << "std::get<0>(particle_statuses[dike_i]) = " << std::get<0>(particle_statuses[dike_i]) << ", it->get_id() = " << it->get_id() << std::endl;
                              if (std::get<0>(particle_statuses[dike_i]) == it->get_id())
                                {
                                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 22: dike = " << dike_i << ", it->get_id() = " << it->get_id() <<std::endl;
                                  new_dike_location = it->get_location();
                                  break;
                                }
                            }
                         //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 23.5: dike = " << dike_i << std::endl;

                          for (unsigned int dim_i = 0; dim_i < dim; ++dim_i)
                            {
                             //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 24: dike = " << dike_i << ", dim_i = " << dim_i << std::endl;
                              new_dike_location[dim_i] = Utilities::MPI::sum(new_dike_location[dim_i],this->get_mpi_communicator());
                             //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 25: dike = " << dike_i << ", dim_i = " << dim_i << std::endl;
                            }
                          dike_locations[dike_i].emplace_back(new_dike_location);

                          // if particle is not lost add a new point to the dike
                          //if (std::get<1>(particle_statuses[dike_i]) == 0)
                          //  dike_locations[dike_i].emplace_back(new_dike_point);
                         //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 30: dike = " << dike_i << std::endl;
                        }

                     //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 31: dike = " << dike_i << std::endl;
                    }
                 //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 32: dike = " << dike_i << std::endl;
                }
             //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 40" << std::endl;
              //for (size_t i = 0; i < dim; i++)
              //  {
              //    MPI_Bcast(&new_dike_point[i], 1, MPI_DOUBLE, cell_global_rank, this->get_mpi_communicator());
              //  }
              //std::cout << "new_dike_point after = " << new_dike_point << std::endl;
              //if (!particle_lost)
              //  dike_location.emplace_back(new_dike_point);
              //int results_rank_size = dike_location.size();
              //MPI_Bcast(&results_rank_size, 1, MPI_INT, cell_global_rank, this->get_mpi_communicator());
              //dike_location.resize(results_rank_size);
              //MPI_Bcast(&dike_location[0], results_rank_size, MPI_INT, cell_global_rank, this->get_mpi_communicator());
            }
        } /*else {
          // prevent deadlock.
          Point<dim> new_dike_point = Point<dim>();

         //std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 1" << std::endl;
 particle_handler->sort_particles_into_subdomains_and_cells();

         //std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 2" << std::endl;
  Utilities::MPI::sum(new_dike_point,this->get_mpi_communicator(),new_dike_point);
         //std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 3" << std::endl;
        }*/



      if (world_rank == 0)
        {
         //std::cout << "dike_location = ";
          for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
            {
             //std::cout << "dike " << dike_i << ": ";
              for (unsigned int segment_i = 0; segment_i < dike_locations[dike_i].size(); ++segment_i)
                {
                 //std::cout << dike_locations[dike_i][segment_i] << ", ";
                }
             //std::cout << std::endl;
            }
         //std::cout << std::endl;
        }

      // If using random dike generation
      /*if (false && enable_random_dike_generation)
        {
          // Dike is randomly generated in the potential dike generation
          // zone at each timestep.
          double x_dike_location = 0.0;
          double depth_change_random_dike = 0.0;

          // 1. generate a random number
          // We use a fixed number as seed for random generator
          // this is important if we run the code on more than 1 processor
          std::mt19937 random_number_generator (static_cast<unsigned int>((seed + 1) * this->get_timestep_number()));
          std::uniform_real_distribution<> dist(0, 1.0);

          // 2.1 Randomly generate the dike location (x_coordinate) by applying
          // quadratic transfer function, which is a parabolic relationship
          // between the random number and the dike x-coordinate.
          // i.e., rad_num =  (coefficent_a * (x_dike - (x_center_dike_generation_zone
          //                 - width_dike_generation_zone / 2)) ^2
          // coefficent_a = 1 / (width_dike_generation_zone/2)
          double x_dike_raw = 0.5 * width_dike_generation_zone
                              * std::sqrt(dist(random_number_generator))
                              + x_center_dike_generation_zone
                              - 0.5 * width_dike_generation_zone;

          // 2.2 Randomly generate the dike top depth change by appling the
          // same function.
          double depth_change_dike_raw = 0.5 * range_depth_change_random_dike
                                         * std::sqrt(dist(random_number_generator))
                                         + ref_top_depth_random_dike
                                         - 0.5 * range_depth_change_random_dike;

          // flip a coin and distribute dikes symmetrically around the center position of
          // dike generation zone (x_center_dike_generation_zone).
          std::uniform_real_distribution<> dist2(0,1.0);
          if (dist2(random_number_generator) < 0.5)
            {
              x_dike_location = x_dike_raw;
              depth_change_random_dike = depth_change_dike_raw;
            }
          else
            {
              x_dike_location = 2 * x_center_dike_generation_zone - x_dike_raw;
              depth_change_random_dike = 2 * ref_top_depth_random_dike - depth_change_dike_raw;
            }

          // 3. Find the x-direction side boundaries of the column where the dike is located.
          // TODO: Applies to all geometry models.
          AssertThrow(Plugins::plugin_type_matches<const GeometryModel::Box<dim>>(this->get_geometry_model()),
                      ExcMessage("Currently, this function only works with the box geometry model."));

          const GeometryModel::Box<dim> &
          geometry_model
            = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());

          // Get the maximum resolution in the x direction.
          const double dx_max = geometry_model.get_extents()[0]
                                / (geometry_model.get_repetitions()[0]
                                   * std::pow(2,total_refinement_levels));

          x_dike_left_boundary = std::floor(x_dike_location / dx_max) * dx_max;
          x_dike_right_boundary = x_dike_left_boundary + width_random_dike;
          top_depth_random_dike = ref_top_depth_random_dike + depth_change_random_dike;
        }*/
      particle_statuses.resize(0);
    }

    template <int dim>
    void
    DikeInjection<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      PrescribedPlasticDilation<dim>
      *prescribed_dilation = (this->get_parameters().enable_prescribed_dilation)
                             ? out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>()
                             : nullptr;
      ReactionRateOutputs<dim>
      *reaction_rate_out = (this->get_parameters().use_operator_splitting)
                           ? out.template get_additional_output<MaterialModel::ReactionRateOutputs<dim>>()
                           : nullptr;
      // Initiallize reaction_rates to 0.0.
      if (reaction_rate_out != nullptr)
        for (auto &row : reaction_rate_out->reaction_rates)
          std::fill(row.begin(), row.end(), 0.0);

      // When calculating other properties such as viscosity, we need to
      // correct for the effect of injection on the strain rate and thus
      // the deviatoric strain rate.
      //AssertThrow(in.current_cell.state() == IteratorState::valid, ExcMessage("error"));
      MaterialModel::MaterialModelInputs<dim> in_corrected_strainrate (in);
      if (this->get_timestep_number() > 0 && this->simulator_is_past_initialization())
        in_corrected_strainrate.requested_properties = in_corrected_strainrate.requested_properties | MaterialProperties::viscosity;
      //AssertThrow(in_corrected_strainrate.current_cell.state() == IteratorState::valid, ExcMessage("error"));
      //AssertThrow(in.current_cell.state() == IteratorState::valid, ExcMessage("error"));

      // Strore dike injection rate for each evaluation point
      std::vector<double> dike_injection_rate(in.n_evaluation_points());

      for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
        {
          if (enable_random_dike_generation)
            {
              // for now just add dike composition to this cell
              double min_distance = std::numeric_limits<double>::max();
              if (dike_locations.size() > 1)
                {
                  double distance;
                  const Point<dim> &P = in.position[q];
                  for (unsigned int dike_i = 0; dike_i < dike_locations.size(); dike_i++)
                    {
                      for (unsigned int point_index = 0; point_index < dike_locations[dike_i].size()-1; ++point_index)
                        {
                          const Point<dim> &X1 = dike_locations[dike_i][point_index];
                          const Point<dim> &X2 = dike_locations[dike_i][point_index+1];
                          if (dim == 3)
                            {
                              //https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                              distance = (cross_product_3d(P-X1,P-X2)).norm_square()/(X2-X1).norm_square();
                            }
                          else if (dim == 2)
                            {
                              // https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
                              //const Point<2> v = Point<2>(X2[1]-X1[1],-(X2[0]-X1[0]));
                              //const Point<2> r = Point<2>(X1[0]-X0[0],X1[1]-X0[1]);
                              //distance = std::fabs(v*r);

                              const Tensor<1,dim> v = (X2 - X1);
                              const Tensor<1,dim> w = (P - X1);

                              double c1 = w*v;
                              if ( c1 <= 0 )
                                {
                                  distance = (P-X1).norm();
                                }
                              else
                                {
                                  double c2 = v*v;
                                  if ( c2 <= c1 )
                                    {
                                      distance = (P-X2).norm();
                                    }
                                  else
                                    {
                                      double b = c1 / c2;
                                      Point Pb = X1 + b * v;
                                      distance = (P- Pb).norm();
                                    }
                                }

                              //if (in.position[q][0] > -2000. && in.position[q][0] < -1000. && in.position[q][1] > 40000 && in.position[q][1] < 41000)
                              //  {
                              //   //std::cout << point_index << "/" << dike_location.size() << ": P = " << P << ", X1 = " << X1 << ", X2 = " << X2 << ", distance = " << distance << ", min_distance = " << min_distance << std::endl;
                              //  }
                            }

                          if (distance < min_distance)
                            {
                              //if(in.position[i][0] > -1000. && in.position[i][0] < 1000. && in.position[i][1] > 49000 && in.position[i][1] < 51000)
                              // //std::cout << "distance = " << distance << ", min_distance = " << min_distance << std::endl;
                              min_distance = distance;
                            }

                        }
                    }
                }

              //if (min_distance < 10000 && this->get_timestep_number() > 0)
              //  {
              //    const double dike_injection_rate_double = 0;//1e-14;
              //    dike_injection_rate[q] = this->convert_output_to_years()
              //                             ? dike_injection_rate_double / year_in_seconds
              //                             : dike_injection_rate_double;
              //  }
              //else
              //  {
              //    dike_injection_rate[q] = 0;
              //  }
              // First find the location of the dike which is either randomly
              // generated or prescribed by the user.
              // Then give the dike_injection_rate to the dike points.
              if (false && enable_random_dike_generation)
                {
                  // Note: when the dike is generated randomly, the prescribed
                  // injection rate in the 'Dike injection function' should be
                  // only time dependent and independent of the xyz-coordinate.
                  //const double point_depth = this->get_geometry_model().depth(in.position[q]);
//
                  //// Find the randomly generated dike location
                  //if (in.position[q][0] >= x_dike_left_boundary
                  //    && in.position[q][0] <= x_dike_right_boundary
                  //    && in.temperature[q] <= T_bottom_dike
                  //    && point_depth >= std::max(top_depth_random_dike, 0.0)
                  //    && this->get_timestep_number() > 0)
                  //  dike_injection_rate[q] = this->convert_output_to_years()
                  //                           ? injection_function.value(in.position[q]) / year_in_seconds
                  //                           : injection_function.value(in.position[q]);
                  //else
                  //  dike_injection_rate[q] = 0.0;
//
                  // Dike injection effect removal
                  // Note that the correction starts from Timestep 1.
                  // Ensure the current cell is located within the dike area.
                  //if (dike_injection_rate[q] > 0.0
                  //    && this->get_timestep_number() > 0
                  //    && in.current_cell.state() == IteratorState::valid
                  //    && in.current_cell->center()[0] >= x_dike_left_boundary
                  //    && in.current_cell->center()[0] <= x_dike_right_boundary)
                  //  in_corrected_strainrate.strain_rate[q][0][0] -= dike_injection_rate[q];
                }
              else
                {
                  // User-defined dikes.
                  // The 'Dike injection function' is related to both the time and
                  // the xyz-coordinate.The bottom depth of the dike is limited by
                  // the isothermal depth of the brittle-ductle transition (BDT).
                  //if (in.temperature[q] <= T_bottom_dike
                  //    && this->get_timestep_number() > 0)
                  //  dike_injection_rate[q] = this->convert_output_to_years()
                  //                           ? injection_function.value(in.position[q]) / year_in_seconds
                  //                           : injection_function.value(in.position[q]);
                  //else
                  //  dike_injection_rate[q] = 0.0;
//
                  //// Dike injection effect removal
                  //if (dike_injection_rate[q] > 0.0
                  //    && this->get_timestep_number() > 0
                  //    && in.current_cell.state() == IteratorState::valid
                  //    && injection_function.value(in.current_cell->center()) > 0.0)
                  //  in_corrected_strainrate.strain_rate[q][0][0] -= dike_injection_rate[q];
                }
            }
        }

      // Fill variable out with the results form the base material model
      // using the corrected model inputs.
      base_model->evaluate(in_corrected_strainrate, out);


      // Below we start to track the motion of the dike injection material.
      AssertThrow(this->introspection().compositional_name_exists("injection_phase"),
                  ExcMessage("Material model 'dike injection' only works if "
                             "there is a compositional field called 'injection_phase'. "));

      // Index for injection phase
      unsigned int injection_phase_index = this->introspection().compositional_index_for_name("injection_phase");
      unsigned int injection_phase_current_index = this->introspection().compositional_index_for_name("injection_phase_current");
      clear_composition_field_index = injection_phase_current_index;
      // Indices for all chemical compositional fields, and not e.g., plastic strain.
      const std::vector<unsigned int> chemical_composition_indices = this->introspection().get_indices_for_fields_of_type(CompositionalFieldDescription::porosity);    //chemical_composition_field_indices();

      const auto &component_indices = this->introspection().component_indices.compositional_fields;

      // Positions of quadrature points at the current cell
      std::vector<Point<dim>> quadrature_positions;

      // The injection material will replace part of the original material
      // based on the injection rate and diking duration, i.e., fraction
      // of injected material to original material for the existence
      // duration of a dike.
      double dike_injection_fraction = 0.0;

      for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
        {
          if (enable_random_dike_generation)
            {
              // for now just add dike composition to this cell
              double min_distance = std::numeric_limits<double>::max();

              double distance;
              const Point<dim> &P = in.position[q];
              for (unsigned int dike_i = 0; dike_i < dike_locations.size(); ++dike_i)
                {
                  if (dike_locations[dike_i].size() > 1)
                    {
                      for (unsigned int point_index = 0; point_index < dike_locations[dike_i].size()-1; ++point_index)
                        {
                          const Point<dim> &X1 = dike_locations[dike_i][point_index];
                          const Point<dim> &X2 = dike_locations[dike_i][point_index+1];
                          if (dim == 3)
                            {
                              //https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
                              distance = (cross_product_3d(P-X1,P-X2)).norm_square()/(X2-X1).norm_square();
                            }
                          else if (dim == 2)
                            {
                              // https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
                              //const Point<2> v = Point<2>(X2[1]-X1[1],-(X2[0]-X1[0]));
                              //const Point<2> r = Point<2>(X1[0]-X0[0],X1[1]-X0[1]);
                              //distance = std::fabs(v*r);

                              const Tensor<1,dim> v = (X2 - X1);
                              const Tensor<1,dim> w = (P - X1);

                              double c1 = w*v;
                              if ( c1 <= 0 )
                                {
                                  distance = (P-X1).norm();
                                }
                              else
                                {

                                  double c2 = v*v;
                                  if ( c2 <= c1 )
                                    {
                                      distance = (P-X2).norm();
                                    }
                                  else
                                    {
                                      double b = c1 / c2;
                                      Point Pb = X1 + b * v;
                                      distance = (P- Pb).norm();
                                    }
                                }

                              //if (in.position[q][0] > -2000. && in.position[q][0] < -1000. && in.position[q][1] > 40000 && in.position[q][1] < 41000)
                              //  {
                              //   //std::cout << point_index << "/" << dike_location.size() << ": P = " << P << ", X1 = " << X1 << ", X2 = " << X2 << ", distance = " << distance << ", min_distance = " << min_distance << std::endl;
                              //  }
                            }

                          if (distance < min_distance)
                            {
                              //if (in.position[q][0] > -2000. && in.position[q][0] < -1000. && in.position[q][1] > 40000 && in.position[q][1] < 41000)
                              // //std::cout << "distance = " << distance << ", min_distance = " << min_distance << std::endl;
                              min_distance = distance;
                            }
                        }
                    }
                }
              if (min_distance < 1000 && this->get_timestep_number() > 0)
                {
                  out.viscosities[q] *= dike_visosity_multiply_factor;
                  const double dike_injection_rate_double = 1;//1e-134;
                  dike_injection_rate[q] = this->convert_output_to_years()
                                           ? dike_injection_rate_double / year_in_seconds
                                           : dike_injection_rate_double;

                  /*
                            // User-defined or timestep-dependent injection fraction.
                            if (this->simulator_is_past_initialization())
                              dike_injection_fraction = dike_injection_rate[q] * this->get_timestep();

                  if (dike_injection_rate[q] > 0.0
                                && this->get_timestep_number() > 0
                                && in.current_cell.state() == IteratorState::valid)
                              {


                            const UpdateFlags update_flags = update_values;// | update_gradients;//property_manager->get_needed_update_flags();

                            std::unique_ptr<SolutionEvaluator<dim>> evaluator = construct_solution_evaluator(*this,
                                                                                 update_flags);

                                      small_vector<double> solution_values(this->get_fe().dofs_per_cell);

                                      in.current_cell->get_dof_values(this->get_old_solution(),
                                                           solution_values.begin(),
                                                           solution_values.end());



                                      //fe_values.reinit(cell);
                                      //evaluator->reinit(in.current_cell, positions, {solution_values.data(), solution_values.size()}, update_flags);


                                // If the "single Advection" nonlinear solver scheme is used,
                                // it is necessary to set the reaction term to 0 to avoid
                                // additional plastic deformation generated by dike injection
                                // within the dike zone.
                                if (this->get_parameters().nonlinear_solver ==
                                    Parameters<dim>::NonlinearSolver::single_Advection_single_Stokes
                                    ||
                                    this->get_parameters().nonlinear_solver ==
                                    Parameters<dim>::NonlinearSolver::single_Advection_iterated_Stokes
                                    ||
                                    this->get_parameters().nonlinear_solver ==
                                    Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes
                                    ||
                                    this->get_parameters().nonlinear_solver ==
                                    Parameters<dim>::NonlinearSolver::single_Advection_iterated_defect_correction_Stokes)
                                  {
                                    if (this->introspection().compositional_name_exists("plastic_strain"))
                                      out.reaction_terms[q][this->introspection().compositional_index_for_name("plastic_strain")] = 0.0;
                                    if (this->introspection().compositional_name_exists("viscous_strain"))
                                      out.reaction_terms[q][this->introspection().compositional_index_for_name("viscous_strain")] = 0.0;
                                    if (this->introspection().compositional_name_exists("total_strain"))
                                      out.reaction_terms[q][this->introspection().compositional_index_for_name("total_strain")] = 0.0;
                                    if (this->introspection().compositional_name_exists("noninitial_plastic_strain"))
                                      out.reaction_terms[q][this->introspection().compositional_index_for_name("noninitial_plastic_strain")] = 0.0;
                                  }
                              }*/

                }
              else
                {
                  dike_injection_rate[q] = 0;
                }
            }
        }

      for (unsigned int q=0; q < in.n_evaluation_points(); ++q)
        {
          // Activate the dike injection by adding the additional RHS
          // terms of injection to Stokes equations.
          if (prescribed_dilation != nullptr)
            prescribed_dilation->dilation[q] = dike_injection_rate[q];

          // User-defined or timestep-dependent injection fraction.
          if (this->simulator_is_past_initialization())
            dike_injection_fraction = dike_injection_rate[q];// * this->get_timestep();

          //if (dike_material_injection_fraction != 0.0)
          //  dike_injection_fraction = dike_material_injection_fraction;

          //            const double diff = 100.;
          //if(in.position[q][0] > -1250.-diff && in.position[q][0] < -1250.+diff && in.position[q][1] > 90087.4 -diff && in.position[q][1] < 90087.4 +diff)
          // //std::cout << q << ": position: " << in.position[q] << ", dike_injection_rate[q] = " << dike_injection_rate[q] << ", dike_injection_fraction = " << dike_injection_fraction << ", dike_material_injection_fraction = " << dike_material_injection_fraction << std::endl;

          if (//dike_injection_rate[q] > 0.0
            //&&
            this->get_timestep_number() > 0
            && in.current_cell.state() == IteratorState::valid)
            {
              // We need to obtain the values of chemical compositional fields
              // at the previous time step, as the values from the current
              // linearization point are an extrapolation of the solution from
              // the old timesteps. Prepare the field function and extract the
              // old solution values at the current cell.
              quadrature_positions.resize(1,this->get_mapping().transform_real_to_unit_cell(in.current_cell, in.position[q]));

              // Use a boost::small_vector to avoid memory allocation if possible.
              // Create 100 values by default, which should be enough for most cases.
              // If there are more than 100 DoFs per cell, this will work like a normal vector.
              boost::container::small_vector<double, 100> old_solution_values(this->get_fe().dofs_per_cell);
              in.current_cell->get_dof_values(this->get_old_solution(),
                                              old_solution_values.begin(),
                                              old_solution_values.end());

              // If we have not been here before, create one evaluator for each compositional field
              if (composition_evaluators.size() == 0)
                composition_evaluators.resize(this->n_compositional_fields());

              // Make sure the evaluators have been initialized correctly, and have not been tampered with
              Assert(composition_evaluators.size() == this->n_compositional_fields(),
                     ExcMessage("The number of composition evaluators should be equal to the number of compositional fields."));

              // Loop only in chemical copositional fields
              for (unsigned int c : chemical_composition_indices)
                {
                  if ((c == injection_phase_index && dike_injection_rate[q] > 0.0) || (c == injection_phase_current_index && dike_injection_rate[q] > 0.0 ))
                    {
                      // Only create the evaluator the first time we get here
                      if (!composition_evaluators[c])
                        composition_evaluators[c]
                          = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                         this->get_fe(),
                                                                         update_values,
                                                                         component_indices[c]);

                      composition_evaluators[c]->reinit(in.current_cell, quadrature_positions);
                      composition_evaluators[c]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                          EvaluationFlags::values);
                      const double old_solution_composition = composition_evaluators[c]->get_value(0);

                      // If the value increases to greater than 1, it is not increased anymore.
                      //if (old_solution_composition + dike_injection_fraction >= 1.0)
                      //  out.reaction_terms[q][c] = 0.0;
                      //else
                      out.reaction_terms[q][c] = std::max(dike_injection_fraction, -old_solution_composition);

                      // Fill reaction rate outputs instead of the reaction terms if
                      // we use operator splitting (and then set the latter to zero).
                      if (reaction_rate_out != nullptr)
                        reaction_rate_out->reaction_rates[q][c] = out.reaction_terms[q][c]
                                                                  / this->get_timestep();

                      if (this->get_parameters().use_operator_splitting)
                        out.reaction_terms[q][c] = 0.0;
                    }
                  //else /*if(c == injection_phase_current_index)
                  /*{
                      // Only create the evaluator the first time we get here
                      if (!composition_evaluators[c])
                        composition_evaluators[c]
                          = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                         this->get_fe(),
                                                                         update_values,
                                                                         component_indices[c]);

                      composition_evaluators[c]->reinit(in.current_cell, quadrature_positions);
                      composition_evaluators[c]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                          EvaluationFlags::values);
                      const double old_solution_composition = composition_evaluators[c]->get_value(0);

                      // If the value increases to greater than 1, it is not increased anymore.
                      //if (old_solution_composition + dike_injection_fraction >= 1.0)
                      //  out.reaction_terms[q][c] = 0.0;
                      //else
                      //const double diff = 100.;
                      //if(in.position[q][0] > -1250.-diff && in.position[q][0] < -1250.+diff){
                      // //std::cout << "position: " << in.position[q] << std::endl;
                      //}
                      if(in.position[q][0] > -1250.-diff && in.position[q][0] < -1250.+diff && in.position[q][1] > 90087.4 -diff && in.position[q][1] < 90087.4 +diff)
                       //std::cout << q << ": A position: " << in.position[q] << ", dike_injection_fraction = " << dike_injection_fraction << ", old_solution_composition = " << old_solution_composition << ", result = " << std::max(dike_injection_fraction-old_solution_composition, -old_solution_composition) << std::endl;
                      if(dike_injection_rate[q] > 0.0 || std::fabs(dike_injection_fraction-old_solution_composition) > 0.0){if(in.position[q][0] > -1250.-diff && in.position[q][0] < -1250.+diff && in.position[q][1] > 90087.4 -diff && in.position[q][1] < 90087.4 +diff)
                       //std::cout << q << ": B position: " << in.position[q] << ", dike_injection_fraction = " << dike_injection_fraction << ", old_solution_composition = " << old_solution_composition << ", result = " << std::max(dike_injection_fraction-old_solution_composition, -old_solution_composition) << std::endl;

                      out.reaction_terms[q][c] = std::max(dike_injection_fraction-old_solution_composition, -old_solution_composition);

                      // Fill reaction rate outputs instead of the reaction terms if
                      // we use operator splitting (and then set the latter to zero).
                      if (reaction_rate_out != nullptr)
                        reaction_rate_out->reaction_rates[q][c] = out.reaction_terms[q][c]
                                                                  / this->get_timestep();

                      if (this->get_parameters().use_operator_splitting)
                        out.reaction_terms[q][c] = 0.0;
                        }

                  } else*/
                  /*{
                    // Only create the evaluator the first time we get here
                    if (!composition_evaluators[c])
                      composition_evaluators[c]
                        = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                       this->get_fe(),
                                                                       update_values,
                                                                       component_indices[c]);

                    composition_evaluators[c]->reinit(in.current_cell, quadrature_positions);
                    composition_evaluators[c]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                        EvaluationFlags::values);
                    const double old_solution_other_composition = composition_evaluators[c]->get_value(0);

                    // When new dike material is injected, the other compositional fields
                    // at the dike point will be reduced in the same proportion (p_c) to
                    // ensure that the sum of all compositional fields is always 1.0.
                    // For example, in the previous step, the dike material has a compostional
                    // field with the value of c_dike_old and another compositional field
                    // with a value of c_1_old. So, c_dike_old + c_1_old = 1.0. Here we
                    // leave the background field alone, because it will be automatically
                    // populated if c_dike_old + c_1_old < 1.0.
                    // In the currest step, when adding a new dike material of amount 'c_dike_add',
                    // c_dike_new =  c_dike_old + c_dike_add. c_1_new = c_1_old * p_c.
                    // Since c_1_new + c_dike_new = 1.0, we get
                    // p_c = (1.0 - c_dike_old - c_dike_add) / c_1_old.
                    // Then the amount of change in c_1 is:
                    // delta_c_1 = c_1_new - c_1_old = c_1_old * (p_c - 1.0)
                    // = - c_1_old * (c_dike_add / (1.0001 - c_dike_old))
                    // To avoid dividing by 0, we will use 1.0001 instead of 1.0.

                    // We limit the value of injection phase compostional
                    // field at previous timestep is [0,1].

                    if (!composition_evaluators[injection_phase_index])
                      composition_evaluators[injection_phase_index]
                        = std::make_unique<FEPointEvaluation<1, dim>>(this->get_mapping(),
                                                                       this->get_fe(),
                                                                       update_values,
                                                                       component_indices[c]);

                    composition_evaluators[injection_phase_index]->reinit(in.current_cell, quadrature_positions);
                    composition_evaluators[injection_phase_index]->evaluate({old_solution_values.data(),old_solution_values.size()},
                                                                            EvaluationFlags::values);
                    double injection_phase_composition = std::max(std::min(composition_evaluators[injection_phase_index]->get_value(0),1.0),0.0);

                    //out.reaction_terms[q][c] = -old_solution_other_composition
                    //                           * std::min(dike_injection_fraction
                    //                                      / (1.0001 - injection_phase_composition), 1.0);

                    // Fill reaction rate outputs instead of the reaction terms if
                    // we use operator splitting (and then set the latter to zero).
                    if (reaction_rate_out != nullptr)
                      reaction_rate_out->reaction_rates[q][c] = out.reaction_terms[q][c]
                                                                / this->get_timestep();

                    if (this->get_parameters().use_operator_splitting)
                      out.reaction_terms[q][c] = 0.0;
                  }*/
                }
              //const double diff = 100.;
              //if(in.position[q][0] > -1250.-diff && in.position[q][0] < -1250.+diff && in.position[q][1] > 96301.8-diff && in.position[q][1] < 96301.8+diff){
              // //std::cout << "position: " << in.position[q] << ", noninitial_plastic_strain = " << out.reaction_terms[0][this->introspection().compositional_index_for_name("noninitial_plastic_strain")] << ", sr = " << std::sqrt(std::max(-second_invariant(deviator(in.strain_rate[0])), 0.))<< std::endl;
              //}
              // If the "single Advection" nonlinear solver scheme is used,
              // it is necessary to set the reaction term to 0 to avoid
              // additional plastic deformation generated by dike injection
              // within the dike zone.
              /*if (this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_single_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes
                  ||
                  this->get_parameters().nonlinear_solver ==
                  Parameters<dim>::NonlinearSolver::single_Advection_iterated_defect_correction_Stokes)
                {
                  if (this->introspection().compositional_name_exists("plastic_strain"))
                    out.reaction_terms[q][this->introspection().compositional_index_for_name("plastic_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("viscous_strain"))
                    out.reaction_terms[q][this->introspection().compositional_index_for_name("viscous_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("total_strain"))
                    out.reaction_terms[q][this->introspection().compositional_index_for_name("total_strain")] = 0.0;
                  if (this->introspection().compositional_name_exists("noninitial_plastic_strain"))
                    out.reaction_terms[q][this->introspection().compositional_index_for_name("noninitial_plastic_strain")] = 0.0;
                }*/
            }

        }
    }

    template <int dim>
    void
    DikeInjection<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dike injection");
        {
          prm.declare_entry("Base model","simple",
                            Patterns::Selection(MaterialModel::get_valid_model_names_pattern<dim>()),
                            "The name of a material model that will be modified by an "
                            "averaging operation. Valid values for this parameter "
                            "are the names of models that are also valid for the "
                            "``Material models/Model name'' parameter. See the documentation for "
                            "that for more information.");
          prm.declare_entry("Dike material injection fraction", "0.0", Patterns::Double(0),
                            "Amount of new injected material from the dike. Units: none.");
          prm.declare_entry("Dike bottom temperature", "873.0", Patterns::Double(0),
                            "Temperature at the bottom of the generated dike. It usually equals to "
                            "the temperature at the intersection of the brittle-ductile transition "
                            "zone and the dike. Units: K.");
          prm.declare_entry("X center of the dike generation zone", "0.0", Patterns::Double(0),
                            "X_coordinate of the center of the dike generation zone. Units: m.");
          prm.declare_entry("Width of the dike generation zone", "0.0", Patterns::Double(0),
                            "Width of the dike generation zone. Units: m.");
          prm.declare_entry("Total refinement levels", "0", Patterns::Double(0),
                            "The total refinment levels in the model, which equals to the sum "
                            "of global refinement levels and adpative refinement levels. This "
                            "is used for calcuting the dike location. Units: none.");
          prm.declare_entry("Random number generator seed", "0", Patterns::Double(0),
                            "The value of the seed used for the random number generator. Units: none.");
          prm.declare_entry("Reference top depth of randomly generated dike", "0.0", Patterns::Double(0),
                            "Randomly generated depth change of the dike corresponding to the dike "
                            "reference top depth. Units: m.");
          prm.declare_entry("Range of randomly generated dike depth change", "0.0", Patterns::Double(0),
                            "Full range of randomly generated dike top depth change. Units: m.");
          prm.declare_entry("Width of randomly generated dike", "0.0", Patterns::Double(0),
                            "Width of the generated dike. Units: m.");
          prm.declare_entry("Enable random dike generation", "false", Patterns::Bool (),
                            "Whether the dikes are generated randomly. If the dike is generated randomly, "
                            "the prescribed injection rate in the 'Dike injection function' should be "
                            "only time dependent and independent of the xyz-coordinate.");
          prm.declare_entry("Dike viscosity multiply factor", "0.1", Patterns::Double(0),
                            "");
          prm.enter_subsection("Dike injection function");
          {
            Functions::ParsedFunction<dim>::declare_parameters(prm,1);
            prm.declare_entry("Function expression","0.0");
          }
          prm.leave_subsection();
          aspect::Particle::Integrator::Interface<dim>::declare_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    DikeInjection<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Dike injection");
        {
          AssertThrow( prm.get("Base model") != "Dike injection",
                       ExcMessage("You may not use ''dike injection'' as the base model for itself."));

          // create the base model and initialize its SimulatorAccess base
          // class; it will get a chance to read its parameters below after we
          // leave the current section
          base_model = create_material_model<dim>(prm.get("Base model"));
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(base_model.get()))
            sim->initialize_simulator (this->get_simulator());

          dike_material_injection_fraction = prm.get_double ("Dike material injection fraction");
          T_bottom_dike = prm.get_double ("Dike bottom temperature");
          enable_random_dike_generation = prm.get_bool("Enable random dike generation");
          x_center_dike_generation_zone = prm.get_double ("X center of the dike generation zone");
          width_dike_generation_zone = prm.get_double ("Width of the dike generation zone");
          total_refinement_levels = prm.get_double ("Total refinement levels");
          seed = prm.get_double ("Random number generator seed");
          ref_top_depth_random_dike = prm.get_double ("Reference top depth of randomly generated dike");
          range_depth_change_random_dike = prm.get_double ("Range of randomly generated dike depth change");
          width_random_dike = prm.get_double ("Width of randomly generated dike");
          dike_visosity_multiply_factor = prm.get_double("Dike viscosity multiply factor");

          //prm.enter_subsection("Dike injection function");
          //{
          //  try
          //    {
          //      injection_function.parse_parameters(prm);
          //    }
          //  catch (...)
          //    {
          //      std::cerr << "ERROR: FunctionParser failed to parse\n"
          //                << "\t Dike injection function\n"
          //                << "with expression \n"
          //                << "\t' " << prm.get("Function expression") << "'";
          //      throw;
          //    }
          //}
          //prm.leave_subsection();

          particle_integrator = aspect::Particle::Integrator::create_particle_integrator<dim>(prm);
          if (SimulatorAccess<dim> *sim = dynamic_cast<SimulatorAccess<dim>*>(particle_integrator.get()))
            sim->initialize_simulator (this->get_simulator());
          particle_integrator->parse_parameters(prm);
          dynamic_cast<Particle::Integrator::RK4<dim>*>(particle_integrator.get())->set(0);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // If 'Free Surface' is used, please ensure the 'Surface velocity
      // projection' is vertical. If the projection is normal, which means
      // the surface mesh can deform both horizontally and vertically, this
      // may distort a surface element and give a non-positive volume fraction
      // in a quadrature point which is invalid.
      prm.enter_subsection("Mesh deformation");
      {
        // Check if "free surface" is specified in Mesh deformation boundary indicators
        std::string boundary_indicators = prm.get("Mesh deformation boundary indicators");
        std::string advection_direction = "nan";
        if (boundary_indicators.find("free surface") != std::string::npos)
          {
            prm.enter_subsection("Free surface");
            {
              advection_direction = prm.get("Surface velocity projection");
            }
            prm.leave_subsection();
            AssertThrow(advection_direction == "vertical",
                        ExcMessage("The projection is " + advection_direction +
                                   ". However, this function currently prefers to use "
                                   "vertical projection if using free surface."));
          }
      }
      prm.leave_subsection();

      // After parsing the parameters for averaging, it is essential
      // to parse parameters related to the base model.
      base_model->parse_parameters(prm);
      this->model_dependence = base_model->get_model_dependence();
    }

    template <int dim>
    bool
    DikeInjection<dim>::
    is_compressible () const
    {
      return base_model->is_compressible();
    }

    template <int dim>
    void
    DikeInjection<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // The base model may have additional outputs, so we need to copy
      // these additional outputs.
      base_model->create_additional_named_outputs(out);

      //Stokes additional RHS for prescribed dilation
      const unsigned int n_points = out.n_evaluation_points();
      if (this->get_parameters().enable_prescribed_dilation
          && out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedPlasticDilation<dim>> (n_points));
        }

      AssertThrow(!this->get_parameters().enable_prescribed_dilation
                  ||
                  out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>()->dilation.size()
                  == n_points, ExcInternalError());

      if (this->get_parameters().use_operator_splitting
          && out.template get_additional_output<MaterialModel::ReactionRateOutputs<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::ReactionRateOutputs<dim>> (n_points,
                                                                        this->n_compositional_fields()));
        }
    }

  }

}

//   template <int dim>
// void signal_connector (aspect::SimulatorSignals<dim> &signals)
// {
//   signals.start_timestep.connect(&aspect::clear_compositional_field<dim>);
//  //std::cout << "Connecting signal" << std::endl;
// }
//ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>, signal_connector<3>)

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(DikeInjection,
                                   "dike injection",
                                   "The material model uses a ``Base model'' from which "
                                   "material properties are derived. It then adds source "
                                   "terms in the Stokes equations that describe a dike "
                                   "injection of material to the model. ")
  }
}
