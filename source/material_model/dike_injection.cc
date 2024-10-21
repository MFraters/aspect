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


#include <deal.II/grid/grid_tools.h>
namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    void
    DikeInjection<dim>::initialize()
    {
      base_model->initialize();

    }

    template <int dim>
    void
    DikeInjection<dim>::set_particle_lost(const typename Particles::ParticleIterator<dim> &particle, const typename Triangulation<dim>::active_cell_iterator &/*cell*/)
    {
      std::cout << "lost particle here!!!!: " << particle->get_location() << std::endl;
      particle_lost = true;
      particle_lost_location = particle->get_location();
    }

    template <int dim>
    std::vector<Tensor<1,dim>>
    DikeInjection<dim>::compute_stress_largest_eigenvector(std::unique_ptr<SolutionEvaluator<dim>> &evaluator,
                                                           typename DoFHandler<dim>::active_cell_iterator &cell,
                                                           std::vector<Point<dim>> &positions,
                                                           small_vector<double> & /*solution_values*/)
    {



      //const UpdateFlags update_flags = update_values | update_gradients;
      //evaluator->reinit(cell, positions, {solution_values.data(), solution_values.size()}, update_flags);
      //std::cout << "ifcsle flag 1: positions.size() = " << positions.size() << ", this->introspection().n_components = " << this->introspection().n_components << std::endl;
      Assert(cell.state() == IteratorState::valid,ExcMessage("Cell state is not valid."));

      std::vector<Vector<double>> solution;
      solution.resize(1,Vector<double>(this->introspection().n_components));

      std::vector<std::vector<Tensor<1,dim>>> gradients;
      gradients.resize(1,std::vector<Tensor<1,dim>>(this->introspection().n_components));

      //std::cout << "ifcsle flag 5" << std::endl;
      //for (unsigned int i = 0; i<1; ++i)
      {
        // Evaluate the solution, but only if it is requested in the update_flags
        //if (update_flags & update_values)
        evaluator->get_solution(0, {&solution[0][0],solution[0].size()});

        //std::cout << "ifcsle flag 6" << std::endl;
        // Evaluate the gradients, but only if they are requested in the update_flags
        //if (update_flags & update_gradients)
        evaluator->get_gradients(0, gradients[0]);
      }

      //std::cout << "ifcsle flag 7" << std::endl;
      // get presure, temp, etc

      // need access to the pressure, viscosity,
      // get velocity

      Tensor<1,dim> velocity;

      for (unsigned int i = 0; i < dim; ++i)
        velocity[i] = solution[0][this->introspection().component_indices.velocities[i]];

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
      const double temperature = solution[0][this->introspection().component_indices.temperature];

      //std::cout << "fevalues temp = " << temperature_values[0] << ", old: " << solution[0][this->introspection().component_indices.temperature] << std::endl;

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
        }

      //std::cout << "ifcsle flag end" << std::endl;
      return {stress_largest_eigenvectors};

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
      particle_lost = false;
      dike_location.resize(0);
      if (dim == 2)
        {
          dike_location.emplace_back(Point<dim>(-1370.4997314869,40489.36393586183));//(0,50225));
        }
      else
        {
          dike_location.emplace_back(Point<dim>(-1370.4997314869,50e3,40489.36393586183));//(0,50225));
        }
      // If we found the correct cell on this MPI process, we have found the right cell.
      //Assert(cell_it.first.state() == IteratorState::valid && cell_it.first->is_locally_owned(), ExcMessage("Internal error: could not find cell to place initial point."));

      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      int world_size;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      //std::cout << "world_rank = " << world_rank << "/" << world_size << ": Flag 0: enable_random_dike_generation = " << enable_random_dike_generation
      //<< ", this->get_timestep_number() = " << this->get_timestep_number() << ", cell_it.first.state() = " << cell_it.first.state()
      ////<< ", cell_it.first->is_locally_owned() =" << cell_it.first->is_locally_owned()
      //<< std::endl;

      if (enable_random_dike_generation && this->get_timestep_number() > 0)// && cell_it.first.state() == IteratorState::valid)// && cell_it.first->is_locally_owned())
        {


          std::pair<const typename parallel::distributed::Triangulation<dim>::active_cell_iterator,Point<dim>> cell_it_start = GridTools::find_active_cell_around_point<>(this->get_mapping(), this->get_triangulation(), dike_location.back());


          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 1" << std::endl;

          if (cell_it_start.first.state() == IteratorState::valid && cell_it_start.first->is_locally_owned())
            {
              unsigned int next_free_id = particle_handler->get_next_free_particle_index();
              particle_handler->insert_particle(dike_location.back(),cell_it_start.second,next_free_id, cell_it_start.first);
            }
          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 1.5" << std::endl;
          //particle_handler->update_cached_numbers();
          particle_handler->sort_particles_into_subdomains_and_cells();
          // get the stress at the point
          // get the solutions and gradients
          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 2" << std::endl;

          const UpdateFlags update_flags = update_values | update_gradients;//property_manager->get_needed_update_flags();

          std::unique_ptr<SolutionEvaluator<dim>> evaluator = construct_solution_evaluator(*this,
                                                               update_flags);


          //const Quadrature<dim> quadrature_formula (std::vector<Point<dim>>(1,particle_handler->begin()->get_reference_location()));

          //const unsigned int n_q_points =  quadrature_formula.size();
          //FEValues<dim> fe_values (this->get_mapping(), this->get_fe(),  quadrature_formula,
          //                         update_flags);

          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3" << std::endl;
          // loop untill point is no longer in any cell;
          // todo: or max number?
          int iteration = 0;
          while (!this->particle_lost)
            {
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.1" << std::endl;
              iteration++;
              if (!(iteration < 500))
                {
                  std::string concat = "";
                  std::cout << "Failing at iteration " << iteration << ", current dike path: " << std::endl;
                  for (auto coords : dike_location)
                    {
                      //concat += std::to_string(coords);
                      std::cout << coords << ", ";
                    }
                  AssertThrow(iteration < 500, ExcMessage ("too many iterations for the dike to reach the surface. rank: " + std::to_string(world_rank)));
                }
              std::vector<Point<dim>> positions = {dim == 3 ? Point<dim>(0,0,0) : Point<dim>(0,0)};

              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.2, cell_it.first.state() = " << cell_it.first.state() << ", IteratorState::valid = " << IteratorState::valid << std::endl;
              if (particle_handler->n_locally_owned_particles() > 0) //        cell_it.first.state() == IteratorState::valid)
                {
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.3, cell_it.first.state() = " << cell_it.first.state() << ", IteratorState::valid = " << IteratorState::valid << ", particle_handler->begin() = " << particle_handler->begin()->get_surrounding_cell().state() << std::endl;
                  positions[0] = particle_handler->begin()->get_reference_location();
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 3.4" << std::endl;
                } //? {{particle_handler->begin()->get_reference_location()}} : {};

              //std::cout << " old position: " << particle_handler->begin()->get_location() << std::endl;


              Point<dim> new_dike_point = Point<dim>();
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 4" << std::endl;
              size_t iter2 = 0;
              do
                {
                  iter2++;
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 4.5" << std::endl;
                  //std::cout << iteration << ":" << iter2 << "(1): particle lost = " << particle_lost << std::endl;
                  particle_handler->sort_particles_into_subdomains_and_cells();
                  //std::cout << iteration << ":" << iter2 << "(2): parwhileticle lost = " << particle_lost << std::endl;
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 5" << std::endl;
                  unsigned int particle_lost_int = (unsigned int)particle_lost;
                  //std::cout << iteration << ":" << iter2 << "(3): parwhileticle lost = " << particle_lost_int << std::endl;
                  particle_lost = Utilities::MPI::sum((unsigned int)particle_lost,this->get_mpi_communicator());
                  //std::cout << iteration << ":" << iter2 << "(4): parwhileticle lost = " << particle_lost_int << std::endl;
                  if (particle_lost)
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

                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 8, positions.size() = " << positions.size() << std::endl;
                  if (particle_handler->n_locally_owned_particles() > 0 && particle_handler->begin()->get_surrounding_cell().state() == IteratorState::valid)
                    {
                      typename DoFHandler<dim>::active_cell_iterator cell = typename DoFHandler<dim>::active_cell_iterator(*particle_handler->begin()->get_surrounding_cell(),&(this->get_dof_handler()));

                      //std::cout << iteration << ": ifworld_rank = " << world_rank << "/" << world_size << ": Flag 9, positions.size() = " << positions.size() 
                      //<< ", cell_it.first.state() = " << cell->state() << ":" << IteratorState::valid << std::endl;

                      Assert(positions.size() == 1, ExcMessage("Internal error."));
                      positions[0] = particle_handler->begin()->get_reference_location();
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
                          evaluator->reinit(cell, positions, {solution_values.data(), solution_values.size()}, update_flags);


                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 10" << std::endl;
                          // function here
                          //Tensor<1,dim> solution_stress =
                          std::vector<Tensor<1,dim>> solution_stress = compute_stress_largest_eigenvector(evaluator,cell,positions,solution_values);;

                          //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 11" << std::endl;
                          cell->get_dof_values(this->get_current_linearization_point(),
                                               solution_values.begin(),
                                               solution_values.end());

                          evaluator->reinit(cell, positions, {solution_values.data(), solution_values.size()}, update_flags);

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
                    }
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 12.5" << std::endl;
                }
              while (particle_integrator->new_integration_step());

              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 13: particle_lost = " << particle_lost << ", cell_it.first.state() = " << cell_it.first.state() << std::endl;
              if (particle_handler->n_locally_owned_particles() > 0) //cell_it.first.state() == IteratorState::valid) {
                {
                  //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 13.5: particle_lost = " << particle_lost << std::endl;
                  new_dike_point = particle_lost ? particle_lost_location : particle_handler->begin()->get_location();
                }
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 14" << std::endl;

              //std::cout << "new_dike_point = " << new_dike_point << ", dike_location.back() = " << dike_location.back() << std::endl;
              //          << ", stress_largest_eigenvectors = " << stress_largest_eigenvectors << ", distance = " << distance << std::endl;
              // TODO: check if still is in domain, otherwise end loop

              //}
              //else
              //  {
              //    //// receive new dike injection vector when the owning processor is done.
              //    //MPI_Status status;
              //    //int results_rank_size;
              //    //MPI_Recv(&results_rank_size, 1, MPI_INT, MPI_ANY_SOURCE, 666, this->get_mpi_communicator(), &status);
              //    //dike_location.resize(results_rank_size);
              //    //MPI_Recv(&dike_location[0], results_rank_size, MPI_INT, MPI_ANY_SOURCE, 777, this->get_mpi_communicator(), &status);
              //  }

              //std::cout << "new_dike_point before = " << new_dike_point << std::endl;
              // Synchronize the appended dike points vector over all mpi processes
              // broadcast dim doubles into a point

              //new_dike_array[dim];
              //for(unsigned int i = 0; i < dim; ++i){
              //  new_dike_array[i] = new_dike_point[i];
              //}

              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 30" << std::endl;
              Utilities::MPI::sum(new_dike_point,this->get_mpi_communicator(),new_dike_point);
              //std::cout << "ifworld_rank = " << world_rank << "/" << world_size << ": Flag 40" << std::endl;
              //for (size_t i = 0; i < dim; i++)
              //  {
              //    MPI_Bcast(&new_dike_point[i], 1, MPI_DOUBLE, cell_global_rank, this->get_mpi_communicator());
              //  }
              //std::cout << "new_dike_point after = " << new_dike_point << std::endl;
              if (!particle_lost)
                dike_location.emplace_back(new_dike_point);
              //int results_rank_size = dike_location.size();
              //MPI_Bcast(&results_rank_size, 1, MPI_INT, cell_global_rank, this->get_mpi_communicator());
              //dike_location.resize(results_rank_size);
              //MPI_Bcast(&dike_location[0], results_rank_size, MPI_INT, cell_global_rank, this->get_mpi_communicator());
            }
        } /*else {
          // prevent deadlock.
          Point<dim> new_dike_point = Point<dim>();

          std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 1" << std::endl;
 particle_handler->sort_particles_into_subdomains_and_cells();

          std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 2" << std::endl;
  Utilities::MPI::sum(new_dike_point,this->get_mpi_communicator(),new_dike_point);
          std::cout << "el world_rank = " << world_rank << "/" << world_size << ": Flag 3" << std::endl;
        }*/



      if (world_rank == 0)
        {
          std::cout << "dike_location = ";
          for (unsigned int i = 0; i < dike_location.size(); ++i)
            {
              std::cout << dike_location[i] << ", ";
            }
          std::cout << std::endl;
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
              if (dike_location.size() > 1)
                {
                  double distance;
                  const Point<dim> &P = in.position[q];
                  for (unsigned int point_index = 0; point_index < dike_location.size()-1; ++point_index)
                    {
                      const Point<dim> &X1 = dike_location[point_index];
                      const Point<dim> &X2 = dike_location[point_index+1];
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
                          //    std::cout << point_index << "/" << dike_location.size() << ": P = " << P << ", X1 = " << X1 << ", X2 = " << X2 << ", distance = " << distance << ", min_distance = " << min_distance << std::endl;
                          //  }
                        }

                      if (distance < min_distance)
                        {
                          //if(in.position[i][0] > -1000. && in.position[i][0] < 1000. && in.position[i][1] > 49000 && in.position[i][1] < 51000)
                          //  std::cout << "distance = " << distance << ", min_distance = " << min_distance << std::endl;
                          min_distance = distance;
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
              if (dike_location.size() > 1)
                {
                  double distance;
                  const Point<dim> &P = in.position[q];
                  for (unsigned int point_index = 0; point_index < dike_location.size()-1; ++point_index)
                    {
                      const Point<dim> &X1 = dike_location[point_index];
                      const Point<dim> &X2 = dike_location[point_index+1];
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
                          //    std::cout << point_index << "/" << dike_location.size() << ": P = " << P << ", X1 = " << X1 << ", X2 = " << X2 << ", distance = " << distance << ", min_distance = " << min_distance << std::endl;
                          //  }
                        }

                      if (distance < min_distance)
                        {
                          //if (in.position[q][0] > -2000. && in.position[q][0] < -1000. && in.position[q][1] > 40000 && in.position[q][1] < 41000)
                          //  std::cout << "distance = " << distance << ", min_distance = " << min_distance << std::endl;
                          min_distance = distance;
                        }
                    }
                }
              if (min_distance < 1000 && this->get_timestep_number() > 0)
                {
                  out.viscosities[q] *= 0.1;
                  const double dike_injection_rate_double = 1e-134;
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
            dike_injection_fraction = dike_injection_rate[q] * this->get_timestep();

          if (dike_material_injection_fraction != 0.0)
            dike_injection_fraction = dike_material_injection_fraction;

          if (dike_injection_rate[q] > 0.0
              && this->get_timestep_number() > 0
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
                  if (c == injection_phase_index)
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
                  else
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
                    }
                }

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
