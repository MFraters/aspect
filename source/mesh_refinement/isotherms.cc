/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/

#include <aspect/mesh_refinement/isotherms.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/derivative_approximation.h>

namespace aspect
{
namespace MeshRefinement
{
template <int dim>
void
Isotherms<dim>::tag_additional_cells () const
{
    if(this->get_dof_handler().n_dofs() != 0) {

        /// Gather all information on the state of the system ///
        LinearAlgebra::BlockVector vec_distributed (this->introspection().index_sets.system_partitioning,
                this->get_mpi_communicator());

        const Quadrature<dim> quadrature(this->get_fe().base_element(this->introspection().base_elements.temperature).get_unit_support_points());
        std::vector<types::global_dof_index> local_dof_indices (this->get_fe().dofs_per_cell);
        FEValues<dim> fe_values (this->get_mapping(),
                                 this->get_fe(),
                                 quadrature,
                                 update_quadrature_points | update_values);
        std::vector<double> pressure_values(quadrature.size());
        std::vector<double> temperature_values(quadrature.size());

        /// the values of the compositional fields are stored as blockvectors for each field
        /// we have to extract them in this structure
        std::vector<std::vector<double> > prelim_composition_values (this->n_compositional_fields(),
                std::vector<double> (quadrature.size()));
        std::vector<std::vector<double> > composition_values (quadrature.size(),
                std::vector<double> (this->n_compositional_fields()));

        typename MaterialModel::Interface<dim>::MaterialModelInputs in(quadrature.size(), this->n_compositional_fields());
        typename MaterialModel::Interface<dim>::MaterialModelOutputs out(quadrature.size(), this->n_compositional_fields());

        typename DoFHandler<dim>::active_cell_iterator
        cell = this->get_dof_handler().begin_active(),
        endc = this->get_dof_handler().end();
        for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
            {
                bool coarsen = false;
                bool clear_refine = false;
                bool refine = false;
                bool clear_coarsen = false;
                fe_values.reinit(cell);
                fe_values[this->introspection().extractors.pressure].get_function_values (this->get_solution(),
                        pressure_values);
                fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
                        temperature_values);
                for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                    fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                            prelim_composition_values[c]);

                cell->get_dof_indices (local_dof_indices);
                in.position = fe_values.get_quadrature_points();
                in.strain_rate.resize(0);// we are not reading the viscosity
                for (unsigned int i=0; i<quadrature.size(); ++i)
                {
                    in.temperature[i] = temperature_values[i];
                    in.pressure[i] = pressure_values[i];
                    for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                        in.composition[i][c] = prelim_composition_values[c][i];
                }
                this->get_material_model().evaluate(in, out);

                // all data has been gethered, now do something with it.

                std::vector<std::vector<int> > isotherms_i(number_of_isotherms,std::vector<int>(2));
                // isotherms_i[0] is the minimum, and isotherms_i[1] is the maximum cell level.

                // convert max,max-1,max-2 etc and min, min-1,min-2 to numbers in integer
                for ( unsigned int it1 = 0; it1 != isotherms_i.size(); ++it1 )
                {
                    if(isotherms[it1][0].compare(0,3,"max")==0) {
                        if(isotherms[it1][0].compare(0,4,"max-")==0) {
                            std::vector<std::string> tmpNumber = Utilities::split_string_list(isotherms[it1][0],'-');
                            isotherms_i[it1][0]=maximum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
                        } else {
                            isotherms_i[it1][0]=maximum_refinement_level;
                        }
                    } else if(isotherms[it1][0].compare(0,3,"min")==0) {
                        if(isotherms[it1][0].compare(0,4,"min+")==0) {
                            std::vector<std::string> tmpNumber = Utilities::split_string_list(isotherms[it1][0],'+');
                            isotherms_i[it1][0]=minimum_refinement_level+Utilities::string_to_int(tmpNumber[1]);
                        } else {
                            isotherms_i[it1][0]=minimum_refinement_level;
                        }
                    } else if(!isotherms[it1][0].empty() && isotherms[it1][0].find_first_not_of("0123456789") == std::string::npos) {
                        isotherms_i[it1][0]=Utilities::string_to_int(isotherms[it1][0]);
                    } else {

                        AssertThrow (true,
                                     ExcMessage ("Not able to read input at inputline "
                                                 +
                                                 Utilities::int_to_string(it1)
                                                 +
                                                 ": "
                                                 +
                                                 isotherms[it1][0]
                                                )
                                    );
                    }

                    if(isotherms[it1][1].compare(0,3,"max")==0) {
                        if(isotherms[it1][1].compare(0,4,"max-")==0) {
                            std::vector<std::string> tmpNumber = Utilities::split_string_list(isotherms[it1][1],'-');
                            isotherms_i[it1][1]=maximum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
                        } else {
                            isotherms_i[it1][1]=maximum_refinement_level;
                        }
                    } else if(isotherms[it1][1].compare(0,3,"min")==0) {
                        if(isotherms[it1][1].compare(0,4,"min+")==0) {
                            std::vector<std::string> tmpNumber = Utilities::split_string_list(isotherms[it1][1],'+');
                            isotherms_i[it1][1]=minimum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
                        } else {
                            isotherms_i[it1][1]=minimum_refinement_level;
                        }
                    } else if(!isotherms[it1][1].empty() && isotherms[it1][1].find_first_not_of("0123456789") == std::string::npos) {
                        isotherms_i[it1][1]=Utilities::string_to_int(isotherms[it1][1]);
                    } else {
                        AssertThrow (true,
                                     ExcMessage ("Not able to read input at inputline "
                                                 +
                                                 Utilities::int_to_string(it1)
                                                 +
                                                 ": "
                                                 +
                                                 isotherms[it1][0]
                                                )
                                    );
                    }
                }
                /// We have now converted max,max-1,max-2 etc and min, min-1,min-2 to numbers in integer

                for (unsigned int i=0; i<quadrature.size(); ++i)
                {
                    for ( unsigned int it1 = 0; it1 != number_of_isotherms; ++it1 )
                    {

                        /// deterine if cell should be refined or coarsened
                        if(excludeComposition > 0 && excludeComposition <= static_cast <signed int> (this->n_compositional_fields())) { // static cast to prevent compiler warnings. Will only go wrong when there are more compositional fields then a the positive part of an int can handle ;)
                            // there is a exclude compostion (>0) and  exclude composition is smaller or equal to the current composition
                            if(rint(in.temperature[i]) <= Utilities::string_to_int(isotherms[it1][3]) && rint(in.temperature[i]) >= Utilities::string_to_int(isotherms[it1][2]) && in.composition[it1][excludeComposition]<0.5) {
                                // the temperature is between the isotherms and the exclude composition is smaller then 0.5 at this location

                                // If the current refinement level is smaller or equal to the refinement minimum level, any coarsening flags should be cleared.
                                if (cell->level() <= isotherms_i[it1][0]) {
                                    clear_coarsen = true;
                                }
                                // If the current refinement level is smaller then the minimum level, a refinment flag should be placed, and we don't have to look any further.
                                if (cell->level() <  isotherms_i[it1][0])
                                {
                                    refine = true;
                                    break;
                                }

                                // If the current refinement level is larger or equal to the maximum refinement level, any refinement flag should be cleared.
                                if (cell->level() >= isotherms_i[it1][1]) {
                                    clear_refine = true;
                                }
                                // If the current refinement level is larger then the maximum level, a coarsening flag should be placed, and we don't have to look any further.
                                if (cell->level() >  isotherms_i[it1][1])
                                {
                                    coarsen = true;
                                    break;
                                }

                            }
                        } else {
                            // there is not a exclude compostion (>0) and/or the exclude composition is larger or equal to the current composition. Now we don't have to check the value of the composition anymore.
                            if(rint(in.temperature[i]) <= Utilities::string_to_int(isotherms[it1][3]) && rint(in.temperature[i]) >= Utilities::string_to_int(isotherms[it1][2])) {
                                // the temperature is between the isotherms
                                // If the current refinement level is smaller or equal to the refinement minimum level, any coarsening flags should be cleared.
                                if (cell->level() <= isotherms_i[it1][0]) {
                                    clear_coarsen = true;
                                }
                                // If the current refinement level is smaller then the minimum level, a refinment flag should be placed, and we don't have to look any further.
                                if (cell->level() <  isotherms_i[it1][0])
                                {
                                    refine = true;
                                    break;
                                }

                                // If the current refinement level is larger or equal to the maximum refinement level, any refinement flag should be cleared.
                                if (cell->level() >= isotherms_i[it1][1]) {
                                    clear_refine = true;
                                }
                                // If the current refinement level is larger then the maximum level, a coarsening flag should be placed, and we don't have to look any further.
                                if (cell->level() >  isotherms_i[it1][1])
                                {
                                    coarsen = true;
                                    break;
                                }

                            }
                        }
                    }
                }

                /// Perform the actual placement of the coarsening and refinement flags
                if (clear_refine)
                    cell->clear_refine_flag ();
                if (coarsen)
                    cell->set_coarsen_flag ();
                if (clear_coarsen)
                    cell->clear_coarsen_flag ();
                if (refine)
                    cell->set_refine_flag ();
            }
    }
}

template <int dim>
void
Isotherms<dim>::
declare_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Mesh refinement");
    {

        prm.enter_subsection("Isotherms");
        {

            prm.declare_entry ("Exclude composition", "-1",
                               Patterns::Double (),
                               "Number of mininum and maximum areas defined by isotherms are going to be defined."
                              );
            prm.declare_entry ("List of isotherms", "",
                               Patterns::List(Patterns::List(Patterns::Anything(),0,100000000,","),0,100000000,";"),
                               "Number of mininum and maximum areas defined by isotherms are going to be defined."
                              );

        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}

template <int dim>
void
Isotherms<dim>::parse_parameters (ParameterHandler &prm)
{
    prm.enter_subsection("Mesh refinement");
    {
        minimum_refinement_level = Utilities::string_to_int(prm.get ("Minimum refinement level"));
        maximum_refinement_level = Utilities::string_to_int(prm.get("Initial global refinement")) + Utilities::string_to_int(prm.get("Initial adaptive refinement"));
        prm.enter_subsection("Isotherms");
        {
            excludeComposition = Utilities::string_to_int(prm.get("Exclude composition"));
            std::vector<std::string> isotherms_outer_loop = Utilities::split_string_list(prm.get ("List of isotherms"),';');
            // process the List of isotherms to get all data out of the structure
            number_of_isotherms = isotherms_outer_loop.size();
            isotherms.resize(number_of_isotherms,std::vector<std::string>(4,"0"));
            // loop through all given isotherms
            for ( unsigned int it1 = 0; it1 != number_of_isotherms; ++it1 )
            {
                std::vector<std::string> isotherms_inner_loop = Utilities::split_string_list(isotherms_outer_loop[it1]);
                // The minimum and maximum refinemnt levels respecively
                isotherms[it1][0]=isotherms_inner_loop[0];
                isotherms[it1][1]=isotherms_inner_loop[1];
                // The temperatures for the min and max isotherms respectively
                isotherms[it1][2]=isotherms_inner_loop[2];
                isotherms[it1][3]=isotherms_inner_loop[3];

            }
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();
}
}
}

// explicit instantiations
namespace aspect
{
namespace MeshRefinement
{
ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(Isotherms,
        "isotherms",
        "A mesh refinement criterion that ensures a "
        "maximum and minimum refinement level between "
        "two temperatures (isotherms), with the posibility "
        "to exclude a composition from this criterion. "
        "To accomplish this there are two parameters "
        "available: 'Exclude composition' and 'List of "
        "isotherms'. The first parameter takes the "
        "number of the compositional field and excludes "
        "it from the min/max refinment level prodedure. "
        "The second parameters takes a list of isoterm "
        "parameters for one isoterm separated by a ';'. "
        "Each line of isoterm parameters contains four "
        "subparameters. The first subparameter sets "
        "the minimun refinement level for this isoterm, "
        "the second subparameter sets the maximum "
        "refiment level for this isoterm, the third "
        "subparameter sets the minimum temperature for "
        "this isoterm and the fourth subparameter sets "
        "the maximum termperature for the isoterm. "
        "The minimum and maximum level can be indicated "
        "by the absolute number of the refinement level "
        "or by using the key words 'min' or 'max', "
        "corrosponding respecively to the set minimum "
        "and maximum refinment level. The keywords "
        "'min' and 'max' may also be used in combination "
        "with addition or substraction, e.g. 'max-1' "
        "or 'min+1'. For formating it is usefull to "
        "make use of the build in option of ASPECT "
        "to split parameters over several line by "
        "putting ' \\' at the end of the line. "
        "\n\n"
        "Example input format of List of isotherms:\n"
        "set List of isotherms = "
        "max,\t max,\t 0,\t 1000; \\ \n\t\t 5,\t max-1,\t"
        " 1000,\t 1500; \\ \n\t\t min,\t max-1,\t 1500,\t"
        " 2000")
}
}
