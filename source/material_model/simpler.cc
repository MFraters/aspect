/*
  Copyright (C) 2011 - 2023 by the authors of the ASPECT code.

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


#include <aspect/material_model/simpler.h>
#include <aspect/material_model/equation_of_state/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    bool
    Simpler<dim>::
    is_compressible () const
    {
      return equation_of_state.is_compressible ();
    }

    template <int dim>
    void
    Simpler<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      PrescribedDirectionalDilation<dim> *prescribed_directional_dilation = out.template get_additional_output<MaterialModel::PrescribedDirectionalDilation<dim>>();
      PrescribedPlasticDilation<dim> *prescribed_plastic_dilation = out.template get_additional_output<MaterialModel::PrescribedPlasticDilation<dim>>();

      // The Simpler model does not depend on composition
      EquationOfStateOutputs<dim> eos_outputs (1);

      thermal_conductivity.evaluate(in,out);

      for (unsigned int i=0; i<in.n_evaluation_points(); ++i)
        {
          if (prescribed_directional_dilation != nullptr)
            {
              const double dilation_x = 2e-16;
              const double dilation_y = 1e-16;//1e-16;
              if (std::fabs(in.position[i][0]) > -25e3 && std::fabs(in.position[i][0]) < 25e3 && in.position[i][1] > -25e3 && in.position[i][1] < 25e3)
                {
                  prescribed_directional_dilation->dilation_term[0][i] = dilation_x;
                  prescribed_directional_dilation->dilation_term[1][i] = dilation_y;//1e-18;
                  //prescribed_directional_dilation->dilation_term[dim-1][i] = 0;
                }
              else if ((((std::fabs(in.position[i][0]) > -400e3 && std::fabs(in.position[i][0]) < -350e3 && in.position[i][1] > -25e3 && in.position[i][1] < 25e3))|| (std::fabs(in.position[i][0]) > 350e3 && std::fabs(in.position[i][0]) < 400e3 && in.position[i][1] > -25e3 && in.position[i][1] < 25e3)))
                {
                  prescribed_directional_dilation->dilation_term[0][i] = -0.5*dilation_x;
                  prescribed_directional_dilation->dilation_term[1][i] = 0;
                }
              else if (((std::fabs(in.position[i][1]) > 350e3 && std::fabs(in.position[i][1]) < 400e3 && in.position[i][0] > -25e3 && in.position[i][0] < 25e3)|| (std::fabs(in.position[i][1]) > 350e3 && std::fabs(in.position[i][1]) < 400e3 && in.position[i][0] > -25e3 && in.position[i][0] < 25e3)))// && in.position[i][1] > 5e3 && in.position[i][1] < 95e3)
                {
                  prescribed_directional_dilation->dilation_term[1][i] = -0.5*dilation_y;
                  prescribed_directional_dilation->dilation_term[0][i] = 0;
                  //prescribed_directional_dilation->dilation_term[dim-1][i] = 0;
                }
              else
                {

                  prescribed_directional_dilation->dilation_term[0][i] = 0;
                  prescribed_directional_dilation->dilation_term[1][i] = 0;
                  prescribed_directional_dilation->dilation_term[dim-1][i] = 0;
                }
            }


          equation_of_state.evaluate(in, i, eos_outputs);

          out.viscosities[i] = constant_rheology.compute_viscosity();
          out.densities[i] = eos_outputs.densities[0];
          out.thermal_expansion_coefficients[i] = eos_outputs.thermal_expansion_coefficients[0];
          out.specific_heat[i] = eos_outputs.specific_heat_capacities[0];
          out.compressibilities[i] = eos_outputs.compressibilities[0];

          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }

    }


    template <int dim>
    void
    Simpler<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Simpler model");
        {
          EquationOfState::LinearizedIncompressible<dim>::declare_parameters (prm);
          ThermalConductivity::Constant<dim>::declare_parameters (prm);
          Rheology::ConstantViscosity::declare_parameters(prm,5e24);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    Simpler<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Simpler model");
        {
          equation_of_state.parse_parameters (prm);
          thermal_conductivity.parse_parameters (prm);
          constant_rheology.parse_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::none;
      this->model_dependence.density = NonlinearDependence::temperature;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
    }

    template <int dim>
    void
    Simpler<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {

      //Stokes additional RHS for prescribed dilation
      const unsigned int n_points = out.n_evaluation_points();
      if (out.template get_additional_output<MaterialModel::PrescribedDirectionalDilation<dim>>() == nullptr)
        {
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedDirectionalDilation<dim>> (n_points));
        }

      AssertThrow(!true //this->get_parameters().enable_prescribed_directional_dilation
                  ||
                  out.template get_additional_output<MaterialModel::PrescribedDirectionalDilation<dim>>()->dilation_term.size()
                  == dim, ExcInternalError());
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(Simpler,
                                   "simpler",
                                   "A material model that has constant values "
                                   "except for density, which depends linearly on temperature: "
                                   "$ \\rho(p,T) = \\left(1-\\alpha (T-T_0)\\right)\\rho_0.$ "
                                   "\n\n"
                                   "Note that this material model fills the role the ``simple'' material "
                                   "model was originally intended to fill, before the latter acquired "
                                   "all sorts of complicated temperature and compositional dependencies. ")
  }
}
