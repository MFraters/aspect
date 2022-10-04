/*
  Copyright (C) 2020 - 2021 by the authors of the ASPECT code.

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

#include <aspect/postprocess/visualization/relative_euler_pole_velocities_NA.h>
#include <aspect/postprocess/boundary_velocity_residual_statistics.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      RelativeEulerPoleVelocitiesNA<dim>::
      RelativeEulerPoleVelocitiesNA ()
        :
        DataPostprocessorVector<dim> ("relative_euler_pole_velocities_NA",
                                      update_values | update_quadrature_points)
      {}



      template <int dim>
      void
      RelativeEulerPoleVelocitiesNA<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        Assert ((computed_quantities[0].size() == dim), ExcInternalError());
        auto cell = input_data.template get_cell<dim>();

        for (unsigned int q=0; q<computed_quantities.size(); ++q)
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[q](d)= 0.;

        const double velocity_scaling_factor =
          this->convert_output_to_years() ? year_in_seconds : 1.0;

        for (unsigned int q=0; q<computed_quantities.size(); ++q)
          {
            // reference: https://www.sciencedirect.com/science/article/pii/S0012821X18301432
            std::array<double,3> euler_pole;
            const double longitude = -33.326*M_PI;
            const double lattitude = -46.094*M_PI;
            const double angular_speed = (1e-6/year_in_seconds)*sin(0.1780/180.*M_PI);
            euler_pole[0] = angular_speed*cos(lattitude)*cos(longitude);
            euler_pole[1] = angular_speed*cos(lattitude)*sin(longitude);
            euler_pole[2] = angular_speed*sin(lattitude);

            for (unsigned int d = 0; d < dim; ++d)
              computed_quantities[q](d) = euler_pole[d] - input_data.solution_values[q][d] * velocity_scaling_factor;
          }

      }

    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(RelativeEulerPoleVelocitiesNA,
                                                  "relative Euler pole velocities NA",
                                                  "")
    }
  }
}
