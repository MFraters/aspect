/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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


#include <aspect/boundary_velocity/euler_poles.h>
#include <aspect/utilities.h>
#include <aspect/global.h>


namespace aspect
{
  namespace BoundaryVelocity
  {

    template <int dim>
    Tensor<1,dim>
    EulerPoles<dim>::
    boundary_velocity (const types::boundary_id ,
                       const Point<dim> &position) const
    {
      // cartesian to spherical
      dealii::Point<2> point_spherical;
      const double position_norm = position.norm();
      auto position_normalized = position/position_norm;

      //point_spherical[0] = position.norm(); // R
      point_spherical[0] = std::atan2(position[1],position[0]); // Phi (long)
      //if (scoord[1] < 0.0)
      //scoord[1] += 2.0*const_pi; // correct phi to [0,2*pi]

      if (position_norm > std::numeric_limits<double>::min())
        point_spherical[1] = 0.5 * M_PI - std::acos(position[2]/position_norm); //lat
      else
        point_spherical[1] = 0.0;

      Point<dim> euler_pole;
      std::vector<Point<2>> polygon = {{-117./180.*M_PI, 32./180.*M_PI},
        {-100./180.*M_PI, 32./180.*M_PI},
        {-100./180.*M_PI, 58./180.*M_PI},
        {-137./180.*M_PI, 58./180.*M_PI}
      };
      //std::cout << "point_spherical = " << point_spherical << std::endl;
      // need to use absolute plate motions, because otherwise I get large flows in the mantle. This is because there is no resitance from the lower mantle
      // helpful info: https://geo.libretexts.org/Courses/University_of_California_Davis/UCD_GEL_56_-_Introduction_to_Geophysics/Geophysics_is_everywhere_in_geology.../04%3A_Plate_Tectonics/4.07%3A_Plate_Motions_on_a_Sphere
      // and https://cpb-us-e1.wpmucdn.com/sites.northwestern.edu/dist/8/1676/files/2017/10/320L4.rms-25x07zj.pdf
      //if(Utilities::polygon_contains_point<dim>(polygon,point_spherical))
      //{
      //  // NA: https://www.sciencedirect.com/science/article/pii/S0012821X18301432
      //  constexpr double longitude = -33.326/180.*M_PI;
      //  constexpr double lattitude = -46.094/180.*M_PI;
      //  euler_pole[0] = cos(lattitude)*cos(longitude);
      //  euler_pole[1] = cos(lattitude)*sin(longitude);
      //  euler_pole[2] = sin(lattitude)               ;
      //  //std::cout << "euler_pole = " << euler_pole << ", position = " << position << ", velocity = " << dealii::cross_product_3d(euler_pole,position) << std::endl;
      //}
      //else
      //{
      //  // PA: https://www.sciencedirect.com/science/article/pii/S0012821X18301432
      //  constexpr double longitude = 97.344/180.*M_PI;
      //  constexpr double lattitude = -59.790/180.*M_PI;
      //  euler_pole[0] = cos(lattitude)*cos(longitude);
      //  euler_pole[1] = cos(lattitude)*sin(longitude);
      //  euler_pole[2] = sin(lattitude)               ;
      ////std::cout << "euler_pole = " << euler_pole << ", position = " << position << ", velocity = " << dealii::cross_product_3d(euler_pole,position) << std::endl;
      //}
      const bool point_in_polygon = Utilities::polygon_contains_point<dim>(polygon,point_spherical);

      // NA: https://www.sciencedirect.com/science/article/pii/S0012821X18301432
      // NA                PA
      const double longitude = point_in_polygon ?  -33.326/180.*M_PI : 97.344/180.*M_PI;  //319.3/180.*M_PI;//−80.64 : 114.70); //
      const double lattitude = point_in_polygon ?  -46.094/180.*M_PI : -59.790/180.*M_PI; //-58.3/180.*M_PI;//−4.85  : −63.58); //
      const double angular_speed = (1e-6/year_in_seconds)*sin(point_in_polygon   ?  0.1780/180.*M_PI :  0.8023/180.*M_PI);//;0.651 : 0.209);   //
      euler_pole[0] = angular_speed*cos(lattitude)*cos(longitude);
      euler_pole[1] = angular_speed*cos(lattitude)*sin(longitude);
      euler_pole[2] = angular_speed*sin(lattitude)    ;

      // PA: https://www.sciencedirect.com/science/article/pii/S0012821X18301432
      //constexpr double longitude = 97.344/180.*M_PI;
      //constexpr double lattitude = -59.790/180.*M_PI;
      // return a zero tensor regardless of position
      //return position_norm*(0.8023e-6/31556952)*dealii::cross_product_3d(euler_pole,position);
      const double prefix = position_norm;
      //if(point_in_polygon)
      //  std::cout << "euler_pole = " << euler_pole << ", position = " << position_normalized << ", angular_speed = " << angular_speed << ", velocity = " << prefix * dealii::cross_product_3d(euler_pole,position_normalized) << std::endl;
      return prefix * dealii::cross_product_3d(euler_pole,position_normalized);
      return dim == 3 ? Point<dim>({prefix*(euler_pole[1]*position[2]-euler_pole[2]*position[1]),
                                    prefix*(euler_pole[2]*position[0]-euler_pole[0]*position[2]),
                                    prefix*(euler_pole[0]*position[1]-euler_pole[1]*position[0])
                                   }) : Point<dim>({0,0});
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace BoundaryVelocity
  {
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(EulerPoles,
                                            "euler poles",
                                            "Implementation of a model in which the boundary "
                                            "velocity is zero. This is commonly referred to as "
                                            "a ``stick boundary condition'', indicating that "
                                            "the material ``sticks'' to the material on the "
                                            "other side of the boundary.")
  }
}
