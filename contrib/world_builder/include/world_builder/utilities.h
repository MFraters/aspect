

#ifndef _world_builder_utilities_h
#define _world_builder_utilities_h


#include <deal.II/base/point.h>
#include <world_builder/coordinate_system.h>
#include <world_builder/coordinate_systems/interface.h>

using namespace dealii;


  namespace WorldBuilder
  {

  namespace CoordinateSystems
	{
  class Interface;
	}
    namespace Utilities
    {

      /**
       * Given a 2d point and a list of points which form a polygon, computes if the point
       * falls within the polygon.
       */
      bool
      polygon_contains_point(const std::vector<std::array<double,2> > &point_list,
                             const std::array<double,2> &point);

      /**
       * Given a 2d point and a list of points which form a polygon, compute the smallest
       * distance of the point to the polygon. The sign is negative for points outside of
       * the polygon and positive for points inside the polygon.
       */
      double
      signed_distance_to_polygon(const std::vector<std::array<double,2> > &point_list_,
                                 const std::array<double,2> &point_);


      /*
      * A class that represents a point in a chosen coordinate system.
      */
      class NaturalCoordinate
      {
        public:
          /**
           * Constructor based on providing the geometry model as a pointer
           */
          NaturalCoordinate(const std::array<double,3> &position,
                            const ::WorldBuilder::CoordinateSystems::Interface &coordinate_system);

          /**
           * Returns the coordinates in the given coordinate system, which may
           * not be Cartesian.
           */
          std::array<double,3> &get_coordinates();

          /**
           * The coordinate that represents the 'surface' directions in the
           * chosen coordinate system.
           */
          std::array<double,2> get_surface_coordinates() const;

          /**
           * The coordinate that represents the 'depth' direction in the chosen
           * coordinate system.
           */
          double get_depth_coordinate() const;

        private:
          /**
           * An enum which stores the the coordinate system of this natural
           * point
           */
          CoordinateSystem coordinate_system;

          /**
           * An array which stores the coordinates in the coordinates system
           */
          std::array<double,3> coordinates;
      };

      /**
       * Returns spherical coordinates of a Cartesian point. The returned array
       * is filled with radius, phi and theta (polar angle). If the dimension is
       * set to 2 theta is omitted. Phi is always normalized to [0,2*pi].
       *
       */
      std::array<double,3>
      cartesian_to_spherical_coordinates(const Point<3> &position);

      /**
       * Return the Cartesian point of a spherical position defined by radius,
       * phi and theta (polar angle). If the dimension is set to 2 theta is
       * omitted.
       */
      Point<3>
      spherical_to_cartesian_coordinates(const std::array<double,3> &scoord);

      /**
       * Returns ellipsoidal coordinates of a Cartesian point. The returned array
       * is filled with phi, theta and radius.
       *
       */
      std::array<double,3>
      cartesian_to_ellipsoidal_coordinates(const Point<3> &position,
                                           const double semi_major_axis_a,
                                           const double eccentricity);

      /**
       * Return the Cartesian point of a ellipsoidal position defined by phi,
       * phi and radius.
       */
      Point<3>
      ellipsoidal_to_cartesian_coordinates(const std::array<double,3> &phi_theta_d,
                                           const double semi_major_axis_a,
                                           const double eccentricity);

      /**
       * A function that takes a string representation of the name of a
       * coordinate system (as represented by the CoordinateSystem enum)
       * and returns the corresponding value.
       */
      CoordinateSystem
      string_to_coordinate_system (const std::string &);
    }
  }


#endif