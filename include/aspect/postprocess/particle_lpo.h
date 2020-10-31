/*
 Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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

#ifndef _aspect_postprocess_particle_lpo_h
#define _aspect_postprocess_particle_lpo_h

#include <aspect/postprocess/interface.h>
#include <aspect/particle/world.h>

#include <aspect/simulator_access.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/base/data_out_base.h>
#include <tuple>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * A Postprocessor that creates particles, which follow the
     * velocity field of the simulation. The particles can be generated
     * and propagated in various ways and they can carry a number of
     * constant or time-varying properties. The postprocessor can write
     * output positions and properties of all particles at chosen intervals,
     * although this is not mandatory. It also allows other parts of the
     * code to query the particles for information.
     */
    template <int dim>
    class LPO : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        LPO();

        /**
         * Destructor.
         */
        ~LPO();


        /**
         * Initialize function.
         */
        virtual void initialize ();

        /**
         * Execute this postprocessor. Derived classes will implement this
         * function to do whatever they want to do to evaluate the solution at
         * the current time step.
         *
         * @param[in,out] statistics An object that contains statistics that
         * are collected throughout the simulation and that will be written to
         * an output file at the end of each time step. Postprocessors may
         * deposit data in these tables for later visualization or further
         * processing.
         *
         * @return A pair of strings that will be printed to the screen after
         * running the postprocessor in two columns; typically the first
         * column contains a description of what the data is and the second
         * contains a numerical value of this data. If there is nothing to
         * print, simply return two empty strings.
         */
        virtual
        std::pair<std::string,std::string> execute (TableHandler &statistics);

        /**
         * This funcion ensures that the particle postprocessor is run before
         * this postprocessor.
         */
        virtual
        std::list<std::string>
        required_other_postprocessors () const;

        /**
         * Todo
         */
        std::vector<std::vector<double>> random_draw_volume_weighting(std::vector<double> fv,
                                                                      std::vector<std::vector<double>> angles) const;

        /**
        * Todo
        */
        double wrap_angle(const double angle) const;

        /**
         * Todo
         */
        std::vector<double> euler_angles_from_rotation_matrix(const Tensor<2,3> &rotation_matrix) const;

        /**
         * Todo
         */
        Tensor<2,3> euler_angles_to_rotation_matrix(double phi1, double theta, double phi2) const;

        /**
        * Declare the parameters this class takes through input files.
        */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

      private:

        double end_time;
        /**
         * todo
         */
        enum class Output
        {
          olivine_volume_fraction, olivine_A_matrix, olivine_Euler_angles,
          enstatite_volume_fraction, enstatite_A_matrix, enstatite_Euler_angles,
          not_found
        };

        Output string_to_output_enum(std::string string);

        const double rad_to_degree = 180.0/M_PI;
        const double degree_to_rad = M_PI/180.0;

        mutable boost::lagged_fibonacci44497            random_number_generator;

        unsigned int random_number_seed;

        /**
         * todo
         */
        unsigned int n_grains;


        /**
         * Interval between output (in years if appropriate simulation
         * parameter is set, otherwise seconds)
         */
        double output_interval;

        /**
         * Records time for next output to occur
         */
        double last_output_time;

        /**
         * Set the time output was supposed to be written. In the simplest
         * case, this is the previous last output time plus the interval, but
         * in general we'd like to ensure that it is the largest supposed
         * output time, which is smaller than the current time, to avoid
         * falling behind with last_output_time and having to catch up once
         * the time step becomes larger. This is done after every output.
         */
        void set_last_output_time (const double current_time);

        /**
         * Consecutively counted number indicating the how-manyth time we will
         * create output the next time we get to it.
         */
        unsigned int output_file_number;

        /**
         * Graphical output format.
         */
        std::vector<std::string> output_formats;

        /**
         * A list of pairs (time, pvtu_filename) that have so far been written
         * and that we will pass to DataOutInterface::write_pvd_record
         * to create a master file that can make the association
         * between simulation time and corresponding file name (this
         * is done because there is no way to store the simulation
         * time inside the .pvtu or .vtu files).
         */
        std::vector<std::pair<double,std::string> > times_and_pvtu_file_names;

        /**
         * A corresponding variable that we use for the .visit files created
         * by DataOutInterface::write_visit_record. The second part of a
         * pair contains all files that together form a time step.
         */
        std::vector<std::pair<double,std::vector<std::string> > > times_and_vtu_file_names;

        /**
         * A list of list of filenames, sorted by timestep, that correspond to
         * what has been created as output. This is used to create a master
         * .visit file for the entire simulation.
         */
        std::vector<std::vector<std::string> > output_file_names_by_timestep;

        /**
         * A set of data related to XDMF file sections describing the HDF5
         * heavy data files created. These contain things such as the
         * dimensions and names of data written at all steps during the
         * simulation.
         */
        std::vector<XDMFEntry>  xdmf_entries;

        /**
         * VTU file output supports grouping files from several CPUs into one
         * file using MPI I/O when writing on a parallel filesystem. 0 means
         * no grouping (and no parallel I/O). 1 will generate one big file
         * containing the whole solution.
         */
        unsigned int group_files;

        /**
         * On large clusters it can be advantageous to first write the
         * output to a temporary file on a local file system and later
         * move this file to a network file system. If this variable is
         * set to a non-empty string it will be interpreted as a temporary
         * storage location.
         */
        std::string temporary_output_location;

        /**
         * File operations can potentially take a long time, blocking the
         * progress of the rest of the model run. Setting this variable to
         * 'true' moves this process into a background thread, while the
         * rest of the model continues.
         */
        bool write_in_background_thread;

        /**
         * Handle to a thread that is used to write master file data in the
         * background. The writer() function runs on this background thread.
         */
        Threads::Thread<void> background_thread_master;

        /**
         * What raw lpo data to write out
         */
        std::vector<Output> write_raw_lpo;

        /**
         * Whether computing raw Euler angles is needed.
         */
        bool compute_raw_euler_angles;

        /**
         * Handle to a thread that is used to write content file data in the
         * background. The writer() function runs on this background thread.
         */
        Threads::Thread<void> background_thread_content_raw;

        /**
         * What draw volume weighted lpo data to write out
         */
        std::vector<Output> write_draw_volume_weighted_lpo;

        /**
         * Whether computing weighted A matrix is needed.
         */
        bool compute_weighted_A_matrix;

        /**
         * Handle to a thread that is used to write content file data in the
         * background. The writer() function runs on this background thread.
         */
        Threads::Thread<void> background_thread_content_draw_volume_weighting;

        /**
         * Whether to compress the raw and weighed lpo data output files with zlib.
         */
        bool compress_lpo_data_files;

        /**
         * Stores the particle property fields which are ouptut to the
         * visualization file.
         */
        std::vector<std::string> exclude_output_properties;

        /**
         * A function that writes the text in the second argument to a file
         * with the name given in the first argument. The function is run on a
         * separate thread to allow computations to continue even though
         * writing data is still continuing. The function takes over ownership
         * of these arguments and deletes them at the end of its work.
         */
        static
        void writer (const std::string filename,
                     const std::string temporary_filename,
                     const std::string *file_contents,
                     const bool compress_contents);
    };
  }
}

#endif
