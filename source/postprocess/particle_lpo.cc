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

#include <aspect/global.h>
#include <aspect/postprocess/particle_lpo.h>
#include <aspect/utilities.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <stdio.h>
#include <unistd.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    LPO<dim>::LPO ()
      :
      // the following value is later read from the input file
      output_interval (0),
      // initialize this to a nonsensical value; set it to the actual time
      // the first time around we get to check it
      last_output_time (std::numeric_limits<double>::quiet_NaN())
      ,output_file_number (numbers::invalid_unsigned_int),
      group_files(0),
      write_in_background_thread(false)
    {}

    template <int dim>
    LPO<dim>::~LPO ()
    {
      // make sure a thread that may still be running in the background,
      // writing data, finishes
      background_thread_master.join ();
      background_thread_content.join ();
    }


    template <int dim>
    // We need to pass the arguments by value, as this function can be called on a separate thread:
    void LPO<dim>::writer (const std::string filename, //NOLINT(performance-unnecessary-value-param)
                           const std::string temporary_output_location, //NOLINT(performance-unnecessary-value-param)
                           const std::string *file_contents)
    {
      std::string tmp_filename = filename;
      if (temporary_output_location != "")
        {
          tmp_filename = temporary_output_location + "/aspect.tmp.XXXXXX";

          // Create the temporary file; get at the actual filename
          // by using a C-style string that mkstemp will then overwrite
          std::vector<char> tmp_filename_x (tmp_filename.size()+1);
          std::strcpy(tmp_filename_x.data(), tmp_filename.c_str());
          const int tmp_file_desc = mkstemp(tmp_filename_x.data());
          tmp_filename = tmp_filename_x.data();

          // If we failed to create the temp file, just write directly to the target file.
          // We also provide a warning about this fact. There are places where
          // this fails *on every node*, so we will get a lot of warning messages
          // into the output; in these cases, just writing multiple pieces to
          // std::cerr will produce an unreadable mass of text; rather, first
          // assemble the error message completely, and then output it atomically
          if (tmp_file_desc == -1)
            {
              const std::string x = ("***** WARNING: could not create temporary file <"
                                     +
                                     tmp_filename
                                     +
                                     ">, will output directly to final location. This may negatively "
                                     "affect performance. (On processor "
                                     + Utilities::int_to_string(Utilities::MPI::this_mpi_process (MPI_COMM_WORLD))
                                     + ".)\n");

              std::cerr << x << std::flush;

              tmp_filename = filename;
            }
          else
            close(tmp_file_desc);
        }

      std::ofstream out(tmp_filename.c_str());

      AssertThrow (out, ExcMessage(std::string("Trying to write to file <") +
                                   filename +
                                   ">, but the file can't be opened!"))

      // now write and then move the tmp file to its final destination
      // if necessary
      out << *file_contents;
      out.close ();

      if (tmp_filename != filename)
        {
          std::string command = std::string("mv ") + tmp_filename + " " + filename;
          int error = system(command.c_str());

          AssertThrow(error == 0,
                      ExcMessage("Could not move " + tmp_filename + " to "
                                 + filename + ". On processor "
                                 + Utilities::int_to_string(Utilities::MPI::this_mpi_process (MPI_COMM_WORLD)) + "."));
        }

      // destroy the pointer to the data we needed to write
      delete file_contents;
    }

    template <int dim>
    std::pair<std::string,std::string>
    LPO<dim>::execute (TableHandler &statistics)
    {
      // if this is the first time we get here, set the last output time
      // to the current time - output_interval. this makes sure we
      // always produce data during the first time step
      if (std::isnan(last_output_time))
        last_output_time = this->get_time() - output_interval;

      // If it's not time to generate an output file or we do not write output
      // return early.
      if (this->get_time() < last_output_time + output_interval)
        return std::make_pair("","");

      if (output_file_number == numbers::invalid_unsigned_int)
        output_file_number = 0;
      else
        ++output_file_number;

      // Now prepare everything for writing the output and choose output format
      std::string particle_file_prefix_master = this->get_output_directory() +  "particle_LPO/particles-" + Utilities::int_to_string (output_file_number, 5);
      std::string particle_file_prefix_content = this->get_output_directory() +  "particle_LPO/LPO-" + Utilities::int_to_string (output_file_number, 5);

      const typename Particles::ParticleHandler<dim> &particle_handler = this->get_particle_world().get_particle_handler();

      std::stringstream string_stream_master;
      std::stringstream string_stream_content;

      // get particle data
      for (typename Particles::ParticleHandler<dim>::particle_iterator it = particle_handler.begin(); it != particle_handler.end(); ++it)
        {

          AssertThrow(it->has_properties(),
                      ExcMessage("No particle properties found. Make sure that the LPO particle property plugin is selected."));

          std::vector<double> volume_fractions_olivine(n_grains);
          std::vector<Tensor<2,3> > a_cosine_matrices_olivine(n_grains);
          std::vector<double> volume_fractions_enstatite(n_grains);
          std::vector<Tensor<2,3> > a_cosine_matrices_enstatite(n_grains);

          unsigned int id = it->get_id();
          const ArrayView<double> &properties = it->get_properties();

          const Particle::Property::ParticlePropertyInformation &property_information = this->get_particle_world().get_property_manager().get_data_info();

          AssertThrow(property_information.fieldname_exists("lpo water content") ,
                      ExcMessage("No LPO particle properties found. Make sure that the LPO particle property plugin is selected."));

          const unsigned int data_position = property_information.n_fields() == 0
                                             ?
                                             0
                                             :
                                             property_information.get_position_by_field_name("lpo water content");

          Point<dim> position = it->get_location();

          // write master file
          string_stream_master << id << " " << properties[data_position] << " " << position << std::endl;

          // write content file

          // loop over grain retrieve from data from each grain
          unsigned int data_grain_i = 0;
          for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
            {
              // retrieve volume fraction for olvine grains
              volume_fractions_olivine[grain_i] = properties[data_position + data_grain_i *
                                                             (Tensor<2,3>::n_independent_components + 1) + 1];

              // retrieve a_{ij} for olvine grains
              //Tensor<2,dim> a_cosine_matrices_olivine;
              for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
                {
                  const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                  a_cosine_matrices_olivine[grain_i][index] = properties[data_position + data_grain_i *
                                                                         (Tensor<2,3>::n_independent_components + 1) + 2 + i];
                }

              // retrieve volume fraction for enstatite grains
              volume_fractions_enstatite[grain_i] = properties[data_position + (data_grain_i+1) *
                                                               (Tensor<2,3>::n_independent_components + 1) + 1];

              // retrieve a_{ij} for enstatite grains
              //Tensor<2,dim> a_cosine_matrices;
              for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
                {
                  const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                  a_cosine_matrices_enstatite[grain_i][index] = properties[data_position + (data_grain_i+1) *
                                                                           (Tensor<2,3>::n_independent_components + 1) + 2 + i];
                }
              data_grain_i = data_grain_i + 2;
            }

          for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
            string_stream_content << id << " "
                                  << volume_fractions_olivine[grain_i] << " "
                                  << a_cosine_matrices_olivine[grain_i][0][0] << " " <<  a_cosine_matrices_olivine[grain_i][0][1] << " " <<  a_cosine_matrices_olivine[grain_i][0][2] << " "
                                  << a_cosine_matrices_olivine[grain_i][1][0] << " " <<  a_cosine_matrices_olivine[grain_i][1][1] << " " <<  a_cosine_matrices_olivine[grain_i][1][2] << " "
                                  << a_cosine_matrices_olivine[grain_i][2][0] << " " <<  a_cosine_matrices_olivine[grain_i][2][1] << " " <<  a_cosine_matrices_olivine[grain_i][2][2] << " "
                                  << volume_fractions_enstatite[grain_i] << " "
                                  << a_cosine_matrices_enstatite[grain_i][0][0] << " " <<  a_cosine_matrices_enstatite[grain_i][0][1] << " " <<  a_cosine_matrices_enstatite[grain_i][0][2] << " "
                                  << a_cosine_matrices_enstatite[grain_i][1][0] << " " <<  a_cosine_matrices_enstatite[grain_i][1][1] << " " <<  a_cosine_matrices_enstatite[grain_i][1][2] << " "
                                  << a_cosine_matrices_enstatite[grain_i][2][0] << " " <<  a_cosine_matrices_enstatite[grain_i][2][1] << " " <<  a_cosine_matrices_enstatite[grain_i][2][2] << std::endl;
        }

      std::string filename_master = particle_file_prefix_master + "." + Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD),4) + ".dat";
      std::string filename = particle_file_prefix_content + "." + Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD),4) + ".dat";

      std::string *file_contents_master = new std::string (string_stream_master.str());
      std::string *file_contents = new std::string (string_stream_content.str());

      if (write_in_background_thread)
        {
          // Wait for all previous write operations to finish, should
          // any be still active,
          background_thread_master.join ();

          // then continue with writing the master file
          background_thread_master = Threads::new_thread (&writer,
                                                          filename_master,
                                                          temporary_output_location,
                                                          file_contents_master);

          // Wait for all previous write operations to finish, should
          // any be still active,
          background_thread_content.join ();

          // then continue with writing our own data.
          background_thread_content = Threads::new_thread (&writer,
                                                           filename,
                                                           temporary_output_location,
                                                           file_contents);
        }
      else
        {
          writer(filename_master,temporary_output_location,file_contents_master);
          writer(filename,temporary_output_location,file_contents);
        }


      // up the next time we need output
      set_last_output_time (this->get_time());

      const std::string particle_lpo_output = particle_file_prefix_content;

      // record the file base file name in the output file
      statistics.add_value ("Particle LPO file name",
                            particle_lpo_output);
      return std::make_pair("Writing particle lpo output:", particle_lpo_output);
    }



    template <int dim>
    void
    LPO<dim>::set_last_output_time (const double current_time)
    {
      // if output_interval is positive, then update the last supposed output
      // time
      if (output_interval > 0)
        {
          // We need to find the last time output was supposed to be written.
          // this is the last_output_time plus the largest positive multiple
          // of output_intervals that passed since then. We need to handle the
          // edge case where last_output_time+output_interval==current_time,
          // we did an output and std::floor sadly rounds to zero. This is done
          // by forcing std::floor to round 1.0-eps to 1.0.
          const double magic = 1.0+2.0*std::numeric_limits<double>::epsilon();
          last_output_time = last_output_time + std::floor((current_time-last_output_time)/output_interval*magic) * output_interval/magic;
        }
    }


    template <int dim>
    void
    LPO<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Particles");
        {
          prm.enter_subsection("LPO");
          {
            prm.declare_entry ("Number of grains per praticle", "50",
                               Patterns::Integer (0),
                               "The number of grains of olivine and the number of grain of enstatite "
                               "each particle contains.");
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
        prm.enter_subsection("LPO");
        {
          prm.declare_entry ("Time between data output", "1e8",
                             Patterns::Double (0),
                             "The time interval between each generation of "
                             "output files. A value of zero indicates that "
                             "output should be generated every time step.\n\n"
                             "Units: years if the "
                             "'Use years in output instead of seconds' parameter is set; "
                             "seconds otherwise.");

          prm.declare_entry ("Write in background thread", "false",
                             Patterns::Bool(),
                             "File operations can potentially take a long time, blocking the "
                             "progress of the rest of the model run. Setting this variable to "
                             "`true' moves this process into a background threads, while the "
                             "rest of the model continues.");

          prm.declare_entry ("Temporary output location", "",
                             Patterns::Anything(),
                             "On large clusters it can be advantageous to first write the "
                             "output to a temporary file on a local file system and later "
                             "move this file to a network file system. If this variable is "
                             "set to a non-empty string it will be interpreted as a "
                             "temporary storage location.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

    }


    template <int dim>
    void
    LPO<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Particles");
        {
          prm.enter_subsection("LPO");
          {
            n_grains = prm.get_integer("Number of grains per praticle"); //10000;
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
        prm.enter_subsection("LPO");
        {
          output_interval = prm.get_double ("Time between data output");
          if (this->convert_output_to_years())
            output_interval *= year_in_seconds;

          //AssertThrow(this->get_parameters().run_postprocessors_on_nonlinear_iterations == false,
          //            ExcMessage("Postprocessing nonlinear iterations in models with "
          //                       "particles is currently not supported."));

          aspect::Utilities::create_directory (this->get_output_directory() + "particle_LPO/",
                                               this->get_mpi_communicator(),
                                               true);

          write_in_background_thread = prm.get_bool("Write in background thread");
          temporary_output_location = prm.get("Temporary output location");

          if (temporary_output_location != "")
            {
              // Check if a command-processor is available by calling system() with a
              // null pointer. System is guaranteed to return non-zero if it finds
              // a terminal and zero if there is none (like on the compute nodes of
              // some cluster architectures, e.g. IBM BlueGene/Q)
              AssertThrow(system((char *)nullptr) != 0,
                          ExcMessage("Usage of a temporary storage location is only supported if "
                                     "there is a terminal available to move the files to their final location "
                                     "after writing. The system() command did not succeed in finding such a terminal."));
            }
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();

    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(LPO,
                                  "lpo",
                                  "A Postprocessor that creates particles that follow the "
                                  "velocity field of the simulation. The particles can be generated "
                                  "and propagated in various ways and they can carry a number of "
                                  "constant or time-varying properties. The postprocessor can write "
                                  "output positions and properties of all particles at chosen intervals, "
                                  "although this is not mandatory. It also allows other parts of the "
                                  "code to query the particles for information.")
  }
}
