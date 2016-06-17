/**
 * @page changes_current Changes after the latest release (v1.4.0)
 *
 * <p> This is the list of changes made after the release of Aspect version
 * 1.4.0. All entries are signed with the names of the author. </p>
 *
 * <li> New: This is a new optional feature for the discontinuous temperature
 * and compositional solutions. After solving the advection equation, 
 * a "bound preserving limiter" working as a correction procedure is applied
 * to the discontinuous advection fields. The limiter will stabilize the 
 * discontinuous advection solutions and keep it in the range of user defined 
 * global maximum/minimum values. 
 * <br>
 * (Ying He, 2016/06/02)
 *
 * <li> New: Tests can now be marked that they are expected to fail by the
 * EXPECT FAILURE keyword in the .prm.
 * <br>
 * (Timo Heister, 2016/06/01)
 *
 * <li> New: There is a new boundary traction model "ascii data"
 * that prescribes the boundary traction according to pressure values
 * read from an ascii data file.
 * <br>
 * (Juliane Dannberg, 2016/05/24)
 *
 * <li> New: A material model plugin for visco-plastic rheologies,
 * which combines diffusion, dislocation or composite viscous
 * creep with a Drucker Prager yield criterion.
 * <br>
 * (John Naliboff, 2016/05/19)
 *
 * <li> Changed: The traction boundary conditions now use a new interface
 * that includes the boundary indicator (analogous to the velocity boundary
 * conditions).
 * <br>
 * (Juliane Dannberg, 2016/05/19)
 *
 * <li> New: There is a new visualization postprocessor "artificial viscosity
 * composition" to visualize the artificial viscosity of a compositional
 * field.
 * <br>
 * (Juliane Dannberg, Timo Heister, 2016/05/03)
 *
 * <li> New: Mesh refinement strategies based on artificial viscosity,
 * composition gradients, or composition threshold.
 * <br>
 * (Juliane Dannberg, Timo Heister, 2016/05/03)
 *
 * <ol>
 *
 * </ol>
 */
