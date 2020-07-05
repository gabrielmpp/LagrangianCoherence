# LagrangianCoherence

Library for computing the Finite-time Lyapunov Exponents (FTLE) of 2D flows built on xarray and scipy.
 
The main purpose of LagrangianCoherence is to allow the computation of parcel trajectories and the FTLE in 
atmospheric data. The trajectories are computed in a limited domain on the sphere by a 
‘Stable Extrapolation Two‐Time‐Level Scheme' (Hortal, 2002). The gradient of the flow-map is computed by
a simple finite-difference approximation. The possible outputs are 1) the FTLE, 2) the departure points or 3) the trajectories.

An idealized example of a vortex can be found at examples/ideal_vortex.py. The plot on the left shows the mixing of a dye after mixed for a fixed time under the influence of this vortex. The vortex departed from the red circle and is moving northeast. The plot on the right shows the FTLE associated with this flow. Ridges indicate attracting Lagrangian coherent structures.

<img src="https://github.com/gabrielmpp/LagrangianCoherence/blob/master/examples/figs/ideal_vortex.png?raw=true" width="300"><img src="https://github.com/gabrielmpp/LagrangianCoherence/blob/master/examples/figs/ideal_vortex_FTLE.png?raw=true" width="300">

## References

Hortal, M. (2002). The development and testing of a new two‐time‐level semi‐Lagrangian scheme (SETTLS) in the ECMWF forecast model. Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric sciences, applied meteorology and physical oceanography, 128(583), 1671-1687.


Haller, G. (2001). Distinguished material surfaces and coherent structures in three-dimensional fluid flows. Physica D: Nonlinear Phenomena, 149(4), 248-277.


