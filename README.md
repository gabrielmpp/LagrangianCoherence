# LagrangianCoherence

Library for computing the Finite-time Lyapunov Exponents (FTLE) of 2D flows built on xarray and scipy.
 
The main purpose of LagrangianCoherence is to allow the computation of parcel trajectories and the FTLE in 
atmospheric data. The trajectories are computed in a limited domain on the sphere by a 
‘Stable Extrapolation Two‐Time‐Level Scheme' (Hortal, 2002). The gradient of the flow-map is computed by
a simple finite-difference approximation. The possible outputs are 1) the FTLE, 2) the departure points or 3) the trajectories.





