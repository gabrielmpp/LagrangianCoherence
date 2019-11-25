
LagrangianCoherence: Finding coherent behavior in atmospheric flow
===============================================
This library provides tools for computing parcel trajectories and applying Lagrangian diagnostics to identify coherent behavior in the atmospheric flow. The library is built using xarray, numpy and numba computational capabilities.



Code
-------------

- `Github <https://github.com/gabrielmpp/LagrangianCoherence/>`_

Usage
--------------



Module reference
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
     :maxdepth: 2
     :caption: Contents:

LCS
---------
.. automodule:: LCS
    :members: LCS
    :special-members: __init__, __call__

Parcel propagation
------------------
.. automodule:: LCS
     :members: parcel_propagation

Cauchy-green strain tensor
------------------------
.. automodule:: LCS
    :members: compute_deformation_tensor

