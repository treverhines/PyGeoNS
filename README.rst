PyGeoNS (Python-based Geodetic Network Smoother)
++++++++++++++++++++++++++++++++++++++++++++++++

PyGeoNS is a suite of command line executables that are used to 
estimate time dependent strain from geodetic data. This package is 
primarily intended to be used with daily GPS position timeseries, but 
it is also possible to bolster strain estimates with data from 
borehole/laser strain meters.

Note: This document is currently under construction. More 
documentation will be coming soon.

Executables
-----------
PyGeoNS contains the following command line executable functions. Call 
these functions with a '-h' flag to see more information.

* pygeons-toh5 : Converts from data a text file to an hdf5 file.
* ``pygeons-totext`` : Converts from data a hdf5 file to a csv file.
* ``pygeons-view`` : Starts an interactive map view and time series 
  view of vector data sets (e.g. displacements, deformation gradients, 
  etc.).
* ``pygeons-strain`` : Starts an interactive map view and time series 
  view of strain. 
* ``pygeons-clean`` : Starts the interactive cleaner, which is used to 
                     manually remove jumps and outliers.
* ``pygeons-crop`` : Bounds the spatial and temporal extent of the data 
  set.
* ``pygeons-tgpr`` : Temporally smooths and differentiates a data set.
* ``pygeons-sgpr`` : Spatially smooths and differentiates a data set.

Data Format
-----------
With the exception of ``pygeons-toh5``, all executables require the 
user to specify and HDF5 data file. An HDF5 data file can be generated 
from some csv data file formations using ``pygeons-toh5``. Each HDF5 
file must contain the following entries



