PyGeoNS (Python-based Geodetic Network Smoother)
++++++++++++++++++++++++++++++++++++++++++++++++

PyGeoNS is a suite of command line executables that are used to 
estimate time dependent strain from geodetic data. This package is 
primarily intended to be used with daily GPS position timeseries, but 
it is also possible to bolster strain estimates with data from 
borehole/laser strain meters.

Note: This document is currently under construction. More 
documentation will be coming soon.

Installation
============

Executables
===========
PyGeoNS contains the following command line executable functions. Call 
these functions with a '-h' flag to see more information.

* ``pygeons-toh5`` : Converts data from a text file to an hdf5 file.
* ``pygeons-totext`` : Converts data from a hdf5 file to a csv file.
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

Text Data Format
================

PBO CSV
-------

PyGeoNS CSV
-----------

PBS POS
-------

HDF5 Data Format
================
With the exception of ``pygeons-toh5``, all executables require the 
user to specify and HDF5 data file. An HDF5 data file can be generated 
from some types of csv data file formats using ``pygeons-toh5``. Each 
HDF5 file must contain the following entries

* ``time`` : Array of integers with shape (Nt,). Integer values of 
  modified Julian dates.
* ``id`` : Array of strings with shape (Nx,). 4-character IDs for each 
  station.
* ``longitude``, ``latitude`` : Array of floats with shape (Nx,). 
  Coordinates for each station.
* ``east``, ``north``, ``vertical`` : Array of floats with shape 
  (Nt,Nx). These are the data components. The units should be in terms 
  of meters and days and should be consistent with the values 
  specified for ``space_exponent`` and ``time_exponent``. For example, 
  if ``time_exponent`` is -1 and *space_exponent* is 1 then the units 
  should be in meters per day. If data is missing for a particular 
  time and station then it should be set to nan.
* ``east_std``, ``north_std``, ``vertical_std`` : Array of floats with 
  shape (Nt,Nx). One standard deviation uncertainties for each 
  component of the data.  The units should be the same as those used 
  for the data components. If data is missing for a particular time 
  and station then it should be set to inf.
* ``time_exponent`` : Integer. This indicates the power of the time 
  units for the data. -1 indicates that the data is a rate, -2 indicates 
  an acceleration, etc.
* ``space_exponent`` : Integer. Indicates the power of the spatial 
  units for the data.

  
Demonstration
=============


