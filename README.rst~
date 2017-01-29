PyGeoNS (Python-based Geodetic Network Strain software)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

What PyGeoNS can do
===================
*PyGeoNS* is a suite of command line executables that are used to 
smooth and differentiate geodetic GPS data in both space and time.  
This analysis is performed in a Bayesian framework, using Gaussian 
process regression, and thus the uncertainties on the data products 
are well-quantified and meaningful. This software is primarily 
intended for estimating time dependent strain rates from GPS networks 
with tens to hundreds of stations.

What PyGeoNS does not do
========================
The core processing algorithms used by *PyGeoNS* come from the *RBF* 
python package, which can be found `here 
<http://www.github.com/treverhines/RBF>`_. *PyGeoNS* mostly does the 
requisit data munging, and it provides functions to interactively view 
and clean the data. There are several assumptions that have been hard 
coded into *PyGeoNS* which may make this software inapplicable to your 
project. In particular, if the Earth's curvature is non-negligible in 
your study region, then the map projection used by *PyGeoNS* 
(transverse-mercator) would not be appropriate. Additionally, *PyGeoNS* 
is designed to handle data with daily sampling rates. Less frequent 
sampling rates can be handled, such as for campaign GPS, but *PyGeoNS* 
cannot use data that has been sampled at higher frequencies. If these 
limitations conflict with your project needs, then it may be better to 
directly interface with the *RBF* package. Finally, the plotting 
functions in *PyGeoNS* are intended to be a convenient way of 
interactively viewing GPS data, but they are not intended to be 
customizable to meet everyones plotting needs. 

Installation
============
*PyGeoNS* requires the standard scientific python packages, which can be 
found in the base Anaconda python installation 
(http://www.continuum.io/downloads). Additionally, *PyGeoNS* requires 
that the *RBF* package be installed 
(http://www.github.com/treverhines/RBF). Once these dependencies are 
satisfied, this package can be downloaded and installed with the 
following commands

.. code-block:: bash

  $ git clone http://www.github.com/treverhines/PyGeoNS.git
  $ python setupy.py install

Executables
===========
*PyGeoNS* contains the following command line executable functions. Call 
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
*PyGeoNS* is currently able to read three text file formats: PBO csv 
files, PBO pos files, and a csv file format designed for *PyGeoNS*. See 
www.unavco.org for information on the PBO data file formats. An 
example of each file format is provided below.

PBO CSV
-------
.. code-block::

  PBO Station Position Time Series.
  Format Version, 1.2.0
  Reference Frame, NAM08
  4-character ID, P403
  Station name, FloeQuaryGWA2005
  Begin Date, 2005-09-13
  End Date, 2017-01-26
  Release Date, 2017-01-27
  Source file, P403.pbo.nam08.pos
  Offset from source file, 48.54 mm North, 60.55 mm East, -5.06 mm Vertical
  Reference position, 48.0623223017 North Latitude, -124.1408746693 East Longitude, 284.67725 meters elevation
  Date, North (mm), East (mm), Vertical (mm), North Std. Deviation (mm), East Std. Deviation (mm), Vertical Std. Deviation (mm), Quality,  
  2005-09-13,0.00, 0.00, 0.00, 4.71, 3.14, 13.2, repro,
  2005-09-14,7.43, 8.65, 2.37, 1.85, 1.34, 5.6, repro,
  ...
  2017-01-26,98.68, 132.58, 6.00, 1.93, 1.49, 6.34, rapid,

PBO POS
-------
.. code-block::

  PBO Station Position Time Series. Reference Frame : NAM08
  Format Version: 1.1.0
  4-character ID: P403
  Station name  : FloeQuaryGWA2005
  First Epoch   : 20050913 120000
  Last Epoch    : 20170126 120000
  Release Date  : 20170127 235743
  XYZ Reference position :  -2396874.51122 -3534734.44146  4721722.14918 (NAM08)
  NEU Reference position :    48.0623223017  235.8591253307  284.67725 (NAM08/WGS84)
  Start Field Description
  YYYYMMDD      Year, month, day for the given position epoch
  HHMMSS        Hour, minute, second for the given position epoch
  JJJJJ.JJJJJ   Modified Julian day for the given position epoch
  X             X coordinate, Specified Reference Frame, meters
  Y             Y coordinate, Specified Reference Frame, meters
  Z             Z coordinate, Specified Reference Frame, meters
  Sx            Standard deviation of the X position, meters
  Sy            Standard deviation of the Y position, meters
  Sz            Standard deviation of the Z position, meters
  Rxy           Correlation of the X and Y position
  Rxz           Correlation of the X and Z position
  Ryz           Correlation of the Y and Z position
  Nlat          North latitude, WGS-84 ellipsoid, decimal degrees
  Elong         East longitude, WGS-84 ellipsoid, decimal degrees
  Height (Up)   Height relative to WGS-84 ellipsoid, m
  dN            Difference in North component from NEU reference position, meters
  dE            Difference in East component from NEU reference position, meters
  du            Difference in vertical component from NEU reference position, meters
  Sn            Standard deviation of dN, meters
  Se            Standard deviation of dE, meters
  Su            Standard deviation of dU, meters
  Rne           Correlation of dN and dE
  Rnu           Correlation of dN and dU
  Reu           Correlation of dEand dU
  Soln          "rapid", "final", "suppl/suppf", "campd", or "repro" corresponding to products  generated with rapid or final orbit products, in supplemental processing, campaign data processing or reprocessing
  End Field Description
  *YYYYMMDD HHMMSS JJJJJ.JJJJ         X             Y             Z            Sx        Sy       Sz     Rxy   Rxz    Ryz            NLat         Elong         Height         dN        dE        dU         Sn       Se       Su      Rne    Rnu    Reu  Soln
   20050913 120000 53626.5000 -2396874.58357 -3534734.44007  4721722.12054  0.00645  0.00812  0.00994  0.811 -0.686 -0.775      48.0623218656  235.8591245168  284.68231    -0.04854  -0.06055   0.00506    0.00471  0.00314  0.01320  0.163 -0.115 -0.095 repro
   20050914 120000 53627.5000 -2396874.57419 -3534734.44167  4721722.12726  0.00261  0.00353  0.00416  0.793 -0.733 -0.788      48.0623219323  235.8591246330  284.68468    -0.04111  -0.05190   0.00743    0.00185  0.00134  0.00560 -0.002 -0.141 -0.016 repro
   ...
   20170126 120000 57779.5000 -2396874.43473 -3534734.45725  4721722.19088  0.00295  0.00382  0.00479  0.797 -0.776 -0.801      48.0623227520  235.8591262989  284.68831     0.05014   0.07203   0.01106    0.00193  0.00149  0.00634 -0.045 -0.073 -0.110 rapid

PyGeoNS CSV
-----------
The *PyGeoNS* CSV file only contains information that *PyGeoNS* uses, 
making it unambigous which fields can influence the results. For 
example, there is no reference frame information in the *PyGeoNS* csv 
format because *PyGeoNS* does not ever use that information. 

.. code-block::

  4-character id, P403
  begin date, 2005-09-13
  end date, 2017-01-26
  longitude, 235.859125331 E
  latitude, 48.0623223017 N
  units, meters**1 days**0
  date, north, east, vertical, north std. deviation, east std. deviation, vertical std. deviation
  2005-09-13, -4.854000e-02, -6.055000e-02, 5.060000e-03, 4.710000e-03, 3.140000e-03, 1.320000e-02
  2005-09-14, -4.111000e-02, -5.190000e-02, 7.430000e-03, 1.850000e-03, 1.340000e-03, 5.600000e-03
  ...
  2017-01-26, 5.014000e-02, 7.203000e-02, 1.106000e-02, 1.930000e-03, 1.490000e-03, 6.340000e-03

HDF5 Data Format
================
To cut out overhead associated with reading and writing, most *PyGeoNS* 
executables read from and write to HDF5 files. Any of the above text 
file formats can be converted to an HDF5 file by doing the following. 
First, concatenate the data files for each station into one file 
separated by ``***``. For example, if the data files are in the 
current directory and contain a ``.csv`` extension then they can be 
concatenated with the following sed incantation

.. code-block::

  $ sed -s 'a***' *.csv | sed '$d' > data.csv 

Second, convert the new text file to an HDF5 file with the *PyGeoNS* 
command ``pygeons-toh5`` and use the ``--file_type`` flag followed by 
either ``csv``, ``pbocsv``, or ``pbopos``. By default, this is set to 
``csv``, indicating the file is a *PyGeoNS* csv file. Once you have 
converted the data to an HDF5 file, it can be passed as an argument to 
the remaining *PyGeoNS* executables for analysis and processing. An HDF5 
file can be converted back to a *PyGeoNS* csv file using 
``pygeons-totext`` followed by the file name. 

An HDF5 file can be read using, for example, the h5py package in 
python. Each HDF5 file contain the following entries

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
  if ``time_exponent`` is -1 and ``space_exponent`` is 1 then the units 
  should be in meters per day. If data is missing for a particular 
  time and station then it should be set to nan.
* ``east_std_dev``, ``north_std_dev``, ``vertical_std_dev`` : Array of 
  floats with shape (Nt,Nx). One standard deviation uncertainties for 
  each component of the data.  The units should be the same as those 
  used for the data components. If data is missing for a particular 
  time and station then it should be set to inf.
* ``time_exponent`` : Integer. This indicates the power of the time 
  units for the data. -1 indicates that the data is a rate, -2 indicates 
  an acceleration, etc.
* ``space_exponent`` : Integer. Indicates the power of the spatial 
  units for the data.
  
Demonstration
=============

See the bash scripts``demo/demo1/run.sh`` and ``demo/demo2/run.sh`` 
for examples of how to use PyGeoNS.  These scripts will open several 
interactive figures. Use the arrow keys to scroll between stations and 
time epochs. Additional instructions will be printed out when the 
figures open. Here is a figure produced from ``demo/demo2/run.sh``, 
which shows the estimate strain rates during a recent slow slip event 
in Washington.

Map view of strain rates during a slow slip event
.. image:: demo/demo2/figures/map_view.png

Time series of strain rate at the station indicated by the black dot
.. image:: demo/demo2/figures/time_series.png

