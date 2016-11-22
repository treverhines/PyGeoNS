# PYGEONS DEMONSTRATION #2:
# TEMPORALLY SMOOTHING GPS DATA

# make work directory
mkdir -p work

# This script demonstrates how to temporally smooth GPS data for a
# single station.  This station, MIDA, recorded displacements
# associated with the 2004 Parkfield earthquake and the 2003 San
# Simeon earthquake. By specifying the date of the earthquakes, we
# can prevent pygeons from smoothing over these events.

# convert the csv file to an HDF5 file. This file was downloaded from
# www.unavco.org and uses the CSV file format established by UNAVCO
# and the Plate Boundary Observatory. This file can be read by
# specifying the file type as "pbo_csv".
pygeons-toh5 MIDA.pbo.nam08.csv pbo_csv -o work/disp.h5

# the earthquake dates are the first day that the displacements are
# observed in the timeseries
PARKFIELD=2004-09-29
SANSIMEON=2003-12-23

# temporal cutoff frequency in days^-1. 0.033 corresponds to a
# wavelength of one month
CUTOFF=0.033
pygeons-tfilter work/disp.h5 --cutoff $CUTOFF --break_dates $PARKFIELD $SANSIMEON --fill interpolate -vv

# view the observed and smoothed displacements
pygeons-view work/disp.h5 work/disp.tfilter.h5 --data_set_labels observed smoothed

# write the smoothed solution to a csv file. the file is saved as work/disp.tfilter.csv
pygeons-tocsv work/disp.tfilter.h5
