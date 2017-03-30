# This script demonstrates how to download GPS data from UNAVCO and
# then use PyGeoNS to calculate time dependent strain rates. This
# script performs the following tasks:
#
#   1. Download GPS data from the urls in urls.txt
#   2. Concatenate the data files into a single text file
#   3. Convert the text file to an HDF5 file
#   4. Crop out data prior to 2015-01-01 and after 2017-01-01
#   5. Temporally differentiate the data to get velocities
#   6. Spatially differentiate the data to get deformation gradients
#   7. View the time dependent strain

# Define the parameters describing the prior Gaussian processes used
# to temporally smooth and differentiate displacements and then
# spatially smooth and differentiate velocities. See demo/demo3/run.py
# for a demonstration of how these values were chosen.
DISP_CLS=0.05  # Characteristic time scale of the prior Gaussian
               # process for displacements. This is in years.
DISP_STD=5.0   # Standard deviation of the prior Gaussian process for
               # displacements. This is in mm.
DISP_ORDER=1   # Order of the polynomial null space. Setting this to 1
               # means that constant and linear trends in the data
               # will not be damped out in the smoothed solution.

VEL_CLS=200.0  # Characteristic length scale of the prior Gaussian
               # process for velocities. This is in kilometers.
VEL_STD=50.0   # Standard deviation of the prior Gaussian process for
               # velocities. This is in mm/yr.
VEL_ORDER=1    # Order of the polynomial null space. Setting this to 1
               # means that constant and linear trends in the data
               # will not be damped out in the smoothed solution.

# The data for each Plate Boundary Observatory (PBO) station can be
# downloaded from UNAVCO with the wget command. The URLs for several
# stations in Washington are saved in the file urls.txt. Use the Data
# Archive Interface (DAI) at www.unavco.org to find the urls for
# other PBO stations.
rm -rf work/csv
mkdir -p work/csv
for i in `cat urls.txt`
  do
  wget -P work/csv $i
  done

# use sed to concatenate all the data files and separate them with ***
sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

# convert the csv file to an hdf5 file
pygeons-toh5 work/data.csv --file-type pbocsv -vv

# crop out data prior to 2015-01-01 and after 2017-01-01
pygeons-crop work/data.h5 --start-date 2015-01-01 --stop-date 2017-01-01 -vv

# Temporally differentiate the displacement dataset
pygeons-tgpr work/data.crop.h5 $DISP_STD $DISP_CLS --order $DISP_ORDER --diff 1 -vv 

# Spatially differentiate the dataset
pygeons-sgpr work/data.crop.tgpr.h5 $VEL_STD $VEL_CLS --output-file work/xdiff.h5 \
             --order $VEL_ORDER --diff 1 0 -vv
pygeons-sgpr work/data.crop.tgpr.h5 $VEL_STD $VEL_CLS --output-file work/ydiff.h5 \
             --order $VEL_ORDER --diff 0 1 -vv

# Save the deformation gradients as text files
pygeons-totext work/xdiff.h5 -vv
pygeons-totext work/ydiff.h5 -vv

# view the estimated strain
pygeons-strain work/xdiff.h5 work/ydiff.h5 --scale 3.0e4 -vv
