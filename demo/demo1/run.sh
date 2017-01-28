# This script demonstrates using the PyGeoNS executables to calculate
# strain from four GPS time series, which are saved in data.csv. The
# strain solution is interpolated to a regular grid whos points are
# saved in pos.txt.

# Define the parameters describing the prior Gaussian processes used
# to temporally smooth and differentiate displacements and then
# spatially smooth and differentiate velocities.
DISP_CLS=0.1   # Characteristic time scale of the prior Gaussian
               # process for displacements. This is in years.
DISP_STD=5.0   # Standard deviation of the prior Gaussian process for
               # displacements. This is in mm.
DISP_ORDER=1   # Order of the polynomial null space. Setting this to 1
               # means that constant and linear trends in the data
               # will not be damped out in the smoothed solution.

VEL_CLS=200.0  # Characteristic length scale of the prior Gaussian
               # process for velocities. This is in kilometers.
VEL_STD=100.0  # Standard deviation of the prior Gaussian process for
               # velocities. This is in mm/yr.
VEL_ORDER=1    # Order of the polynomial null space. Setting this to 1
               # means that constant and linear trends in the data
               # will not be damped out in the smoothed solution.

POS_FILE="pos.txt" # text file containing the coordinates where strain
                   # is to be interpolated

# make a work directory to store intermediate files and move the csv
# file into it
python make_data.py
mkdir -p work
cp data.csv work/data.csv

# convert the csv file to a hdf5 file. The default output file name is
# the input file name but with the extension changed to .h5
pygeons-toh5 work/data.csv --file_type csv

# uncomment to view the input data set
#pygeons-view work/data.h5

# Temporally smooth and differentiate the data set using Gaussian
# process regression (GPR)
pygeons-tgpr work/data.h5 $DISP_STD $DISP_CLS --order $DISP_ORDER --diff 1

# Spatially smooth and differentiate the data set using GPR. This
# produces the deformation gradients at the interpolation points
pygeons-sgpr work/data.tgpr.h5 $VEL_STD $VEL_CLS --output_file work/xdiff.h5 \
             --order $VEL_ORDER --diff 1 0
pygeons-sgpr work/data.tgpr.h5 $VEL_STD $VEL_CLS --output_file work/ydiff.h5 \
             --order $VEL_ORDER --diff 0 1

# Save the deformation gradients as text files
pygeons-totext work/xdiff.h5
pygeons-totext work/ydiff.h5

# view the estimated strain
pygeons-strain work/xdiff.h5 work/ydiff.h5 --scale 3.0e4
