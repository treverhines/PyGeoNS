# PYGEONS DEMONSTRATION #1:
# SPATIALLY SMOOTHING SYNTHETIC CASCADIA DATA

# make work directory
mkdir -p work

# This script demonstrates how to spatially smooth GPS data.  We use
# the synthetic data saved in synthetic.csv. The underlying signal in
# the synthetic data consists of sinusoids with a 400km wavelength and
# we obscure this signal with white noise.

# Convert the csv file to and HDF5 file. The first and second
# positional arguments are the file name and file type respectively.
# Possible file types are csv, pbo_csv, and pbo_pos
pygeons-toh5 synthetic.csv --file_type csv -o work/synthetic.h5 -vv

## spatial cutoff frequency in meters^-1. The underlying signal in the
## synthetic data has a frequency of 2.5e-6.  If we pick a slightly
## larger cutoff frequency then the underlying signal will be kept in
## tact.
#LENGTHSCALE=200
#STD=10.0
#pygeons-sgpr work/synthetic.h5 $STD $LENGTHSCALE -vv -o work/out1.h5
#STD=100.0
#pygeons-sgpr work/synthetic.h5 $STD $LENGTHSCALE -vv -o work/out2.h5
#STD=1000.0
#pygeons-sgpr work/synthetic.h5 $STD $LENGTHSCALE -vv -o work/out3.h5
#
## View original data set
##pygeons-view work/synthetic.h5 --data_set_labels synthetic smoothed
## View smoothed data set
#pygeons-view work/out1.h5 work/out2.h5 work/out3.h5 --data_set_labels 10 100 1000
#





