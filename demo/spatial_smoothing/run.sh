# PYGEONS DEMONSTRATION #1:
# SPATIALLY SMOOTHING SYNTHETIC CASCADIA DATA

# This script demonstrates how to spatially smooth GPS data.  We use
# the synthetic data saved in synthetic.csv. The underlying signal in
# the synthetic data consists of sinusoids with a 400km wavelength and
# we obscure this signal with white noise.

# Convert the csv file to and HDF5 file. The first and second
# positional arguments are the file name and file type respectively.
# Possible file types are csv, pbo_csv, and pbo_pos
pygeons-toh5 synthetic.csv csv -o work/synthetic.h5

# spatial cutoff frequency in meters^-1. The underlying signal in the
# synthetic data has a frequency of 2.5e-6.  If we pick a slightly
# larger cutoff frequency then the underlying signal will be kept in
# tact.
SPATIAL_CUTOFF=5e-6
pygeons-sfilter work/synthetic.h5 --cutoff $SPATIAL_CUTOFF

# Compare the synthetic data to the smoothed data
pygeons-view work/synthetic.h5 work/synthetic.sfilter.h5 --image_clim -15 15\
             --image_array_size 1000 --data_set_labels synthetic smoothed

# Compare the smoothed data to the underlying signal
pygeons-toh5 synthetic.nonoise.csv csv -o work/synthetic.nonoise.h5
pygeons-view work/synthetic.nonoise.h5 work/synthetic.sfilter.h5 --image_clim -15 15\
             --image_array_size 1000 --data_set_labels true smoothed






