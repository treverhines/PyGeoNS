# This script demonstrates how to use PyGeoNS to calculate postseismic
# velocities from a single displacement timeseries.

# Download the data for station P496, which about 50km North of 2010
# El Mayor-Cucapah epicenter.
rm -rf 'work/csv'
mkdir -p 'work/csv'
URL='ftp://data-out.unavco.org/pub/products/position/P496/P496.pbo.nam08.csv'
wget -P 'work/csv' $URL

# Convert CSV file to an HDF5 file
pygeons toh5 'work/csv/P496.pbo.nam08.csv' --file-type 'pbocsv' \
             --output-file 'work/data.h5'

# date of the EMC earthquake in MJD
EMC=55291.0

# fit a linear trend, seasonal terms, a step, and integrated brownian
# motion to the timeseries
pygeons tgpr 'work/data.h5' 'linear+seasonal+step+ramp+ibm+fogm' $EMC $EMC 50.0 $EMC 1.5 0.1 \
             -vv --output-file 'work/fit.h5'

# perform the same fit but return the time derivative of the
# integrated brownian motion.
pygeons tgpr 'work/data.h5' 'ramp+ibm' $EMC 50.0 $EMC \
             --noise-model 'linear+seasonal+step' \
             --noise-params $EMC \
             --diff 1 \
             -vv --output-file 'work/ps.h5'

pygeons view 'work/data.h5' 'work/fit.h5' --dataset-labels 'raw disp.' 'fit disp.'
pygeons view 'work/ps.h5' --dataset-labels 'postseismic vel.'
