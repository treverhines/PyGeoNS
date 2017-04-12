# This script demonstrates how to use PyGeoNS to calculate postseismic
# velocities from a single displacement timeseries.

# Download the data for station P496, which is about 50km North of
# 2010 El Mayor-Cucapah epicenter.
rm -rf 'work/csv'
mkdir -p 'work/csv'
URL='ftp://data-out.unavco.org/pub/products/position/P496/P496.pbo.nam08.csv'
wget -P 'work/csv' $URL

# Convert CSV file to an HDF5 file
pygeons toh5 -v 'work/csv/P496.pbo.nam08.csv' --file-type 'pbocsv' \
             --output-file 'work/data.h5'

# crop out some time to make this demo a bit faster
pygeons crop -v 'work/data.h5' \
             --start-date '2007-01-01' \
             --stop-date  '2015-01-01' \
             --output-file 'work/data.h5'

# It is assumed that the displacement timeseries consists of:
#   - A linear polynomial (linear)
#   - Sinusoids with annual and semiannual periods (seasonal)
#   - A step at the time of the EMC earthquake (step)
#   - A linear postseismic trend (ramp)
#   - Integrated Brownian motion starting after the earthquake (ibm)
#
# We fit these terms to the observations with *pygeons tgpr*. This
# function takes at least three arguments; the name of the data file,
# a string specifying each component of the model, and a list of
# hyperparameters for the model. The components that require the user
# to specify a hyperparameter are *step*, *ramp*, and *ibm*. *step*
# and *ramp* require the user to specify their start times. *ibm*
# requires a scale parameter and a start time. These four
# hyperparameters are specified in the same order that the components
# were specified in. The start times are all the time of the EMC
# earthquake (in MJD), and the scale parameter for *ibm* is 50
# mm/yr^-1.5.
EMC=55291.0
pygeons tgpr -v 'work/data.h5' \
             'linear+seasonal+step+ramp+ibm'  $EMC  $EMC  50.0 $EMC \
             --output-file 'work/fit.h5'

# We now perform a similar fit but we treat *linear*, *seasonal*, and
# *step* as noise. We also take the time derivative of the fit.
pygeons tgpr -vv 'work/data.h5' \
             'ramp+ibm'  $EMC  50.0 $EMC \
             --noise-model 'linear+seasonal+step' --noise-params $EMC \
             --diff 1 \
             --output-file 'work/ps.h5'

# plot the results
pygeons view -v 'work/data.h5' 'work/fit.h5' \
             --dataset-labels 'raw disp.' 'fit disp.'

pygeons view -v 'work/ps.h5' \
             --dataset-labels 'postseismic vel.'

# write the postseismic velocities to a text file
pygeons totext -v 'work/ps.h5' \
               --output-file 'work/ps.csv'
