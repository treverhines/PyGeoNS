# This script demonstrates how to download GPS data from UNAVCO and
# then use PyGeoNS to calculate the time-dependent strain rates during
# a slow slip event. This will take several minutes to run.

# download data from the urls in *urls.txt*
rm -rf 'work'
mkdir -p 'work/csv'
for i in `cat 'urls.txt'`
  do
  wget -P 'work/csv' $i
  done

# Use sed to concatenate all the data files and separate them with ***
sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

# Convert the csv file to an hdf5 file
pygeons toh5 'work/data.csv' \
             --file-type 'pbocsv' \
             -vv

# Crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop 'work/data.h5' \
             --start-date '2015-05-01' \
             --stop-date '2017-05-01' \
             -vv

# Remove outliers. Outliers are data points that are abnormally
# inconsistent with our prior. The spatio-temporal prior is a Gaussian
# process that is described spatially by a squared exponential (se)
# and temporally by a Wendland covariance function (spwen12). The
# prior also consists of an linear trend (lin) and seasonal terms
# (per) for each station
pygeons autoclean 'work/data.crop.h5' \
                  --network-model 'spwen12-se' \
                  --network-params 1.0 0.1 100.0 \
                  --station-model 'linear' 'per' \
                  --station-params \
                  --outlier-tol 4.0 \
                  -vv

# calculate deformation gradients from 2015-10-01 to 2016-04-01 using
# data from 2015-05-01 to 2017-05-01. Outputting over a wider range of
# time will increase run time. This function requires us to specify a
# prior model for transient deformation and a noise model. The prior
# model is the spatio-temporal Gaussian process described above. The
# noise model consists of the linear trend and seasonal terms in
# addition to the white noise that is specified in the data files.
pygeons strain 'work/data.crop.autoclean.h5' \
               --network-prior-model 'spwen12-se' \
               --network-prior-params 1.0 0.1 100.0 \
               --station-noise-model 'linear' 'per' \
               --station-noise-params \
               --start-date '2015-10-01' \
               --stop-date '2016-04-01' \
               -vv

# view the strain rates. Use the arrow keys to cycle through times and
# stations
pygeons strain-view 'work/data.crop.autoclean.strain.dudx.h5' \
                    'work/data.crop.autoclean.strain.dudy.h5' \
                    --scale 20000.0 \
                    --key-magnitude 1.0 \
                    --key-position 0.15 0.85 \
                    -vv

# Convert the deformation gradients from an hdf5 file to a
# user-friendly text file. The output files will be named
# work/dudx.csv and work/dudy.csv
pygeons totext 'work/data.crop.autoclean.strain.dudx.h5' \
        --output-stem 'work/dudx' \
        -vv
pygeons totext 'work/data.crop.autoclean.strain.dudy.h5' \
        --output-stem 'work/dudy' \
        -vv
