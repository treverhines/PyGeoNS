# This script demonstrates how to download GPS data from UNAVCO and
# then use PyGeoNS to calculate the time-dependent strain accumulated
# over a slow slip event.

## download data from the urls in *urls.txt*
rm -rf 'work/csv'
mkdir -p 'work/csv'
for i in `cat 'urls.txt'`
  do
  wget -P 'work/csv' $i
  done

## use sed to concatenate all the data files and separate them with ***
sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

# convert the csv file to an hdf5 file
pygeons toh5 -v 'work/data.csv' --file-type 'pbocsv'

# crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop -v 'work/data.h5' \
             --start-date '2015-11-01' \
             --stop-date '2016-03-01' \

pygeons autoclean -vv 'work/data.crop.h5' \
             --network-model 'p10' 'p11' 'se-se' \
             --network-params east     5.0e0 5.0e-2 5.0e1 \
                              north    5.0e0 5.0e-2 5.0e1 \
                              vertical 5.0e0 5.0e-2 5.0e1 \

pygeons strain -vv 'work/data.crop.autoclean.h5' \
             --network-prior-model 'p10' 'p11' 'se-se' \
             --network-prior-params east     5.0e0 5.0e-2 5.0e1 \
                                    north    5.0e0 5.0e-2 5.0e1 \
                                    vertical 5.0e0 5.0e-2 5.0e1 \

pygeons strain-view 'work/data.crop.autoclean.strain.xdiff.h5' \
                    'work/data.crop.autoclean.strain.ydiff.h5'
