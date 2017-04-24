# This script demonstrates how to download GPS data from UNAVCO and
# then use PyGeoNS to calculate the time-dependent strain accumulated
# over a slow slip event.

## download data from the urls in *urls.txt*
#rm -rf 'work/csv'
#mkdir -p 'work/csv'
#for i in `cat 'urls.txt'`
#  do
#  wget -P 'work/csv' $i
#  done

## use sed to concatenate all the data files and separate them with ***
#sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

# convert the csv file to an hdf5 file
pygeons toh5 -v 'work/data.csv' --file-type 'pbocsv'

# crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop -v 'work/data.h5' \
             --start-date '2015-11-01' \
             --stop-date '2016-03-01' \

pygeons fit -v 'work/data.crop.h5' \
            --network-model 'bm-se' \
            --network-params 5.0 57370.0 1.0 \
            --station-model 'p0' 'p1'

pygeons vector-view 'work/data.crop.h5' 'work/data.crop.fit.h5'

#pygeons autoclean -vv 'work/data.crop.h5' \
#             --network-model 'p10' 'p11' 'se-se' \
#             --network-params east     5.0e0 5.0e-2 5.0e1 \
#                              north    5.0e0 5.0e-2 5.0e1 \
#                              vertical 5.0e0 5.0e-2 5.0e1 \
#
#pygeons strain -vv 'work/data.crop.autoclean.h5' \
#             --network-prior-model 'p10' 'p11' 'se-se' \
#             --network-prior-params east     5.0e0 5.0e-2 5.0e1 \
#                                    north    5.0e0 5.0e-2 5.0e1 \
#                                    vertical 5.0e0 5.0e-2 5.0e1 \
#
#pygeons strain-view 'work/data.crop.autoclean.strain.xdiff.h5' \
#                    'work/data.crop.autoclean.strain.ydiff.h5'
#
## 57370
#
## postseismic strain
#pygeons strain -vv 'work/data.crop.autoclean.h5' \
#             --network-prior-model 'ibm-se' 'ramp-se' \
#             --station-noise-model 'p0' 'p1' 'peri' 'stepi'\
#             --network-prior-params 10.0 1000.0 50.0
#                                    1.0 1000.0 50.0
#                                    1.0 0.01 50.0
#             --network-prior-params east     5.0e0 5.0e-2 5.0e1 \
#                                    north    5.0e0 5.0e-2 5.0e1 \
#                                    vertical 5.0e0 5.0e-2 5.0e1 \
