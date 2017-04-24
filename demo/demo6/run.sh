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
#
## use sed to concatenate all the data files and separate them with ***
#sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

# convert the csv file to an hdf5 file
pygeons toh5 -v 'work/data.csv' --file-type 'pbocsv'

# crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop -v 'work/data.h5' \
             --start-date '2015-11-01' \
             --stop-date '2016-03-01' \
             --output-stem 'work/data'

pygeons autoclean -vv 'work/data.h5' \
             --network-model 'p10' 'p11' 'se-se' \
             --network-params east     1.1e0 2.4e-2 4.4e1 \
                              north    1.0e0 4.6e-2 2.9e1 \
                              vertical 4.1e0 4.2e-4 9.9e9 \
             --station-model 'p0' \
             --station-params \
             --output-stem 'work/data'

pygeons strain -vv 'work/data.h5' \
             --network-prior-model 'p10' 'p11' 'se-se' \
             --network-prior-params east     1.1e0 2.4e-2 4.4e1 \
                                    north    1.0e0 4.6e-2 2.9e1 \
                                    vertical 4.1e0 4.2e-4 9.9e9 \
             --station-noise-model 'p0' \
             --station-noise-params \
             --output-stem 'work/data.noclean'

pygeons strain-view 'work/data.xdiff.h5' 'work/data.ydiff.h5'

#pygeons reml -vv 'work/data.edited.h5' \
#             --network-model 'p10' 'p11' 'se-se' \
#             --network-params 1.0 0.05 50.0 \
#             --station-model 'p0' \
#             --station-params \
#
#pygeons vector-view -v work/data.crop.h5 \
#                       work/data.edited.h5 \
#                       work/data.fit.h5 \
#                    --colors 'C0' 'C1' 'C2' \
#                    --line-marker '.' '.' ' '\
#                    --line-style 'none' 'none' '-'
###
## estimate the displacements resulting from the slow slip event. We
# assume that the surface displacements from the slow slip event can
# be described with integrated Brownian motion which starts at
# 2015-12-01
#START=57357.0 # start date in MJD

