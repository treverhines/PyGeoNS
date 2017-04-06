# This script demonstrates how to download GPS data from UNAVCO and
# then use PyGeoNS to calculate the time-dependent strain accumulated
# over a slow slip event.

# download data from the urls in *urls.txt*
#rm -rf 'work/csv'
#mkdir -p 'work/csv'
#for i in `cat 'urls.txt'`
#  do
#  wget -P 'work/csv' $i
#  done
#
## use sed to concatenate all the data files and separate them with ***
#sed -s '$a***' work/csv/* | sed '$d' > work/data.csv
#
# convert the csv file to an hdf5 file
pygeons toh5 'work/data.csv' --file-type 'pbocsv'

# crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop 'work/data.h5' \
             --start-date '2015-01-01' \
             --stop-date '2017-01-01' \
             --output-file 'work/data.h5'

# estimate the displacements resulting from the slow slip event. We
# assume that the surface displacements from the slow slip event can
# be described with integrated Brownian motion which starts at
# 2015-12-01
START=57357.0 # start date in MJD
pygeons tgpr 'work/data.h5' \
             'ibm' 50.0 $START \
             --noise-model 'linear+seasonal' \
             --output-file 'work/sse.h5'

pygeons sgpr 'work/sse.h5' \
             'linear+mat32' 1.0 200.0 \
             --output-file 'work/sse.smooth.h5'

pygeons view 'work/sse.h5' 'work/sse.smooth.h5'

## Temporally differentiate the displacement dataset
#
## Spatially differentiate the dataset
#pygeons sgpr work/data.crop.tgpr.h5 'linear+se' $VEL_STD $VEL_CLS --output-file work/xdiff.h5 \
#             --diff 1 0 -vv
#pygeons sgpr work/data.crop.tgpr.h5 'linear+se' $VEL_STD $VEL_CLS --output-file work/ydiff.h5 \
#             --diff 0 1 -vv
#
## Save the deformation gradients as text files
#pygeons totext work/xdiff.h5 -vv
#pygeons totext work/ydiff.h5 -vv
#
## view the estimated strain
#pygeons strain work/xdiff.h5 work/ydiff.h5 --scale 3.0e4 -vv
