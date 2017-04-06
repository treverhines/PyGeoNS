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
## convert the csv file to an hdf5 file
#pygeons toh5 'work/data.csv' --file-type 'pbocsv'
#
## crop out data prior to 2015-01-01 and after 2017-01-01
#pygeons crop 'work/data.h5' \
#             --start-date '2015-01-01' \
#             --stop-date '2017-01-01' \
#             --output-file 'work/data.h5'
#
# estimate the displacements resulting from the slow slip event. We
# assume that the surface displacements from the slow slip event can
# be described with integrated Brownian motion which starts at
# 2015-12-01
START=57357.0 # start date in MJD

# determine hyperparameters
#pygeons treml -vv 'work/data.h5' \
#             'linear+seasonal+ibm' 200 $START \
#             --fix 1 --procs 5 --parameters-file 'tparams.txt'

#pygeons tgpr -v 'work/data.h5' \
#             'ibm' 200.0 $START \
#             --noise-model 'linear+seasonal' \
#             --output-file 'work/sse.h5'

# crop out data prior to the start of the slow slip event
#pygeons crop 'work/sse.h5' \
#             --start-date 2015-12-02 \
#             --output-file 'work/sse.h5'

## compute the east derivative
pygeons sreml -vv 'work/sse.h5' \
              'linear+mat32' 100.0 100.0 \
              --parameters-file 'sparams.txt' \
              --procs 6 

#pygeons sgpr -v 'work/sse.h5' \
#             'linear+mat32' 100.0 100.0 \
#             --diff 1 0 \
#             --output-file 'work/sse.diffx.h5'
#
## compute the north derivative
#pygeons sgpr -v 'work/sse.h5' \
#             'linear+mat32' 100.0 100.0 \
#             --diff 0 1 \
#             --output-file 'work/sse.diffy.h5'
#
## view the strain
#pygeons strain 'work/sse.diffx.h5' 'work/sse.diffy.h5' \
#               --scale 5e5 \
#               --key-magnitude 0.05 \
#               --key-position 0.15 0.85
