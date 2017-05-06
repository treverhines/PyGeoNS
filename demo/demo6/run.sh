#!/bin/bash
#PBS -A ehetland_flux
#PBS -M hinest@umich.edu
#PBS -N run
#PBS -m abe
#PBS -V
#PBS -j oe
#PBS -o run.log
#PBS -q flux
#PBS -l qos=flux
#PBS -l nodes=1:ppn=8,mem=32000mb,walltime=40:00:00
#cd $PBS_O_WORKDIR

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
#
## convert the csv file to an hdf5 file
pygeons toh5 -v 'work/data.csv' --file-type 'pbocsv'

## crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop -v 'work/data.h5' \
             --start-date '2015-10-01' \
             --stop-date '2016-03-01' \

pygeons strain -vv 'work/data.crop.h5' \
             --network-prior-model 'se-se' \
             --network-prior-params 1.0 0.03 100.0 \
             --station-noise-model 'p0' 'p1' 'per' 'fogm' \
             --station-noise-params 0.5 0.01 \

#pygeons strain-view -v 'work/data.crop.strain.dudx.h5' \
#                       'work/data.crop.strain.dudy.h5'
