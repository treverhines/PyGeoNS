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
cd $PBS_O_WORKDIR

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

## convert the csv file to an hdf5 file
pygeons toh5 -v 'work/data.csv' --file-type 'pbocsv'

## crop out data prior to 2015-01-01 and after 2017-01-01
pygeons crop -v 'work/data.h5' \
             --start-date '2015-06-01' \
             --stop-date '2016-06-01' \

pygeons strain -vv 'work/data.crop.h5' \
             --network-prior-model 'ibm-se' \
             --network-prior-params east 1.0e2 57357.0 5.0e1 \
                                    north 1.0e2 57357.0 5.0e1 \
                                    vertical 1.0e2 57357.0 5.0e1 \
             --network-noise-model 'exp-p0' \
             --network-noise-params 0.5 0.001 \
             --station-noise-model 'p0' 'p1' \
             --station-noise-params \
             --output-stem 'work/comm.strain'

#pygeons strain-view -v 'work/data.crop.strain.dudx.h5' \
#                       'work/data.crop.strain.dudy.h5'
