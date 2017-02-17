# This script runs through the examples in README.rst

## Download the GPS data and format it as an HDF5 data file.

# Download the station files from *urls.txt* and save them in a
#directory named *csv*.
#mkdir csv
#for i in `cat urls.txt`; do wget -P csv $i; done 
# combine station files into a single data file where each station
# file is separated by three asterisks
sed -s '$a***' csv/* | sed '$d' > data.csv
# Convert csv file to an HDF5 file
pygeons-toh5 data.csv --file_type pbocsv \
                      --output_file data.h5
# crop out data before 2015-01-01 and after 2017-01-01
pygeons-crop data.h5 --start_date 2015-01-01 \
                     --stop_date 2017-01-01 \
                     --output_file data.h5

## Temporally smooth displacements

STD='10.0' # prior standard deviation for displacements in millimeters
CLS='0.05' # prior characteristic time-scale for displacement in years
pygeons-tgpr data.h5 $STD $CLS --output_file smooth_disp.h5
# compare the observed and smoothed displacements
pygeons-view data.h5 smooth_disp.h5 --data_set_labels 'observed' 'smoothed' \
                                    --image_resolution 400 \
                                    --line_markers '.' 'None' \
                                    --line_style 'None' 'solid'

## Estimate and spatially smooth velocities

STD='10.0' # prior standard deviation for displacements in millimeters
CLS='0.05' # prior characteristic time-scale for displacements in years
# Temporally smooth and differentiate displacements to get velocities
pygeons-tgpr data.h5 $STD $CLS --diff 1 --output_file vel.h5
STD='100.0' # prior standard deviation for velocities in millimeters per year
CLS='150.0' # prior characteristic length-scale for velocities in kilometers
# Spatially smooth the velocities
pygeons-sgpr vel.h5 $STD $CLS --output_file smooth_vel.h5
# view the smoothed velocities
pygeons-view smooth_vel.h5 --data_set_labels 'velocities' \
                           --image_resolution 400


## Estimate strain rates

STD='10.0' # prior standard deviation for displacements in millimeters
CLS='0.05' # prior characteristic time-scale for displacements in years
# Temporally smooth and differentiate displacements to get velocities
pygeons-tgpr data.h5 $STD $CLS --diff 1 --output_file vel.h5
STD='100.0' # prior standard deviation for velocities in millimeters per year
CLS='150.0' # prior characteristic length-scale for velocities in kilometers
# Compute x derivative of velocities
pygeons-sgpr vel.h5 $STD $CLS --diff 1 0 --output_file xdiff.h5
# Compute y derivative of velocities
pygeons-sgpr vel.h5 $STD $CLS --diff 0 1 --output_file ydiff.h5

## view the strain rates
pygeons-strain xdiff.h5 ydiff.h5 --scale 1e4 \
                                 --key_magnitude 1.0 \
                                 --key_position 0.1 0.9


