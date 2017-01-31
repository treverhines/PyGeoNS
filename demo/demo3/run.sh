# This script demonstrates how to draw a prior sample from pygeons-gpr 
# and interpolate the solution onto a regular grid

# characteristic length scale of the prior Gaussian process
CLS=50.0
# standard deviation of the prior Gaussian process
STD=1.0

mkdir -p work

# convert the csv file to a HDF5 file
pygeons-toh5 data.csv --file_type csv --output_file work/data.h5

# make a csv file containing a regular grid of points
rm -rf pos.txt
touch pos.txt
for lon in `seq -87.0 0.15 -81.0`
  do
  for lat in `seq 41.0 0.15 47.0`
    do
    echo $lon $lat >> pos.txt
    done
  done


# the --sample_prior flag means that the the returned data set will 
# just be a sample of the prior, making the data in work/data.h5 
# irrelevant. The times and position in data.h5 are used to determine 
# where the prior samples will be evaluated. Note that prior Gaussian 
# process used by sgpr has no time correlation.
pygeons-sgpr work/data.h5 $STD $CLS --positions pos.txt --sample_prior -vv
#pygeons-tgpr work/data.h5 $STD $CLS --sample_prior -vv --start_date 2000-01-01 --stop_date 2001-01-01

pygeons-view work/data.sgpr.h5 -vv



