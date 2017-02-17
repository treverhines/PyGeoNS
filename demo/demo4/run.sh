# This script contains the commands from the demonstration in 
# README.rst
rm -r work
mkdir -p work/csv
for i in `cat urls.txt`; do wget -P work/csv $i; done
sed -s '$a***' work/csv/* | sed '$d' > work/data.csv

pygeons-toh5 work/data.csv \
             --file_type pbocsv \
             --output_file work/data.h5
pygeons-crop work/data.h5 \
             --start_date 2015-01-01 \
             --stop_date 2017-01-01 \
             --output_file work/data.h5

pygeons-info work/data.h5

pygeons-view work/data.h5

pygeons-tgpr work/data.h5 10.0 0.05 \
             --output_file work/smooth.h5 -vv

pygeons-view work/data.h5 work/smooth.h5

pygeons-tgpr work/data.h5 10.0 0.05 \
             --diff 1 \
             --output_file work/velocity.h5 -vv

pygeons-sgpr work/velocity.h5 100.0 150.0 \
             --diff 1 0 \
             --output_file work/xdiff.h5 -vv
pygeons-sgpr work/velocity.h5 100.0 150.0 \
             --diff 0 1 \
             --output_file work/ydiff.h5 -vv

pygeons-strain work/xdiff.h5 work/ydiff.h5 \
               --scale 1e4 \
               --key_magnitude 1.0 \
               --key_position 0.1 0.9

pygeons-totext work/xdiff.h5
pygeons-totext work/ydiff.h5

