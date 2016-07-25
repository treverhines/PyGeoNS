#!/bin/bash
# coordinates of first endpoint of dislocation
lons1="-120.566"
lats1="36.0"
# coordinates of second endpoint of dislocation
lons2="-120.357"
lats2="35.8349"

pygeons-convert data.csv work/data.h5

pygeons-downsample work/data.h5 1 \
  --start_date 2004-07-01 --stop_date 2005-07-01 \
  --cut_dates 2004-09-29 -vv

pygeons-zero work/data.downsample.h5 2004-09-01 7

pygeons-smooth work/data.downsample.zero.h5 --time_scale 50.0 --length_scale 10000 \
  --cut_dates 2004-09-29 \
  --cut_endpoint1_lons $lons1 \
  --cut_endpoint1_lats $lats1 \
  --cut_endpoint2_lons $lons2 \
  --cut_endpoint2_lats $lats2 -vv

pygeons-view work/data.downsample.zero.smooth.h5 -vv

