#!/bin/bash
# coordinates of first endpoint of dislocation
lons1="-120.566"
lats1="36.0"
# coordinates of second endpoint of dislocation
lons2="-120.357"
lats2="35.8349"

pygeons-zero data.csv 2005.0 0.01

#pygeons-view data.zero.h5

pygeons-downsample data.zero.h5 7 \
  --start_date 2004-07-01 --stop_date 2005-07-01 \
  --cut_times 2004.742

pygeons-smooth data.zero.downsample.h5 --time_scale 0.2 --length_scale 10000.0 \
  --cut_dates 2004-09-29 \
  --cut_endpoint1_lons $lons1 \
  --cut_endpoint1_lats $lats1 \
  --cut_endpoint2_lons $lons2 \
  --cut_endpoint2_lats $lats2 -vv 

pygeons-view data.zero.downsample.h5 data.zero.downsample.smooth.h5 \
  --cut_endpoint1_lons $lons1 \
  --cut_endpoint1_lats $lats1 \
  --cut_endpoint2_lons $lons2 \
  --cut_endpoint2_lats $lats2 -vv
