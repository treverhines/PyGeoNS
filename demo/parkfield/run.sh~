#!/bin/bash
# coordinates of first endpoint of dislocation
lons="-120.566 -120.357"
lats="36.0 35.8349"
conn="0-1"

pygeons-toh5 data.csv pbo_csv -o work/data.h5

pygeons-downsample work/data.h5 \
  --start_date 2004-07-01 --stop_date 2005-07-01 \
  --break_dates 2004-09-29 -vv

pygeons-tfilter work/data.downsample.h5 --cutoff 0.02 --diff 1 \
  --break_dates 2004-09-29 \
  --procs 6 -vv

pygeons-sfilter work/data.downsample.tfilter.h5 --cutoff 1e-4 \
  --break_lons $lons \
  --break_lats $lats \
  --break_conn $conn \
  --procs 6 -vv

pygeons-view work/data.downsample.tfilter.sfilter.h5 work/data.downsample.tfilter.h5 \
  --break_lons $lons \
  --break_lats $lats \
  --break_conn $conn \
  -vv

