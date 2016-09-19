# Bash commands to process the synthetic data
pygeons-csvtoh5 data/synthetic.csv -vv

# time smoothing
#pypygeons-tfilter data/synthetic.h5 --cutoff 0.01 --break_dates 2001-01-01 -vv

# view the data before and after filtering
#pygeons-view data/synthetic.h5 data/synthetic.tfilter.h5 -vv

# time smoothing and differentiation
#pygeons-tfilter data/synthetic.h5 --cutoff 0.01 --diff 1 --break_dates 2001-01-01 -vv --procs 3

# spatial x derivative
#pygeons-sfilter data/synthetic.h5 --diff 1 0 --cutoff 0.00001 \
#--break_lons1 -85.3530628568 -85.3530628568 -83.0353452111 \
#--break_lats1 42.9880621292 42.9880621292 43.6001605868 \
#--break_lons2 -83.0353452111 -87.3166 -81.5607 \
#--break_lats2 43.6001605868 42.3336 44.3666 -vv --procs 3 --output_file data/diffx.h5

# spatial y derivative
#pygeons-sfilter data/synthetic.h5 --diff 0 1 --cutoff 0.00001 \
#--break_lons1 -85.3530628568 -85.3530628568 -83.0353452111 \
#--break_lats1 42.9880621292 42.9880621292 43.6001605868 \
#--break_lons2 -83.0353452111 -87.3166 -81.5607 \
#--break_lats2 43.6001605868 42.3336 44.3666 -vv --procs 3 --output_file data/diffy.h5

# view the estimated velocities
# pygeons-view data/synthetic.tfilter.sfilter.h5 \
# --break_lons1 -85.3530628568 -85.3530628568 -83.0353452111 \
# --break_lats1 42.9880621292 42.9880621292 43.6001605868 \
# --break_lons2 -83.0353452111 -87.3166 -81.5607 \
# --break_lats2 43.6001605868 42.3336 44.3666 --quiver_scale 1e-7 -vv

pygeons-strain data/diffx.h5 data/diffy.h5 -vv --scale 200000000.0
