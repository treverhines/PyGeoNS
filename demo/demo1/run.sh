mkdir -p work
pygeons toh5 data.csv \
        --output-stem work/data
pygeons toh5 soln.dx.csv \
        --output-stem work/soln.dx
pygeons toh5 soln.dy.csv \
        --output-stem work/soln.dy

#pygeons fit work/data.h5 \
#        --network-model spwen12-se \
#        --network-params 5.0 0.05 100.0 \
#        --station-model linear \
#        --station-params \
#        -vv
#
pygeons strain work/data.h5 \
        --network-prior-model spwen12-se \
        --network-prior-params 5.0 0.1 100.0 \
        --station-noise-model linear \
        --station-noise-params \
        -vv

