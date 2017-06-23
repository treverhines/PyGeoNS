mkdir -p work

# convert the csv data file to an hdf5 file
pygeons toh5 data.csv \
        --output-stem work/data

# view the raw dataset
pygeons vector-view work/data.h5 \
                    --quiver-scale 0.0005 \
                    --quiver-key-length 10.0 \
                    --dataset-labels 'data' \
                    --no-show-vertical \
                    -vv

# calculate deformation gadients
pygeons strain work/data.h5 \
        --network-prior-model spwen12-se \
        --network-prior-params 1.0 0.1 100.0 \
        --station-noise-model linear \
        --station-noise-params \
        --positions A000 -83.74 42.28 \
                    A001 -83.08 42.33 \
                    A002 -83.33 41.94 \
                    A004 -83.36 42.16 \
        -vv


# compare the estimated east gradient to the true solution
pygeons toh5 soln.dx.csv \
        --output-stem work/soln.dudx

pygeons vector-view work/data.strain.dudx.h5 \
                    work/soln.dudx.h5 \
                    --quiver-scale 0.0001 \
                    --quiver-key-length 1.0 \
                    --dataset-labels 'east grad.' 'true soln.' \
                    --no-show-vertical \
                    -vv

# compare the estimated north gradient to the true solution
pygeons toh5 soln.dy.csv \
        --output-stem work/soln.dudy

pygeons vector-view work/data.strain.dudy.h5 \
                    work/soln.dudy.h5 \
                    --quiver-scale 0.0001 \
                    --quiver-key-length 1.0 \
                    --dataset-labels 'north grad.' 'true soln.' \
                    --no-show-vertical \
                    -vv

# view the estimated strain rates
pygeons strain-view work/data.strain.dudx.h5 \
                    work/data.strain.dudy.h5 \
                    --scale 20000.0 \
                    --key-magnitude 0.5 \
                    -vv
