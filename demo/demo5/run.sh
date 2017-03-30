# Download sample data set
#rm -rf work/csv
#mkdir -p work/csv
#for i in `cat urls.txt`
#  do
#  wget -P work/csv $i
#  done
#
## use sed to concatenate all the data files and separate them with ***
#sed -s '$a***' work/csv/* | sed '$d' > work/data.csv
#

## convert the csv file to a HDF5 file
pygeons-toh5 work/data.csv --file-type pbocsv --output-file work/data.h5 -v
pygeons-crop work/data.h5 --start-date 2015-01-01 --stop-date 2017-01-01 --output-file work/data.h5 -v
#pygeons-treml work/data.h5 se 1.0 1.0 -vv --procs 2
pygeons-treml work/data.h5 fogm 1.0 1.0 -vv --procs 2



pygeons-view work/data.h5

#
## crop out data prior to 2015-01-01 and after 2017-01-01
#pygeons-crop work/data.h5 --start_date 2015-01-01 --stop_date 2017-01-01 -vv
#
## create a dataset that contains the prior expected value and
## uncertainty. If order > -1 then the expected value of the prior is
## set to the best fitting polynomial trend of the data.
#pygeons-tgpr work/data.crop.h5 $STD $CLS --do_not_condition \
#             --output_file work/prior.h5 --order $ORDER -vv
#
## generate a sample of the prior. Note that generating a sample can be
## expensive for large data sets.
#pygeons-tgpr work/data.crop.h5 $STD $CLS --do_not_condition --return_sample \
#             --output_file work/prior_sample.h5 --order $ORDER -vv
#
## Compare the data to the prior and the prior sample. Make sure that
## the prior confidence interval overlaps any potential signal in the
## data. Also make sure that the wavelength of the sample is comparable
## to the wavelength of any potential signal.
#pygeons-view work/data.crop.h5 work/prior.h5 work/prior_sample.h5 \
#             --data_set_labels 'data' 'prior' 'sample' \
#             --colors 'k' 'b' 'b' \
#             --line_styles 'None' 'solid' 'dashed' \
#             --line_markers '.' 'None' 'None'
#
## After verifying that the prior is appropriate, we can now condition
## the prior with the data
#pygeons-tgpr work/data.crop.h5 $STD $CLS \
#             --output_file work/posterior.h5 --order $ORDER -vv
#
## View the data and the posterior
#pygeons-view work/data.crop.h5 work/posterior.h5 \
#             --colors 'k' 'g' \
#             --data_set_labels 'data' 'posterior' \
#             --line_styles 'None' 'solid' \
#             --line_markers '.' 'None'
#
#

