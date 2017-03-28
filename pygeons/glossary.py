''' 
This module defines a glossary of terms used in the help 
documentation. This is used to aid in building and maintaining the 
help documentation for each executable.
'''

#####################################################################
INPUT_TEXT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input data file. This should be a text file containing the 
content of station files separated by three asterisks, ***. The 
station files can have the PBO csv format, the PBO pos format, or the 
PyGeoNS csv format. The format is specified with the *file_type* 
argument.
'''
}
#####################################################################
INPUT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input data file. This should be an HDF5 file.  
'''
}
#####################################################################
INPUT_FILES = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Names of the input data files. These should be HDF5 files. 
'''
}
#####################################################################
XDIFF_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input HDF5 file containing the x derivatives of a 
displacement or velocity field. This should be the output of 
pygeons-sgpr with the flag "--diff 1 0".
'''
}
#####################################################################
YDIFF_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input HDF5 file containing the y derivatives of a 
displacement or velocity field. This should be the output of 
pygeons-sgpr with the flag "--diff 0 1".
'''
}
#####################################################################
FILE_TYPE = {
'type':str,
'metavar':'STR',
'default':'csv',
'help':
''' 
The format for the station files. This can either be 'csv', 'pbocsv', 
'pbopos', or 'tdecsv'. See the README for a description of each 
format. Defaults to 'csv'.
'''
}
#####################################################################
OUTPUT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Output file name. If this is not specified then the output file name 
will be the input file name but with a new extension.
'''
}
#####################################################################
OUTPUT_EDITS_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the file where edits will be logged. Defaults to "edits.log".
'''
}
#####################################################################
INPUT_EDITS_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the file containing edits that will be applied to the dataset. 
This can be the name of the output edits file from a previous 
cleaning.
'''
}
#####################################################################
POSITIONS = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the file containing ids, latitudes, and longitudes of the 
output positions. If this is not specified then the output positions 
will be the same as the positions in the input dataset.
'''
}
#####################################################################
VERBOSE = {
'action':'count',
'default':0,
'help':
''' 
Controls verbosity. 
'''
}
#####################################################################
RETURN_SAMPLE = {
'action':'store_true',
'help':
''' 
If True, then the returned dataset will be a random sample of the
posterior, rather than its expected value and standard deviation.
'''
}
#####################################################################
COLORS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Color string for each dataset. This can be any valid matplotlib color 
string (e.g. 'r', 'g', or 'b' for red green or blue). There must be at 
least as many specified colors as there are datasets.
'''
}
#####################################################################
LINE_STYLES = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Line style string for each dataset. This can be almost any valid 
matplotlib line style string (e.g. '-', 'dashed', ':'). Note that 
'--' conflicts with the command line argument parsing utility and so 
'dashed' must be used instead. There must be at least as many 
specified line styles as there are datasets.
'''
}
#####################################################################
LINE_MARKERS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Marker string for each dataset. This can be any valid matplotlib 
marker string (e.g. 'o', '.','p'). There must be at least as many 
specified markers as there are datasets.
'''
}
#####################################################################
DATASET_LABELS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Label for each data sets.
'''
}
#####################################################################
MAP_RESOLUTION = {
'type':str,
'metavar':'STR',
'help':
''' 
Sets the basemap resolution. Can be either 'c', 'i', or 'h' for 
coarse, intermediate, or high resolution.
'''
}
#####################################################################
SNR_MASK = {
'action':'store_true',
'help':
''' 
If this flag is raised, then strain glyphs will only be plotted if 
their signal to noise ratio (SNR) is greater than 1.0. 
'''
}
#####################################################################
ALPHA = {
'type':float,
'metavar':'float',
'help':
''' 
Opacity of the strain confidence intervals. 0.0 is transparent and 1.0 
is opaque.
'''
}
#####################################################################
VERTICES = {
'type':int,
'metavar':'INT',
'help':
''' 
Number of vertices used in plotting the strain glyphs. Making this 
number lower will decrease the rendering time at the expense of lower 
quality glyphs.
'''
}
#####################################################################
COMPRESSION_COLOR = {
'type':str,
'metavar':'STR',
'help':
''' 
String indicating the color for the compressional strain.
'''
}
#####################################################################
EXTENSION_COLOR = {
'type':str,
'metavar':'STR',
'help':
''' 
String indicating the color for the extensional strain.
'''
}
#####################################################################
KEY_POSITION = {
'type':float,
'nargs':2,
'metavar':'FLOAT',
'help':
''' 
Sets the position of the key. This should be in axis coordinates.
'''
}
#####################################################################
KEY_MAGNITUDE = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Controls the magnitude of strain in the strain glyph. 
'''
}
#####################################################################
SCALE = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Controls size of the strain glyphs.
'''
}
#####################################################################
QUIVER_SCALE = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Controls the length of the vectors
'''
}
#####################################################################
QUIVER_KEY_LENGTH = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Length of the key vector.
'''
}
#####################################################################
QUIVER_KEY_POS = {
'nargs':2,
'type':float,
'metavar':'FLOAT',
'help':
''' 
Location of the vector key in axis coordinate.
'''
}
#####################################################################
SCATTER_SIZE = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Size of the scatter points showing vertical deformation.
'''
}
#####################################################################
IMAGE_CLIM = {
'nargs':2,
'type':float,
'metavar':'FLOAT',
'help':
''' 
Range of the colorbar showing vertical deformation.
'''
}
#####################################################################
IMAGE_CMAP = {
'type':str,
'metavar':'STR',
'help':
''' 
Colormap for the vertical deformation.
'''
}
#####################################################################
IMAGE_RESOLUTION = {
'type':int,
'metavar':'INT',
'help':
''' 
Size of the array used in plotting the vertical deformation. Larger 
numbers result in a higher resolution vertical field.
'''
}
#####################################################################
TS_TITLE = {
'type':str,
'metavar':'STR',
'help':
''' 
Title of the time series figure
'''
}
#####################################################################
MAP_TITLE = {
'type':str,
'metavar':'STR',
'help':
''' 
Title of the map figure.
'''
}
#####################################################################
MAP_XLIM = {
'nargs':2,
'type':float,
'metavar':'FLOAT',
'help':
''' 
x bounds of the map in meters.
'''
}
#####################################################################
MAP_YLIM = {
'nargs':2,
'type':float,
'metavar':'FLOAT',
'help':
''' 
y bounds of the map in meters.
'''
}
#####################################################################
FONTSIZE = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Fontsize used in all figures
'''
}
#####################################################################
BREAK_LONS = {
'nargs':'+',
'type':float,
'metavar':'FLOAT',
'help':
''' 
Longitude components of the vertices making up spatial 
discontinuities.
'''
}
#####################################################################
BREAK_LATS = {
'nargs':'+',
'type':float,
'metavar':'FLOAT',
'help':
''' 
Latitude components of the vertices making up spatial discontinuities.
'''
}
#####################################################################
BREAK_CONN = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Connectivity of the spatial discontinuity vertices. This can be one or 
multiple strings where each string contains the vertex indices making 
up each discontinuity separated by a '-'. For example, '0-1-2 3-4' 
indicates that there are two discontinuties, one contains vertices 0, 
1, and 2 and the other contains vertices 3 and 4.
'''
}
#####################################################################
BREAK_DATES = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Lists of dates with temporal discontinuities specified as YYYY-MM-DD. 
These dates should be when the discontinuity is first observed.
'''
}
#####################################################################
START_DATE = {
'type':str,
'metavar':'STR',
'help':
''' 
Start date for the output dataset in YYYY-MM-DD. This defaults to the 
start date for the input dataset.
'''
}
#####################################################################
STOP_DATE = {
'type':str,
'metavar':'STR',
'help':
''' 
Stop date for the output data set in YYYY-MM-DD. This defaults to the 
stop date for the input dataset.
'''
}
#####################################################################
STATIONS = {
'type':str, 
'nargs':'+',
'metavar':'STR', 
'help': 
''' 
List of stations that will not be included in the output dataset. This 
is in addition to the stations that will be excluded by the 
longitude/latitude bounds.
'''
}
#####################################################################
MIN_LAT = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Minimum latitude of stations in the ouput dataset.
'''
}
#####################################################################
MAX_LAT = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Maximum latitude of stations in the ouput dataset.
'''
}
#####################################################################
MIN_LON = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Minimum longitude of stations in the ouput dataset.
'''
}
#####################################################################
MAX_LON = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Maximum longitude of stations in the ouput dataset.
'''
}
#####################################################################
NO_ANNUAL = {
'action':'store_true',
'help':
''' 
If this flag is raised, then annual sinusoids will not be included in
the noise model.
'''
}
#####################################################################
NO_SEMIANNUAL = {
'action':'store_true',
'help':
''' 
If this flag is raised, then semi-annual sinusoids will not be
included in the noise model.
'''
}
#####################################################################
FOGM = {
'type':float,
'nargs':2,
'metavar':'FLOAT',
'help':
''' 
Hyperparameters for the FOGM noise model. The first hyperparameter is
the standard deviation of the white noise driving the process in units
of mm/yr^0.5. The second hyperparameter is the cutoff frequency in
units of 1/yr. Defaults to (0.5, 0.2).
'''
}
#####################################################################
S_CUTOFF = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Cutoff frequency in 1/meters.
'''
}
#####################################################################
T_CUTOFF = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Cutoff frequency in 1/days.
'''
}
#####################################################################
T_SIGMA = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
First hyperparameter for the prior model. This describes the standard
deviation of displacements in units of mm.
'''
}
#####################################################################
S_SIGMA = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
First hyperparameter for the prior model. This describes the standard
deviation of displacements (velocities) in units of mm (mm/yr).
'''
}
#####################################################################
T_CLS = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Second hyperparameter for the prior model. This describes the
characteristic time-scale of displacements in units of yr.
'''
}
#####################################################################
S_CLS = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Second hyperparameter for the prior model. This describes the
characteristic length-scale for displacements (velocities) in units of
km.
'''
}
#####################################################################
OUTLIER_TOL = {
'type':float, 
'metavar':'FLOAT', 
'help': 
''' 
Tolerance for outlier detection. Smaller values make the detection
algorithm more sensitive. This should not be any lower than about 2.0.
Defaults to 4.0.
'''
}
#####################################################################
ORDER = {
'default':1,
'type':int, 
'metavar':'INT', 
'help': 
''' 
Order of the polynomial basis functions. Defaults to 1.
'''
}
#####################################################################
S_DIFF = {
'nargs':2,
'type':int, 
'metavar':'INT', 
'help': 
''' 
Derivative order for each dimension. For example, setting *diff* to (1, 
0) computes the first derivative in the x direction. Defaults to (0, 0), 
meanings that no derivatives will be computed.
'''
}
#####################################################################
T_DIFF = {
'nargs':1,
'type':int, 
'metavar':'INT', 
'help': 
''' 
Derivative order. For example, setting *diff* to 1 computes the first
time derivative. Defaults to 0, meaning that no derivative will be
computed.
'''
}
#####################################################################
PROCS = {
'type':int,
'metavar':'INT',
'help':
''' 
Number of subprocesses to use. Defaults to 0, meaning that no
subprocesses will be spawned.
'''
}
#####################################################################
SAMPLES = {
'type':int,
'metavar':'INT',
'help':
''' 
Number of samples to use for estimating the uncertainty.
'''
}
#####################################################################
N = {
'type':int,
'metavar':'INT',
'help':
''' 
Stencil size.
'''
}
#####################################################################
FILL = {
'type':str,
'metavar':'STR',
'help':
''' 
Indicates how to handle missing data. either 'none', 'interpolate', or 
'extrapolate'
'''
}
#####################################################################
USE_PINV = {
'type':bool,
'metavar':'BOOL',
'help':
''' 
Indicate whether to use a pseudo-inversion to calculate the RBF-FD 
weights. This should be used when there are duplicate stations.
'''
}
#####################################################################
CHECK_ALL_EDGES = {
'type':bool,
'metavar':'BOOL',
'help':
''' 
Enforces that no stencil contains stations which form an edge that 
crosses the boundary.
'''
}
#####################################################################

GLOSSARY = {
'input_text_file':INPUT_TEXT_FILE,
'input_file':INPUT_FILE,
'input_files':INPUT_FILES,
'output_file':OUTPUT_FILE,
'input_edits_file':INPUT_EDITS_FILE,
'output_edits_file':OUTPUT_EDITS_FILE,
'positions':POSITIONS,
'return_sample':RETURN_SAMPLE,
'verbose':VERBOSE,
'file_type':FILE_TYPE,
'xdiff_file':XDIFF_FILE,
'ydiff_file':YDIFF_FILE,
'key_magnitude':KEY_MAGNITUDE,
'key_position':KEY_POSITION,
'compression_color':COMPRESSION_COLOR,
'extension_color':EXTENSION_COLOR,
'vertices':VERTICES,
'alpha':ALPHA,
'snr_mask':SNR_MASK,
'colors':COLORS,
'line_styles':LINE_STYLES,
'line_markers':LINE_MARKERS,
'dataset_labels':DATASET_LABELS,
'quiver_scale':QUIVER_SCALE,
'scale':SCALE,
'quiver_key_length':QUIVER_KEY_LENGTH,
'quiver_key_pos':QUIVER_KEY_POS,
'scatter_size':SCATTER_SIZE,
'image_clim':IMAGE_CLIM,
'image_cmap':IMAGE_CMAP,
'image_resolution':IMAGE_RESOLUTION,
'map_resolution':MAP_RESOLUTION,
'ts_title':TS_TITLE,
'map_title':MAP_TITLE,
'map_xlim':MAP_XLIM,
'map_ylim':MAP_YLIM,
'fontsize':FONTSIZE,
'break_lons':BREAK_LONS,
'break_lats':BREAK_LATS,
'break_conn':BREAK_CONN,
'break_dates':BREAK_DATES,
'start_date':START_DATE,
'stop_date':STOP_DATE,
'stations':STATIONS,
'min_lat':MIN_LAT,
'max_lat':MAX_LAT,
'min_lon':MIN_LON,
'max_lon':MAX_LON,
't_cutoff':T_CUTOFF,
's_cutoff':S_CUTOFF,
't_sigma':T_SIGMA,
's_sigma':S_SIGMA,
't_cls':T_CLS,
's_cls':S_CLS,
'outlier_tol':OUTLIER_TOL,
'fogm':FOGM,
'no_annual':NO_ANNUAL,
'no_semiannual':NO_SEMIANNUAL,
'order':ORDER,
't_diff':T_DIFF,
's_diff':S_DIFF,
'samples':SAMPLES,
'procs':PROCS,
'n':N,
'fill':FILL,
'use_pinv':USE_PINV,
'check_all_edges':CHECK_ALL_EDGES
}
