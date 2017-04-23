# -*- coding: utf-8 -*-
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
'metavar':'STR [STR ...]',
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
Name of the input data file containing the x derivatives of a
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
Name of the input data file containing the y derivatives of a 
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
OUTPUT_STEM = {
'type':str,
'metavar':'STR',
'help':
''' 
Stem for the output filenames. This does not include extensions.
'''
}
#####################################################################
NO_DISPLAY = {
'action':'store_true',
'help':
''' 
If this flag is raised then the interactive cleaner will not open up.
The edits from *input_edits_file* will be applied and the output data
file still be generated.
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
Label for each dataset.
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
Stop date for the output dataset in YYYY-MM-DD. This defaults to the 
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
NETWORK_FIX = {
'type':int,
'nargs':'*',
'metavar':'INT',
'help': 
''' 
Indices of network hyperparameters that should be fixed at the initial
guess.
'''
}
STATION_FIX = {
'type':int,
'nargs':'*',
'metavar':'INT',
'help': 
''' 
Indices of station hyperparameters that should be fixed at the initial
guess.
'''
}
#####################################################################
NETWORK_PRIOR_MODEL = {
'type':str,
'metavar':'STR',
'nargs':'*',
'help': 
''' 
Strings specifying the network prior model. 
'''
}
#####################################################################
NETWORK_PRIOR_PARAMS = {
'type':str,
'metavar':'FLOAT',
'nargs':'*',
'help': 
''' 
Hyperparameters for the network prior model.
'''
}
#####################################################################
NETWORK_NOISE_MODEL = {
'type':str,
'metavar':'STR',
'nargs':'*',
'help': 
''' 
Strings specifying the network noise model.
''' 
}
#####################################################################
NETWORK_NOISE_PARAMS = {
'type':str,
'metavar':'FLOAT',
'nargs':'*',
'help': 
''' 
Hyperparameters for the network noise model.
'''
}
#####################################################################
STATION_NOISE_MODEL = {
'type':str,
'metavar':'STR',
'nargs':'*',
'help': 
''' 
Strings specifying the station noise model.
''' 
}
#####################################################################
STATION_NOISE_PARAMS = {
'type':str,
'metavar':'FLOAT',
'nargs':'*',
'help': 
''' 
Hyperparameters for the station noise model.
'''
}
#####################################################################
NETWORK_MODEL = {
'type':str,
'metavar':'STR',
'nargs':'*',
'help': 
''' 
Strings specifying the network model. 
''' 
}
#####################################################################
NETWORK_PARAMS = {
'type':str,
'metavar':'FLOAT',
'nargs':'*',
'help': 
''' 
Initial guess for the network model hyperparameters.
'''
}
#####################################################################
STATION_MODEL = {
'type':str,
'metavar':'STR',
'nargs':'*',
'help': 
''' 
Strings specifying the station model. 
''' 
}
#####################################################################
STATION_PARAMS = {
'type':str,
'metavar':'FLOAT',
'nargs':'*',
'help': 
''' 
Initial guess for the station model hyperparameters.
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

GLOSSARY = {
'input_text_file':INPUT_TEXT_FILE,
'input_file':INPUT_FILE,
'input_files':INPUT_FILES,
'output_stem':OUTPUT_STEM,
'input_edits_file':INPUT_EDITS_FILE,
'no_display':NO_DISPLAY,
'positions':POSITIONS,
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
'start_date':START_DATE,
'stop_date':STOP_DATE,
'stations':STATIONS,
'min_lat':MIN_LAT,
'max_lat':MAX_LAT,
'min_lon':MIN_LON,
'max_lon':MAX_LON,
'network_params':NETWORK_PARAMS,
'network_model':NETWORK_MODEL,
'station_params':STATION_PARAMS,
'station_model':STATION_MODEL,
'network_prior_params':NETWORK_PRIOR_PARAMS,
'network_prior_model':NETWORK_PRIOR_MODEL,
'network_noise_params':NETWORK_NOISE_PARAMS,
'network_noise_model':NETWORK_NOISE_MODEL,
'station_noise_params':STATION_NOISE_PARAMS,
'station_noise_model':STATION_NOISE_MODEL,
'network_fix':NETWORK_FIX,
'station_fix':STATION_FIX,
'outlier_tol':OUTLIER_TOL,
}
