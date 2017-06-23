# -*- coding: utf-8 -*-
''' 
This modules defines command line arguments that are used by the
PyGeoNS executable. Each argument is a dictionary of key word
arguments that are passed to the *add_argument* method of an
*ArgumentParser*.
'''
#####################################################################
INPUT_TEXT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input text data file. 
'''
}
#####################################################################
INPUT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input HDF5 data file. 
'''
}
#####################################################################
INPUT_FILES = {
'nargs':'+',
'type':str,
'metavar':'STR [STR ...]',
'help':
''' 
Names of the input HDF5 data files.
'''
}
#####################################################################
XDIFF_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input HDF5 data file containing deformation east
derivative.
'''
}
#####################################################################
YDIFF_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the input HDF5 data file containing deformation north
derivative.
'''
}
#####################################################################
FILE_TYPE = {
'type':str,
'metavar':'STR',
'default':'csv',
'help':
''' 
Format for the input text file. This can either be 'csv', 'pbocsv', or
'pbopos'.
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
Do not display the PyGeoNS Interactive Cleaner. The edits from
*input-edits-file* will be applied and the output data file will still
be generated.
'''
}
#####################################################################
INPUT_EDITS_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Name of the file containing edits that will be applied to the dataset. 
'''
}
#####################################################################
POSITIONS_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
File containing output positions. Each line contains an ID, longitude,
and latitude. These are in addition to positions specified with
"positions". If no output positions are specified then the posterior
will be evaluated at each station in the input data file.
'''
}
#####################################################################
POSITIONS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Output positions specified as a sequence of IDs, longitudes, and
latitudes. For example "A000 -124.0 45.0 A001 -125.0 46.0"
'''
}
#####################################################################
VERBOSE = {
'action':'count',
'default':0,
'help':
''' 
Verbosity is set based on the number of times this flag is raised.
'''
}
#####################################################################
COLORS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Color string for each dataset (e.g. 'r', 'g', 'b').
'''
}
#####################################################################
LINE_STYLES = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Line style string for each dataset (e.g. '-', 'dashed', ':', 'None').
'''
}
#####################################################################
ERROR_STYLES = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Style of displaying the uncertainties for each dataset. This can be
'fill', 'bar', or 'None'.
'''
}
#####################################################################
LINE_MARKERS = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Marker string for each dataset (e.g. 'o', '.','p'). 
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
Sets the basemap resolution. This can be either 'c', 'i', or 'h' for
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
Number of vertices used in plotting the strain glyphs. 
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
Controls the length of the vectors.
'''
}
#####################################################################
QUIVER_KEY_LENGTH = {
'type':float,
'metavar':'FLOAT',
'help':
''' 
Length of the vector key.
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
Title of the time series figure.
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
Fontsize used in all figures.
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
Additional stations that should not be included in the output dataset.
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
NO_RATE = {
'dest':'rate',
'action':'store_false',
'help':
''' 
If this flag is raised then displacements and displacement gradients
will be returned instead of velocities and velocity gradients.
'''
}
#####################################################################
NO_SHOW_VERTICAL = {
'dest':'show_vertical',
'action':'store_false',
'help':
''' 
If this flag is raised then vertical components will not be shown.
'''
}
#####################################################################
NO_VERTICAL = {
'dest':'vertical',
'action':'store_false',
'help':
''' 
If this flag is raised then vertical deformation gradients will not be
computed.
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
'positions_file':POSITIONS_FILE,
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
'error_styles':ERROR_STYLES,
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
'no_rate':NO_RATE,
'no_vertical':NO_VERTICAL,
'no_show_vertical':NO_SHOW_VERTICAL,
'network_fix':NETWORK_FIX,
'station_fix':STATION_FIX,
'outlier_tol':OUTLIER_TOL,
}
