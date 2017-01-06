''' 
This module defines a glossary of terms used in the help 
documentation. This is used to aid in building and maintaining the 
help documentation for each executable.
'''
from argparse import RawDescriptionHelpFormatter

#####################################################################
INPUT_FILE = {
'type':str,
'metavar':'STR',
'help':
''' 
Input file name. This must be an HDF5 file containing the fields 
described in the main help documentation.
'''
}
#####################################################################
FILE_TYPE = {
'type':str,
'metavar':'STR',
'default':'csv',
'help':
''' 
The format for the input file. This can either be "csv", "pbocsv", 
"pbopos", or "tdecsv". See the main documentation for a description of 
each format.
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
VERBOSE = {
'action':'count',
'default':0,
'help':
''' 
Controls verbosity.
'''
}
#####################################################################
COLOR_CYCLE = {
'nargs':'+',
'type':str,
'metavar':'STR',
'help':
''' 
Sets the colors of the data sets. 
'''
}
#####################################################################
RESOLUTION = {
'type':str,
'metavar':'STR',
'help':
''' 
Sets the basemap resolution. Can be either 'c', 'i', or 'h' for 
coarse, intermediate, or high resolution.
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
Length of the key vector. This should be in the units of the 
data set being plotted.
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
IMAGE_ARRAY_SIZE = {
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
GLOSSARY = {
'input_file':INPUT_FILE,
'output_file':OUTPUT_FILE,
'verbose':VERBOSE,
'file_type':FILE_TYPE,
'color_cycle':COLOR_CYCLE,
'quiver_scale':QUIVER_SCALE,
'quiver_key_length':QUIVER_KEY_LENGTH,
'quiver_key_pos':QUIVER_KEY_POS,
'scatter_size':SCATTER_SIZE,
'image_clim':IMAGE_CLIM,
'image_cmap':IMAGE_CMAP,
'image_array_size':IMAGE_ARRAY_SIZE,
'resolution':RESOLUTION,
'ts_title':TS_TITLE,
'map_title':MAP_TITLE,
'map_xlim':MAP_XLIM,
'map_ylim':MAP_YLIM,
'fontsize':FONTSIZE,
'break_lons':BREAK_LONS,
'break_lats':BREAK_LATS,
'break_conn':BREAK_CONN
}
