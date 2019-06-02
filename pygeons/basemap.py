''' 
This module contains a function for defining an appropriate Basemap
for a dataset.
'''
import numpy as np
import cartopy

def make_basemap(lon, lat):
  ''' 
  Creates a transverse mercator projection which is centered about the 
  given positions. 

  Modifying this function should be sufficient to change the way that 
  all map projections are generated in PyGeoNS
  '''
  lon = np.asarray(lon)
  lat = np.asarray(lat)
  lon_center = lon.min() + 0.5*lon.ptp()
  lat_center = lat.min() + 0.5*lat.ptp()
  proj = cartopy.crs.TransverseMercator(
    central_longitude=lon_center,
    central_latitude=lat_center)
  return proj    

