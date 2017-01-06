import numpy as np
from mpl_toolkits.basemap import Basemap

def make_basemap(lon,lat,resolution=None):
  ''' 
  Creates a transverse mercator projection which is centered about the 
  given positions. 

  Modifying this function should be sufficient to change the way that 
  all map projections are generated in PyGeoNS
  '''
  lon = np.asarray(lon)
  lat = np.asarray(lat)
  lon_buff = max(0.1,lon.ptp()/20.0)
  lat_buff = max(0.1,lat.ptp()/20.0)
  llcrnrlon = min(lon) - lon_buff
  llcrnrlat = min(lat) - lat_buff
  urcrnrlon = max(lon) + lon_buff
  urcrnrlat = max(lat) + lat_buff
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 resolution=resolution,
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat)


