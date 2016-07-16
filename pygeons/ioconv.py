''' 
module for converting between the three input/output data formats
'''
import time as timemod
import datetime
import numpy as np
import matplotlib.pyplot as plt

def decyear_inv(decyear,fmt='%Y-%m-%dT%H:%M:%S'):
  ''' 
  convert decimal year to date string

  Note
  ----
    this function is the bottle neck for writing data
  '''
  year = int(np.floor(decyear))
  remainder = decyear - year
  year_start = datetime.datetime(year,1,1)
  year_end = datetime.datetime(year+1,1,1)
  days_in_year = (year_end - year_start).days
  decdays = remainder*days_in_year
  date = year_start + datetime.timedelta(days=decdays)
  return date.strftime(fmt)

def decyear(datestr,fmt='%Y-%m-%dT%H:%M:%S'):
  ''' 
  converts date string to decimal year

  Note
  ----
    this function is the bottle neck for reading data
  '''
  d = datetime.datetime.strptime(datestr,fmt)
  date_tuple = d.timetuple()
  # time in seconds of d
  time_in_sec = timemod.mktime(date_tuple)
  date_tuple = datetime.datetime(d.year,1,1,0,0).timetuple()
  # time in seconds of start of year
  time_year_start = timemod.mktime(date_tuple)
  date_tuple = datetime.datetime(d.year+1,1,1,0,0).timetuple()
  # time in seconds of start of next year
  time_year_end = timemod.mktime(date_tuple)
  decimal_time = (d.year + (time_in_sec - time_year_start)/
                           (time_year_end - time_year_start))

  return decimal_time

def get_line_with(sub,str):
  ''' 
  gets line with the first occurrence of sub
  '''
  idx = str.find(sub)
  if idx == -1:
    raise ValueError('Cannot find substring "%s"' % sub)

  #print(str[0:idx])
  line_start = str.rfind('\n',0,idx)
  if line_start == -1:
    # this is if sub is on the first line
    line_start = 0
  else:
    line_start += 1

  line_end = str.find('\n',line_start)
  if line_end == -1:
    # this is if sub is on the last line
    line_end = len(str)

  return str[line_start:line_end]

def get_field(field,str,delim=':'):
  ''' 
  finds the first line containing *field*, splits the line by *delim*, 
  then returns the list element which follows the one containing 
  *field*
  '''
  if delim in field:
    raise ValueError('Field "%s" contains the delimiter "%s"' % (field,delim))

  # first line containing field
  line = get_line_with(field,str)
  # split by delimiter
  lst = [i for i in line.split(delim)]
  # find the index containing field
  for i,j in enumerate(lst): 
    if field in j: 
      field_idx = i
      break

  # entry after the one containing field
  if (field_idx + 1) >= len(lst):
    raise ValueError(
      'No value associated with the field "%s". Make sure the '
      'correct delimiter is being used' % field)    

  out = lst[field_idx + 1]
  # remove white space
  out = out.strip()
  return out

def parse_pbo_csv(file_name):
  ''' 
  returns a dictionary with the following fields

    id
    longitude
    latitude
    time
    east
    north
    vertical 
    east_std
    north_std
    vertical_std

  '''
  str = file(file_name).read()
  fmt = '%Y-%m-%d'
  delim = ','

  # date_converter 
  def date_conv(str): 
    return decyear(str,fmt=fmt)

  # make everything lowercase so that field searches are not case 
  # sensitive
  str = str.lower()
  id = get_field('4-character id',str,delim=delim)
  start = get_field('begin date',str,delim=delim)
  end = get_field('end date',str,delim=delim)
  pos = get_line_with('reference position',str)
  lon,lat = pos.split()[5],pos.split()[2]

  data_start_idx = str.rfind(start)
  data = str[data_start_idx:]
  data = np.genfromtxt(data.split('\n'),
                       converters={0:date_conv},
                       delimiter=delim,
                       usecols=(0,1,2,3,4,5,6))
  output = {}
  output['id'] = id.upper()
  output['longitude'] = np.float(lon)
  output['latitude'] = np.float(lat)
  output['time'] = data[:,0]
  output['north'] = data[:,1]
  output['east'] = data[:,2]
  output['vertical'] = data[:,3]
  output['north_std'] = data[:,4]
  output['east_std'] = data[:,5]
  output['vertical_std'] = data[:,6]
  return output 

a = timemod.time()
out = parse_pbo_csv('/cmld/data2/hinest/Desktop/LVEG.pbo.nam08.csv')
plt.plot(out['time'],out['vertical'],'ko')
plt.show()
print(timemod.time() - a)
#first_date = get_field('Begin Date',str,delim=',')
#start_idx = str.rfind(first_date)
#data = str[start_idx:]
#print(data)
                  
