''' 
provides a unit conversion function
'''

def unit_conversion(units,time='day',space='m'):
  ''' 
  returns a factors that converts data with units of *units* (e.g.
  km/hr) to units that are in terms of *time* and *space*. 
  
  Example
  -------
  >>> unit_conversion('mm/yr',time='day',space='m')
      2.73785078713e-06

  >>> units_ustrain
  '''
  # replace carets with two asterisks
  units = units.replace('^','**')
  # converts to m
  to_m = {'mm':1e-3,
          'cm':1e-2,
          'm':1e0,
          'km':1e3}
  # converts to seconds
  to_s = {'s':1.0,
          'min':60.0,
          'hr':60.0*60.0,
          'day':24.0*60.0*60.0,
          'mjd':24.0*60.0*60.0,
          'yr':365.25*24.0*60.0*60.0}
  # converts to user-specified space units and time units
  conv = dict([(k,v/to_m[space]) for k,v in to_m.iteritems()] +
              [(k,v/to_s[time]) for k,v in to_s.iteritems()])
  out = eval(units,conv)
  return out
