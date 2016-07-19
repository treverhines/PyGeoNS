#!/usr/bin/env python
import pygeons.decyear
import numpy as np
import unittest
import datetime

class Test(unittest.TestCase):
  def test_decyear(self):
    fmt = '%Y-%m-%d-%H-%M-%S'

    dt = datetime.datetime(2000,1,1,0,0)
    date = dt.strftime(fmt)
    time = pygeons.decyear.decyear(date,fmt)
    self.assertTrue(time==2000.0)        

    # test leap year
    dt = datetime.datetime(2000,1,1,0,0,0)
    dt += datetime.timedelta(366/2.0)
    date = dt.strftime(fmt)
    time = pygeons.decyear.decyear(date,fmt)
    self.assertTrue(np.isclose(time,2000.5,atol=1e-3))

    # test non leap year
    dt = datetime.datetime(2001,1,1,0,0,0)
    dt += datetime.timedelta(365/2.0)
    date = dt.strftime(fmt)
    time = pygeons.decyear.decyear(date,fmt)
    self.assertTrue(np.isclose(time,2001.5,atol=1e-3))

  def test_decyear_inv(self):
    # verify that decyear_inv is the inverse of decyear    
    fmt = '%Y-%m-%d-%H-%M-%S'
    np.random.seed(1)
    for i in range(100):
      time1 = np.random.uniform(1900,2100)
      date = pygeons.decyear.decyear_inv(time1,fmt)
      time2 = pygeons.decyear.decyear(date,fmt)
      self.assertTrue(np.isclose(time1,time2,atol=1e-4))
    

  def test_decyear_range(self):
    # verify that decyear range starts and ends on the indicated dates 
    # and has the right length.
    fmt = '%Y-%m-%d'
    date1 = '2000-01-01'
    date2 = '2001-01-01'
    times = pygeons.decyear.decyear_range(date1,date2,1,fmt)
    self.assertTrue(times[0] == 2000.0)
    self.assertTrue(times[-1] == 2001.0)
    self.assertTrue(len(times) == 367)

    date1 = '2001-01-01'
    date2 = '2002-01-01'
    times = pygeons.decyear.decyear_range(date1,date2,1,fmt)
    self.assertTrue(times[0] == 2001.0)
    self.assertTrue(times[-1] == 2002.0)
    self.assertTrue(len(times) == 366)

unittest.main()
