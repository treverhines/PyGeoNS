#!/usr/bin/env python
import pygeons.downsample
import numpy as np
import matplotlib.pyplot as plt
import sympy
import unittest
np.random.seed(1)

class Test(unittest.TestCase):
  def test_weighted_mean(self):
    data = np.array([2.0,3.0,4.0])
    sigma = np.array([5.0,6.0,7.0])
    true_mean = np.sum(data/sigma**2)/np.sum(1.0/sigma**2)
    true_sigma = np.sqrt(1.0/np.sum(1.0/sigma**2))
    out = pygeons.downsample.weighted_mean(data,sigma)
    self.assertTrue(np.isclose(out[0],true_mean))
    self.assertTrue(np.isclose(out[1],true_sigma))

  def test_weighted_mean_shape(self):
    # for typical non-inf input values, weighted mean should function 
    # like np.average
    # make sure the functions work the same under different shaped 
    # arguments
    data = np.random.random((2,))
    sigma = np.random.random((2,))
    out1 = np.average(data,weights=1/sigma**2)
    out2 = pygeons.downsample.weighted_mean(data,sigma)[0]
    self.assertTrue(np.isclose(out1,out2))

    data = np.random.random((2,3))
    sigma = np.random.random((2,3))
    out1 = np.average(data,weights=1/sigma**2,axis=0)
    out2 = pygeons.downsample.weighted_mean(data,sigma,axis=0)[0]
    self.assertTrue(np.all(np.isclose(out1,out2)))

    data = np.random.random((2,3))
    sigma = np.random.random((2,3))
    out1 = np.average(data,weights=1/sigma**2,axis=1)
    out2 = pygeons.downsample.weighted_mean(data,sigma,axis=1)[0]
    self.assertTrue(np.all(np.isclose(out1,out2)))

  def test_weighted_mean_zero_length(self):
    # zero length along the axis should return nan mean and inf 
    # uncertanity
    data = np.zeros((0,))  
    sigma = np.zeros((0,))  
    out = pygeons.downsample.weighted_mean(data,sigma)
    self.assertTrue(np.isnan(out[0]))
    self.assertTrue(np.isinf(out[1]))

    data = np.zeros((0,2))  
    sigma = np.zeros((0,2))  
    out = pygeons.downsample.weighted_mean(data,sigma,axis=0)
    self.assertTrue(np.all(np.isnan(out[0])))
    self.assertTrue(np.all(np.isinf(out[1])))

    data = np.zeros((2,0))  
    sigma = np.zeros((2,0))  
    out = pygeons.downsample.weighted_mean(data,sigma,axis=1)
    self.assertTrue(np.all(np.isnan(out[0])))
    self.assertTrue(np.all(np.isinf(out[1])))

  def test_weighted_mean_inf_sigma(self):
    # make sure that values with inf sigma have zero weight
    data = np.array([3.0,4.0,5.0])
    sigma = np.array([1.0,1.0,np.inf])
    out = pygeons.downsample.weighted_mean(data,sigma)
    self.assertTrue(out[0]==3.5)
    self.assertTrue(out[1],np.sqrt(2)/2)

    data = np.array([3.0,4.0,5.0])
    sigma = np.array([np.inf,np.inf,1.0])
    out = pygeons.downsample.weighted_mean(data,sigma)
    self.assertTrue(out[0]==5.0)
    self.assertTrue(out[1],1.0)
    
  def test_mean_interp(self):
    x = np.array([1.0,2.0,3.0])[:,None]
    data = np.array([3.0,4.0,5.0])
    sigma = np.array([6.0,7.0,8.0])
    M = pygeons.downsample.MeanInterpolant(x,data,sigma=sigma)
    # if called with one interp. point then it should return the 
    # weighted mean
    out1_mean,out1_sigma = M([[0.0]])
    out2_mean,out2_sigma = pygeons.downsample.weighted_mean(data,sigma)
    self.assertTrue(out1_mean[0]==out2_mean)
    self.assertTrue(out1_sigma[0]==out2_sigma)

    # if called with the observation points then it should return the 
    # input values
    out_mean,out_sigma = M(x)
    self.assertTrue(np.all(np.isclose(out_mean,data)))
    self.assertTrue(np.all(np.isclose(out_sigma,sigma)))
    
unittest.main()
