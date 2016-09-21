''' 
Provides an input function which has a scope confined to this 
module and is unable to modify the scope
'''
from numpy import *
from matplotlib import *

def restricted_input(prompt):
  return input(prompt)
