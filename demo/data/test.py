#!/usr/bin/env python
import pygeons.ioconv
import time as timemod
import numpy as np
import pygeons.interface
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
#files = open('data.csv','r').read().split('***')
#print(len(files))
data1 = pygeons.ioconv.dict_from_csv('out_data.csv')
#pygeons.ioconv.hdf5_from_dict('out_data.h5',data1)
#data2 = pygeons.ioconv.dict_from_hdf5('out_data.h5')

#pygeons.ioconv.csv_from_dict(data,'out2_data.csv')
pygeons.interface.view([data1])

#a = timemod.time()
#dicts = [pygeons.ioconv.parse_csv(f) for f in files]
#print(sum(len(d['time']) for d in dicts))
#print(timemod.time() - a)

#print(pygeons.ioconv.decyear_range('11-07-1989','11-10-1989',1,fmt='%m-%d-%Y'))
#print(np.diff(pygeons.ioconv.decyear_range('11-07-1989','11-10-1989',1,fmt='%m-%d-%Y')))
