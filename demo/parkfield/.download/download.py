#!/usr/bin/env python
import urllib

f = open('url.txt','r')

for url in f:
  ref_i = url.split('.')[-2]
  agc = url.split('.')[-3]
  if ref_i != 'nam08':
    continue

  if agc != 'pbo':
    continue

  fid = url.strip().split('/')[-1]
  print(fid)
  try:
    buff = urllib.urlopen(url)
  except IOError:
    print('An error has occurred when trying to access %s' % url)
    continue

  pos_string = buff.read()
  buff.close()
  f = open('pos/' + fid,'w')
  f.write(pos_string)
  f.close()
  
