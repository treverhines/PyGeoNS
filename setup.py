#!/usr/bin/env python
if __name__ == '__main__':
  from numpy.distutils.core import setup
  from numpy.distutils.extension import Extension
  ext = []
  setup(name='PyGeoNS',
        version='0.1',
        description='Python-based Geodetic Network Smoother',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/PyGeoNS',
        packages=['pygeons'],
        license='MIT')


