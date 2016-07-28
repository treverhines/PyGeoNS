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
        scripts=['exec/pygeons-tsmooth',
                 'exec/pygeons-ssmooth',
                 'exec/pygeons-tdiff',
                 'exec/pygeons-sdiff',
                 'exec/pygeons-clean',
                 'exec/pygeons-view',
                 'exec/pygeons-csvtoh5',
                 'exec/pygeons-postoh5',
                 'exec/pygeons-h5tocsv',
                 'exec/pygeons-zero',
                 'exec/pygeons-perturb',
                 'exec/pygeons-downsample'],
        packages=['pygeons'],
        license='MIT')


