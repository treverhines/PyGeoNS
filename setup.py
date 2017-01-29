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
        scripts=['exec/pygeons-tgpr',
                 'exec/pygeons-sgpr',
                 'exec/pygeons-tfilter',
                 'exec/pygeons-sfilter',
                 'exec/pygeons-clean',
                 'exec/pygeons-crop',
                 'exec/pygeons-view',
                 'exec/pygeons-strain',
                 'exec/pygeons-toh5',
                 'exec/pygeons-totext'],
        packages=['pygeons','pygeons.io','pygeons.plot',
                  'pygeons.clean'],
        license='MIT')


