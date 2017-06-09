if __name__ == '__main__':
  from numpy.distutils.core import setup
  from numpy.distutils.extension import Extension
  from Cython.Build import cythonize
  ext = []
  ext += [Extension(name='pygeons.main.cbasis',sources=['pygeons/main/cbasis.pyx'])]
  setup(name='PyGeoNS',
        version='0.1',
        description='Python-based Geodetic Network Smoother',
        author='Trever Hines',
        author_email='treverhines@gmail.com',
        url='www.github.com/treverhines/PyGeoNS',
        scripts=['exec/pygeons'],
        packages=['pygeons','pygeons.io','pygeons.plot',
                  'pygeons.clean','pygeons.main'],
        ext_modules=cythonize(ext),                  
        license='MIT')


