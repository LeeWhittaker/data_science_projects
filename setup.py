# Cython compile instructions

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

# Use python setup.py build --inplace
# to compile

__author__ = "Lee Whittaker"
__email__ = "leewhitt369@gmail.com"
__version__ = "0.1.0"
__status__ = "2 - Pre-Alpha"
__copyright__ = "Copyright 2021 Lee Whittaker"


setup_requires = ['setuptools', 'numpy']
install_requires = ['numpy', 'scipy', 'tensorflow', 'pytest', 'h5py']

setup(
    name='DataScienceProjects',
    version=__version__,
    classifiers=[
        #   1 - Planning
        #   2 - Pre-Alpha
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        #   6 - Mature
        #   7 - Inactive
        'Development Status :: ' + __status__,
        'Programming Language :: Python :: 3.6',
        ],
    description='Python package for data science projects',
    url='https://github.com/LeeWhittaker/data_science_projects',
    author=__author__,
    author_email=__email__,
    license=__copyright__,
    packages=['DataScienceProjects'],
    package_dir={'DataScienceProjects': 'python'},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires
)


    
