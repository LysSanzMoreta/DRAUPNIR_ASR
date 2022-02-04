#!/usr/bin/env python

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
from os.path import splitext
from os.path import basename
from glob import glob
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(name='draupnir',
      version='0.0.11',
      # list folders, not files
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/draupnir/*.py')],
      #scripts=['bin/script1.py'],
      package_data={'draupnir': ['data/data.txt']},
      description= 'Ancestral sequence reconstruction using a tree structured Ornstein Uhlenbeck variational autoencoder',
      long_description=long_description,
      classifiers=[  # Optional
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # Pick your license as you wish
            'License :: OSI Approved :: MIT License',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate you support Python 3. These classifiers are *not*
            # checked by 'pip install'. See instead 'python_requires' below.
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3 :: Only',
      ],
      python_requires='>=3.6, <4',
      project_urls={
            'Changelog': 'https://github.com/LysSanzMoreta/DRAUPNIR_ASR/blob/master/CHANGELOG.rst',
            'Issue Tracker': 'https://github.com/LysSanzMoreta/DRAUPNIR_ASR/issues',
      },

      )