# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import codecs
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt')) as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs[:-1] if 'git+' not in x]

def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(path.join(here, *parts), 'r').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='DEERPREdict',
    version=find_version('DEERPREdict', '__init__.py'),
    description='A package for DEER and PRE predictions based on molecular dynamics ensembles. Can be installed with pip.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/KULL-Centre/DEERpredict',
    license='GPLv3',
    classifiers=[
      'Environment :: Console',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Natural Language :: English',
      'Operating System :: OS Independent',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
      'Topic :: Scientific/Engineering :: Chemistry',
      'License :: OSI Approved :: GNU General Public License (GPL)',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Giulio Tesei, João M Martins, Micha BA Kunze, Ramon Crehuet, and Kresten Lindorff-Larsen',
    author_email='giulio.tesei@bio.ku.dk',
    install_requires=install_requires
)
