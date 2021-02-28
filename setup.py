#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("eensight", "__init__.py")


install_requires = [
  "adtk",                    
  "catboost",                       
  "numpy",        
  "optuna",                 
  "pampy",                 
  "pandas",                                                
  "scikit-learn",                 
  "scipy",                          
  "statsmodels",                 
  "stumpy"                  
]


# Run the setup
setup(
    name="eensight",
    version=version,
    description="Python tools for automated M&V of energy efficiency improvements",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Sotiris Papadelis",
    url="https://senseih2020.eu/",
    author_email="spapadelis@hebes.io",
    license="MIT",
    classifiers=["Development Status :: 3 - Alpha", "Programming Language :: Python :: 3"],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "notebooks": ["dalex", "ipython", "ipywidgets", "matplotlib-base", "notebook", "patsy", "tslearn"],
    },
)     
