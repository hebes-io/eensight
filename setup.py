import os
import re
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

name = "eensight"
entry_point = "eensight = eensight.__main__:main"

here = Path(os.path.dirname(__file__)).resolve()

# get package version
with open(os.path.join(str(here), name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())
    if not result:
        raise ValueError("Can't find the version in eensight/__init__.py")
    version = result.group(1)


# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()


# get the dependencies and installs
with open("requirements.txt", encoding="utf-8") as f:
    # Make sure we strip all comments and options (e.g "--extra-index-url")
    # that arise from a modified pip.conf file that configure global options
    # when running kedro build-reqs
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)


data_files = [name for name in glob(str(here / name / "conf") + "/**/*.*", recursive=True)]

configuration = {
    "name": name,
    "version": version,
    "python_requires": ">=3.7",
    "description": (
        "A library for measurement and verification of energy efficiency projects."
    ),
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    "keywords": "measurement, verification, pipelines",
    "url": "https://github.com/hebes-io/eensight",
    "maintainer": "Sotiris Papadelis",
    "maintainer_email": "spapadel@gmail.com",
    "license": "Apache License, Version 2.0",
    "packages": find_packages(exclude=["tests"]),
    "entry_points": {"console_scripts": [entry_point]},
    "install_requires": requires,
    "include_package_data": True,
    "package_data": {name: data_files},
}

setup(**configuration)
