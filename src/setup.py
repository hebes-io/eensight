import os
import re
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

name = "eensight"
entry_point = "eensight = eensight.__main__:main"

here = Path(os.path.dirname(__file__)).resolve()
docs = str(here.parent.joinpath("docs"))


# get package version
with open(os.path.join(str(here), name, "__init__.py"), encoding="utf-8") as f:
    result = re.search(r'__version__ = ["\']([^"\']+)', f.read())
    if not result:
        raise ValueError("Can't find the version in kedro/__init__.py")
    version = result.group(1)


# Get the long description from the README file
with open(os.path.join(docs, "source", "overview.md"), encoding="utf-8") as f:
    readme = f.read()


# get the dependencies and installs
with open("requirements.txt", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]


template_files = []
for pattern in ["**/*", "**/.*", "**/.*/**", "**/.*/.**"]:
    template_files.extend(
        [
            name.replace("eensight/", "", 1)
            for name in glob("eensight/templates/" + pattern, recursive=True)
        ]
    )


def dev_extras_require():
    extras = [
        "flake8 >= 3.9.2",
        "flake8-docstrings >= 1.6.0",
        "black >= 21.7b0",
        "pytest >= 6.2.5",
    ]
    return extras


def docs_extras_require():
    extras = [
        "Sphinx >= 3.0.0",  # Force RTD to use >= 3.0.0
        "docutils < 0.18",
        "nbsphinx >= 0.8.7",
        "pylons-sphinx-themes >= 1.0.8",  # Ethical Ads
        "myst-parse >= 0.15.2",
    ]
    return extras


configuration = {
    "name": name,
    "version": version,
    "python_requires": ">=3.7, <3.9",
    "description": (
        "A library for measurement and verification of energy efficiency projects."
    ),
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
    "packages": find_packages(),
    "entry_points": {"console_scripts": [entry_point]},
    "install_requires": requires,
    "include_package_data": True,
    "package_data": {name: template_files},
    "extras_require": {"dev": dev_extras_require(), "docs": docs_extras_require()},
}

setup(**configuration)
