import setuptools

setuptools.setup(
    name="eensight",
    packages=setuptools.find_packages(exclude=["eensight_tests"]),
    install_requires=[
        "pytest",
    ],
)
