import setuptools

setuptools.setup(
    name="eensight",
    packages=setuptools.find_packages(exclude=["eensight_tests"]),
    install_requires=[
        "dagster==0.11.4",
        "dagit==0.11.4",
        "pytest",
    ],
)
