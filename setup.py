from setuptools import find_packages, setup


# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

setup(
    name="eensight",
    version="0.1",
    packages=find_packages(exclude=["tests"]),
    install_requires=requires
)
