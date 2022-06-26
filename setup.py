import setuptools
from setuptools import setup

## README
with open("README.md", "r") as fh:
    long_description = fh.read()

## Requirements
with open("requirements.txt", "r") as r:
    requirements = [i.strip() for i in r.readlines()]

## Run Setup
setup(
    name="semshift",
    version="0.0.1",
    author="Keith Harrigian",
    author_email="kharrigian@jhu.edu",
    description="Semantic Shift Detection and Semantically-informed Feature Selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kharrigian/semantic-shift-websci-2022",
    packages=setuptools.find_packages("./"),
    python_requires='>=3.7, <3.8',
    install_requires=requirements,
)