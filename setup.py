import sys
from setuptools import setup, find_packages


assert sys.version_info >= (3, 6, 0), "NGBoost-Tuner requires Python 3.6+"

with open("README.md", "r") as fh:
    long_description = fh.read()

tune_requires = [
    "pandas>=1.0.3",
    "ngboost>=0.2.0",
    "hyperopt>=0.2.4",
    "lightgbm>=2.3.0",
]

setup(
    name="ngboost-tuner",
    version="0.0.2",
    author="Ryan Wolbeck",
    author_email="wolbeck.ryan@gmail.com",
    description="A CLI to tune NGBoost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryan-wolbeck/ngboost-tuner",
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[],
    extras_require={"tune": tune_requires,},
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={"console_scripts": ["ngboost_tuner=ngboost_tuner.__main__:main"]},
)
