"""setup.py -- note use setuptools==73.0.1; older versions fuck up the data files, newer versions include resources."""
from pathlib import Path
from setuptools import setup, find_packages

NAME = "microecs"
VERSION = "0.3.8"
DESCRIPTION = "MicroECS: Minimal Entity Component System (ECS) in python and numpy"
URL = "https://gitlab.com/meehai/microecs"

CWD = Path(__file__).absolute().parent
with open(CWD/"README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRED_CORE = [
    "numpy>=2.2.0",
    "loggez>=0.8",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED_CORE,
    extras_require={},
    dependency_links=[],
    license="MIT",
    python_requires=">=3.12",
    scripts=[], # cli/xxx in the future
)
