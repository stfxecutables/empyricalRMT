import os

from distutils.core import setup


NAME = "empyricalRMT"


def read_file(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as fp:
        return fp.read()


setup(
    name=NAME,
    packages=[NAME],
    author="Derek Berger",
    maintainer="Derek Berger",
    author_email="dmberger.dev@gmail.com",
    version="0.1dev",
    description="Eigenvalue unfolding and spectral observable computation",
    url="https://github.com/stfxecutables/empyricalRMT",
    license="MIT",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
