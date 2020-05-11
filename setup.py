import os

from setuptools import setup, find_packages


NAME = "empyricalRMT"
REQUIREMENTS = [
    "colorama",
    "EMD-signal",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "progressbar",
    "pyod",
    "scipy",
    "seaborn",
]


def read_file(path):  # type: ignore
    with open(os.path.join(os.path.dirname(__file__), path)) as fp:
        return fp.read()


setup(
    name=NAME,
    packages=find_packages(),
    author="Derek Berger",
    maintainer="Derek Berger",
    author_email="dmberger.dev@gmail.com",
    version="0.4.1",
    description="Eigenvalue unfolding and spectral observable computation",
    url="https://github.com/stfxecutables/empyricalRMT",
    license="MIT",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    install_requires=read_file("requirements.txt").split("\n"),
    python_requires=">=3.5",
    keywords="RMT RandomMatrixTheory spectral observables eigenvalues unfolding",  # noqa E501
)
