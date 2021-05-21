import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Packages required for this module to be executed
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

setuptools.setup(
    name="strappy",
    version="0.0.2",
    author="Greg Strabel",
    author_email="gregory.strabel@gmail.com",
    license="BSD 3",
    description="Python Utilities for Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Strabes/strappy",
    download_url = "https://github.com/Strabes/strappy/archive/refs/tags/v0.0.2.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires = list_reqs(),
    python_requires='>=3.6',
)