# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import sys

if sys.version_info < (3,6):
    sys.exit('Sorry, Python < 3.6 is not supported')

setup(
    name="dlmslib",
    version="0.1",
    author="keyi.tang",
    author_email="keyit92@gmail.com",
    packages=find_packages(),
    url="https://github.com/KeyiT/dlmslib",
    description="deep learning model architectures.",
    long_description=open("README.md").read(),
    zip_safe=True,
    include_package_data=True,
    package_data={},
    dependency_links=[],
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ]
)


