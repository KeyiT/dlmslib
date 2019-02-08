# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

test_requirements = [
    'numpy==1.14.5',
    'tensorflow==1.12.0',
    'keras==2.2.4',
    'torch=1.0.0'
]

setup(
    name="dlmslib",
    version="0.44",
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
    test_suite="tests",
    test_require=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
