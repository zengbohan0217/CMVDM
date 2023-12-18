#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='CMVDM-siluette-extraction',
    version='1.0.0',
    author="Bohan Zeng, Shanglin Li, et al.",
    author_email='shanglin@buaa.edu.cn',
    description="Controllable Mind Visual Diffusion Model (Official PyTorch implementation for siluette extraction)",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords="fMRI Image Reconstruction; Computer Vision; Generative Models",
    url='https://github.com/zengbohan0217/CMVDM',
    packages=find_packages(),
    include_package_data=True,
    tests_require=['pytest'],
    license="Yeda",
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    test_suite='tests',
)
