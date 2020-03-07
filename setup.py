#!/usr/bin/env python
from setuptools import setup
import setuptools


with open('README.md') as file:
    long_description = file.read()

setup(
    name='lung_report',
    version='0.1',
    description='Predict COVID-19 and hightlight focal infection',
    long_description='Predict COVID-19 and hightlight focal infection',
    url='https://github.com/iiat-project',
    author='Felix Li',
    author_email='lilao@163.com',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='COVID-19, IIAT',
    packages=setuptools.find_packages(),
)


#pip install git+https://github.com/Flyfoxs/task_distribute@master