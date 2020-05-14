#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ['pandas>=0.25.3']

setup_requirements = ['pytest-runner', ] + requirements

test_requirements = ['pytest>=3', ] + requirements

setup(
    author="gucky92",
    author_email='gucky@gucky.eu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Handling and transforming complex pandas.DatataFrame objects",
    install_requires=requirements,
    license="MIT license",
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='puffbird',
    name='puffbird',
    packages=find_packages(include=['puffbird', 'puffbird.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gucky92/puffbird',
    version='0.0.1',
    zip_safe=False,
)
