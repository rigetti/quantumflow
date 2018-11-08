#!/usr/bin/env python

"""QuantumFlow setup"""

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='quantumflow',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow',
        'networkx',
        'pyquil>=2.0.0b1',
        'pillow',
        'cvxpy',
        'sympy'
    ],
    use_scm_version={'write_to': 'quantumflow/version.py'},
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    package_data={'': ['datasets/data/*.*']},

    description="QuantumFlow: A Quantum Algorithms Development Toolkit",
    long_description=long_description,
    license='Apache-2.0',
    maintainer="Gavin Crooks",
    maintainer_email="gavin@rigetti.com",
    url="https://github.com/rigetticomputing/quantumflow/",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent'
        ]
    )
