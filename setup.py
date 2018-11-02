#!/usr/bin/python

"""QuantumFlow setup"""

from setuptools import setup, find_packages

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
        'antlr4-python3-runtime',
        'cvxpy',
        'sympy'
    ],
    use_scm_version={'write_to': 'quantumflow/version.py'},
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    package_data={'': ['datasets/data/*.*']},
)
