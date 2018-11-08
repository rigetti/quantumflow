>Notice: This is research code that will not necessarily be maintained to
>support further releases of Forest and other Rigetti Software. We welcome
>bug reports and PRs but make no guarantee about fixes or responses.

# QuantumFlow: A Quantum Algorithms Development Toolkit

[![Build Status](https://travis-ci.org/gecrooks/quantumflow.svg?branch=master)](https://travis-ci.org/gecrooks/quantumflow)

## Installation for development

It is easiest to install QuantumFlow's requirements using conda.
```
git clone https://github.com/rigetticomputing/quantumflow.git
cd quantumflow
conda install -c conda-forge --file requirements.txt
pip install -e .
```

You can also install with pip. However some of the requirements are tricky to install (notably tensorflow & cvxpy), and (probably) not everything in QuantumFlow will work correctly.
```
git clone https://github.com/rigetticomputing/quantumflow.git
cd quantumflow
pip install -r requirements.txt
pip install -e .
```

## Example
Train the QAOA algorithm, with back-propagation gradient descent, to perform
MAXCUT on a randomly chosen 6 node graph. 

```bash
./examples/qaoa_maxcut.py --verbose --steps 5 --nodes 6 random
```


