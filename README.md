>Notice: This is research code that will not necessarily be maintained to
>support further releases of Forest and other Rigetti Software. We welcome
>bug reports and PRs but make no guarantee about fixes or responses.

# QUANTUMFLOW: A Quantum Algorithms Development Toolkit

## Installation for development
```
git clone https://github.com/rigetticomputing/quantumflow.git
cd quantumflow
pip install -rrequirements.txt
pip install -e .
make docs
```

## Example
Train the QAOA algorithm, with back-propagation gradient descent, to perform
MAXCUT on a randomly chosen 6 node graph. 

```bash
./examples/qaoa_maxcut.py --verbose --steps 5 --nodes 6 random
```


