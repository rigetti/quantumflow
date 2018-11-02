=======
Backend
=======

.. module:: quantumflow.backend

.. contents:: :local:


Tensor Library Backends
#######################
QuantumFlow is designed to use a modern tensor library as a backend. 
The current options are tensorflow, eager, pytorch, and numpy (default).

- numpy (Default)  
    Python classic. Relatively fast on a single CPU, but no GPU
    acceleration, and no backprop.

- eager      
    Tensorflow eager mode. Fast with GPU. Very useful for debugging tensorflow
    code. GPU acceleration for simulation, but not optimization. 

- tensorflow 
    Tensorflow can automatically figure out back-propagated gradients,
    so we can efficiently optimize quantum networks using
    stochastic gradient descent. Tensorflow can use GPU accelerated for quantum
    simulation, but alas does not (currently) fully support complex numbers  
    for optimization.

- pytorch 
    Experimental prototype. Fast on CPU and GPU. Unfortunately stochastic gradient
    decent not available due to pytorch's lack of support for complex math.
    See the pytorch website for installation instrustions.



Configuration
#############

The default backend can be set in the configuration file, and can be overridden 
with the QUANTUMFLOW_BACKEND environment variable. e.g.  ::

	> QUANTUMFLOW_BACKEND=numpy pytest tests/test_flow.py

Options are tensorflow, eager, numpy, and torch.

You can also set the environment variable in python before quantumflow is imported.

    >>> import os
    >>> os.environ["QUANTUMFLOW_BACKEND"] = "numpy"
    >>> import quantumflow as qf


GPU
###

The tensorflow, eager, and torch backends will use available GPUs to accelerate
simulation. (The numpy backend does not have GPU acceleration.) Unfortunately,
none of these backends fully supports complex numbers combined with backpropagation,
so we cannot run stochastic gradient descent on GPUs at present.

Note that the main limiting factor is GPU memory. A single state uses 16 x 2^N bytes. 
We need to be able to place 2 states (and a bunch of smaller tensors) on a single GPU.
Thus a 16 GiB GPU can simulate a 28 qubit system.

The visible GPUs can be controlled with the CUDA_VISIBLE_DEVICES environment variable::

    > CUDA_VISIBLE_DEVICES=0    tools/benchmark.py 20    # Use first GPU
    > CUDA_VISIBLE_DEVICES=0,1  tools/benchmark.py 20    # Use first or second GPU
    > CUDA_VISIBLE_DEVICES=''   tools/benchmark.py 20    # Use CPU


Backend API
###########

.. autofunction:: quantumflow.backend.tensormul

Each backend is expected to implement the following methods, with semantics that match numpy.
(For instance, tensorflow's acos() method is adapted to match numpy's arccos())

- absolute
- arccos
- conj
- cos 
- diag
- exp 
- matmul
- minimum 
- real 
- reshape 
- sin 
- sum
- transpose


In addition each backend implements the following methods and variables.



.. automodule:: quantumflow.backend.numpybk
   :members:
