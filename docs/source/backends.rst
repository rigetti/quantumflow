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
    Tensorflow eager mode. Tensorflow can automatically figure out back-propagated
    gradients, so we can efficiently optimize quantum networks using
    stochastic gradient descent.

- tensorflow
    Regular tensorflow. Eager mode recommened.

- tensorflow2
    Tensorflow 2.x backend. Eager is now the default operation mode.

- torch (Experimental)
    Experimental prototype. Fast on CPU and GPU. Unfortunately stochastic gradient
    decent not available due to pytorch's lack of support for complex math.
    Pytorch is not installed by default. See the pytorch website for installation
    instructions.


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

Unfortunately, tensorflow does not fully supports complex numbers,
so we cannot run with eager or tensofrlow mode on GPUs at present.
The numpy backend does not have GPU acceleration either.

The torch backened can run with GPU acceleration, which can lead to
significant speed increase for simulation of large quantum states.
Note that the main limiting factor is GPU memory. A single state uses 16 x 2^N bytes.
We need to be able to place 2 states (and a bunch of smaller tensors) on a single GPU.
Thus a 16 GiB GPU can simulate a 28 qubit system.

    > QUANTUMFLOW_DEVICE=gpu QUANTUMFLOW_BACKEND=torch ./benchmark.py 24
    > QUANTUMFLOW_DEVICE=cpu QUANTUMFLOW_BACKEND=torch ./benchmark.py 24


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
