
===========
QuantumFlow
===========

QuantumFlow: A Quantum Algorithms Development Toolkit

The core of QuantumFlow is a simulation of a gate based quantum computer, which can run  
on top of modern optimized tensor libraries (numpy, tensorflow, or torch). The 
tensorflow backend can calculate the analytic gradient of a quantum circuit
with respect to the circuit's parameters, and circuits can be optimized to perform a function
using (stochastic) gradient descent. The torch and tensorflow backend can also accelerate the
quantum simulation using commodity classical GPUs.

Notice: This is research code that will not necessarily be maintained to support further
releases of Forest and other like `Rigetti <https://rigetti.com>`_ Software. We welcome bug reports
and PRs but make no guarantee about fixes or responses.

Please refer to the github repository https://github.com/rigetticomputing/quantumflow for source code, and to submit issues and pull requests. Documentation is hosted at readthedocs https://quantumflow.readthedocs.io/ .


.. toctree::
      :maxdepth: 3
      :caption: Contents:

      intro
      qubits
      states
      ops
      gates
      channels
      circuits
      measures
      programs
      forest   
      backends
      datasets
      examples


.. toctree::
      :hidden:

      devnotes

* :ref:`devnotes`
* :ref:`genindex`



