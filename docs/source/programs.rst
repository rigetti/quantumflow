========
Programs
========

.. contents:: :local:
.. currentmodule:: quantumflow

A QuantumFlow Program is an implementation of the Quantum Abstract Machine from
*A Practical Quantum Instruction Set Architecture*. [1]_ A Program can be built
programatically by appending Instuctions, or can be parsed from code written
in Rigetti's Quantum Instruction Language (Quil). This Quantum Virtual Machine
represents a hybrid device with quantum computation and classical control flow.

.. [1]  A Practical Quantum Instruction Set Architecture
        Robert S. Smith, Michael J. Curtis, William J. Zeng
        arXiv:1608.03355 https://arxiv.org/abs/1608.03355

.. doctest::

    import quantumflow as qf

    # Create an empty Program
    prog = qf.Program()

    # Measure qubit 0 and store result in classical bit (cbit) o
    prog += qf.Measure(0, 0)
    
    # Apply an X gate to qubit 0
    prog += qf.Call('X', params=[], qubits=[0])

    # Measure qubit 0 and store result in cbit 1.
    prog += qf.Measure(0, 1)

    # Compile and run program
    prog.run()

    # Contents of classical memory
    assert prog.memory == {0: 0, 1: 1}


Programs
#########
.. autoclass:: Instruction
    :members:

.. autoclass:: Program
    :members:
    
.. autoclass:: DefCircuit
.. autoclass:: Wait
.. autoclass:: Nop
.. autoclass:: Halt
.. autoclass:: Reset
.. autoclass:: And
.. autoclass:: Or
.. autoclass:: Move
.. autoclass:: Exchange
.. autoclass:: Not
.. autoclass:: Label
.. autoclass:: Jump
.. autoclass:: JumpWhen
.. autoclass:: JumpUnless
.. autoclass:: Pragma
.. autoclass:: Measure
.. autoclass:: Include
.. autoclass:: Call


