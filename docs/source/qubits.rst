
====================
Qubit Hilbert Spaces
====================

.. module:: quantumflow

.. contents:: :local:
.. currentmodule:: quantumflow


States, gates, and various other methods accept a list of qubits labels upon which the given State or Gate acts. A Qubit label can be any hashable python object, but typically an integer or string. e.g. `[0, 1, 2]`, or `['a', 'b', 'c']`. Note that some operations expect the qubits to be sortable, so don't mix different uncomparable data types.


.. autoclass:: QubitVector
    :members:

