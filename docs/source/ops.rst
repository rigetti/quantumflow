==========
Operations
==========

.. contents:: :local:
.. currentmodule:: quantumflow


QuantumFlow supports several different quantum operations that act upon either pure or mixed states (or both). The four main types are Gate, which represents the action of an operator (typically unitary) on a state; Channel, which represents the action of a superoperator on a state (used for mixed quantum-classical dynamics); Kruas, which represents a Channel as a collection of operators; and Circuit, which is a list of other operations that act in sequence. Circuits can contain any instance of the abstract quantum operation superclass, Operation, including other circuits.

We consider the elemental quantum operations, such as Gate, Channel, and Kraus, as immutable. (Although immutability is not enforced in general.) Transformations of these operations return new copies. On the other hand the composite operation Circuit is mutable. 


.. autoclass:: Operation
    :members:

