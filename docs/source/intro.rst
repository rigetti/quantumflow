===============
Getting Started
===============
 

Installation
############

It is easiest to install QuantumFlow's requirements using conda.

.. code-block:: console

	> git clone https://github.com/rigetti/quantumflow.git
	> cd quantumflow
	> conda install -c conda-forge --yes --file requirements.txt
	> pip install -e .
	> make docs

You can also install with pip. However some of the requirements are tricky to install (i.e. tensorflow & cvxpy), and (probably) not everything in QuantumFlow will work correctly.

.. code-block:: console

	> git clone https://github.com/rigetti/quantumflow.git
	> cd quantumflow
	> pip install -r requirements.txt
	> pip install -e .
	> make docs


Basic Operation
###############

The principle operations are creating states and then applying sequences of quantum gates

.. doctest::

	>>> import quantumflow as qf            # Import QuantumFlow
	>>> ket = qf.zero_state(1)              # Create a zero state with 1 qubit
	>>> qf.print_state(ket)                 # Print state amplitudes
	0 : (1+0j)
	1 : 0j

	>>> ket = qf.H(0).run(ket)              # Apply a Hadamard gate to the 0th qubit
	>>> qf.print_state(ket)
	0 : (0.7071067811865475+0j)
	1 : (0.7071067811865475+0j)


Create a Bell state

.. doctest::

	>>> ket = qf.zero_state(2)             # Create a zero state with 2 qubits
	>>> ket = qf.H(0).run(ket)
	>>> ket = qf.CNOT(0, 1).run(ket)       # Apply a cnot to qubits 0 and 1
	>>> qf.print_state(ket)
	00 : (0.7071067811865475+0j)
	01 : 0j
	10 : 0j
	11 : (0.7071067811865475+0j)

States
######

States contain a tensor object defined by QuantumFlow's tensor library backend. The
``qf.asarray()`` method converts backend tensors back to ordinary python objects.

.. doctest::

	>>> ket.tensor
	array([[0.70710678+0.j, 0.        +0.j],
	       [0.        +0.j, 0.70710678+0.j]])
	>>> ket.tensor.shape
	(2, 2)

Typically the state of N qubits would be represented by a complex vector of
length 2**N.  However, QuantumFlow takes advantage of the tensor product 
structure of N qubits, and instead represents an N qubit
state by a complex tensor of rank N and shape ([2]*N). (e.g. (2,2) for N=2,
or (2, 2, 2, 2) for N=4). Each rank of the tensor represents a different qubit.

We can also generate a new state directly from a vector of amplitudes. QuantumFlow
will take care of reshaping and converting the vector into a QuantumFlow tensor.
(As long as the number of elements is a power of 2).

.. doctest::

	>>> import numpy as np
	>>> ket = qf.State(np.array([1,0,0,1]))
	>>> ket = ket.normalize()
	>>> qf.print_state(ket)
	00 : (0.7071067811865475+0j)
	01 : 0j
	10 : 0j
	11 : (0.7071067811865475+0j)

Since we can only measure our quantum computer in the computational basis, the measurement hermitian
operator must be diagonal. We represent these measurements by arrays (or tensors) of shape ([2]*N).

.. doctest::

	>>> qf.asarray(ket.expectation(np.array([1,0,0,0])))   # Probability of being in 00 state
	0.4999999999999999

Values are returned as a backend Tensor object, which can be converted
to an ordinary python or numpy value with the ``qf.asarray(tensor)`` method. We can convert an 
array to a backend tensor explicitly if desired. But for ordinary operations
you should not need to interact with the backend directly.

.. doctest::

	>>> from quantumflow import backend as bk
	>>> tensor = bk.astensor(np.array([1,0,0,0]))

Gates
#####

A gate acting on K qubits is a unitary operator of shape (2**K, 2**K), which
QuantumFlow represents as a mixed tensor of shape ([2]*(2*K)). e.g. for 
2 qubits the gate tensor's shape is (2, 2), and for 4 qubits
the gate shape is (2, 2, 2, 2, 2, 2, 2, 2).

.. doctest::

	>>> qf.X().asoperator()
	array([[0.+0.j, 1.+0.j],
	       [1.+0.j, 0.+0.j]])


The speed critical core of QuantumFlow is the Gate.run() method, which applies the action of a
K-qubit gate to an N-qubit state. Rather than promoting the gate to the full
N-qubit state space (As discussed in the quil paper), we instead reshape the
state so that it is (essentially) a tensor product of K and N-K qubit spaces.
The necessary permutations and resizings of the state array can be succinctly
expressed with a few standard tensor methods thanks to the product
representation of states.


We can also apply the action of a gate upon another gate.

.. doctest::

	>>> gate0 = qf.CNOT(0, 1)
	>>> gate1 = qf.CNOT(0, 1)	
	>>> gate = gate1 @ gate0 				# A cnot followed by a cnot is the identity
	>>> op = gate.asoperator()
	>>> np.reshape(op, (4,4))
	array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
	       [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
	       [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
	       [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])

There are various other methods for manipulating, inspecting, and comparing gates and states.
For instance, we can calculate the gate angle (a measure of distance between two gates)
between the previous gate and the 2-qubit identity, proving that they are identical.

.. doctest::

	>>> qf.asarray(qf.gate_angle(qf.identity_gate(2), gate))
	0.0


Circuits
########

A QuantumFlow circuit is a sequence of gates.

.. doctest::

	>>> circ = qf.Circuit()             # Build a Bell state preparation circuit
	>>> circ += qf.H(0)                 # Apply a Hadamard gate to the 0th qubit
	>>> circ += qf.CNOT(0, 1)           # Apply a CNOT between qubits 0 and 1
	>>> ket = qf.zero_state([0, 1])     # Prepare initial state
	>>> ket = circ.run(ket)             # Run circuit
	>>> qf.print_state(ket)
	00 : (0.7071067811865475+0j)
	01 : 0j
	10 : 0j
	11 : (0.7071067811865475+0j)







