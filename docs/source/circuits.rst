========
Circuits
========

.. contents:: :local:
.. currentmodule:: quantumflow


Circuit objects
###############
.. autoclass:: Circuit
    :members:

.. autoclass:: DAGCircuit
    :members:


Standard circuits
#################

.. autofunction:: qft_circuit
.. autofunction:: reversal_circuit
.. autofunction:: control_circuit
.. autofunction:: ccnot_circuit
.. autofunction:: zyz_circuit
.. autofunction:: phase_estimation_circuit
.. autofunction:: addition_circuit
.. autofunction:: ghz_circuit


Gate decompositions
###################

.. autofunction:: bloch_decomposition
.. autofunction:: zyz_decomposition
.. autofunction:: kronecker_decomposition
.. autofunction:: canonical_decomposition
.. autofunction:: canonical_coords


Visualizations
##############

.. autofunction:: circuit_to_latex
.. autofunction:: render_latex


