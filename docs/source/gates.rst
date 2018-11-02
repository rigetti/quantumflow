
=====
Gates
=====

.. contents:: :local:
.. currentmodule:: quantumflow


Gate objects
############
.. autoclass:: Gate
    :members:

Actions on gates
#################
.. autofunction:: join_gates
.. autofunction:: control_gate
.. autofunction:: conditional_gate
.. autofunction:: almost_unitary
.. autofunction:: print_gate


Standard gates
##############

Standard gate set, as detailed in Quil whitepaper (arXiv:1608:03355v2)


Standard one-qubit gates
************************
.. autoclass:: I
.. autoclass:: X
.. autoclass:: Y
.. autoclass:: Z
.. autoclass:: H
.. autoclass:: S
.. autoclass:: T
.. autoclass:: PHASE
.. autoclass:: RX
.. autoclass:: RY
.. autoclass:: RZ


Standard two-qubit gates
************************
.. autoclass:: CZ
.. autoclass:: CNOT
.. autoclass:: SWAP
.. autoclass:: ISWAP
.. autoclass:: CPHASE00
.. autoclass:: CPHASE01
.. autoclass:: CPHASE10
.. autoclass:: CPHASE
.. autoclass:: PSWAP


Standard three-qubit gates
**************************
.. autoclass:: CCNOT
.. autoclass:: CSWAP



Additional gates
################


One-qubit gates
***************
.. autoclass:: S_H
.. autoclass:: T_H
.. autoclass:: TX
.. autoclass:: TY
.. autoclass:: TZ
.. autoclass:: TH
.. autoclass:: ZYZ
.. autoclass:: P0
.. autoclass:: P1



Two-qubit gates
***************
.. autoclass:: CANONICAL
.. autoclass:: XX
.. autoclass:: YY
.. autoclass:: ZZ
.. autoclass:: PISWAP
.. autoclass:: EXCH



Multi-qubit gates
*****************
.. autofunction:: identity_gate
.. autofunction:: random_gate

