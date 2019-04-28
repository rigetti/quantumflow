
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" QuantumFlow Standard Gates """

# Kudos: Many gate definitions adapted from Nick Rubin's reference-qvm

from math import sqrt, pi
import copy
import numpy as np

from . import backend as bk
from .qubits import Qubit
from .ops import Gate
from .gates import I


__all__ = ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'PHASE', 'RX', 'RY', 'RZ', 'CZ',
           'CNOT', 'SWAP', 'ISWAP', 'CPHASE00', 'CPHASE01', 'CPHASE10',
           'CPHASE', 'PSWAP', 'CCNOT', 'CSWAP',
           'RN', 'TX', 'TY', 'TZ', 'TH', 'ZYZ',
           'CAN', 'XX', 'YY', 'ZZ', 'PISWAP', 'EXCH',
           'CANONICAL',
           'S_H', 'T_H', 'STDGATES']


# Standard 1 qubit gates

class X(Gate):
    r"""
    A 1-qubit Pauli-X gate.

    .. math::
        X() &\equiv \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
     """
    def __init__(self, q0: Qubit = 0) -> None:
        qubits = [q0]
        super().__init__([[0, 1], [1, 0]], qubits)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian

    def __pow__(self, t: float) -> Gate:
        return TX(t, *self.qubits)


class Y(Gate):
    r"""
    A 1-qubit Pauli-Y gate.

    .. math::
        Y() &\equiv \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

    mnemonic: "Minus eye high".
    """
    def __init__(self, q0: Qubit = 0) -> None:
        qubits = [q0]
        super().__init__(np.asarray([[0, -1.0j], [1.0j, 0]]), qubits)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian

    def __pow__(self, t: float) -> Gate:
        return TY(t, *self.qubits)


class Z(Gate):
    r"""
    A 1-qubit Pauli-Z gate.

    .. math::
        Z() &\equiv \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        qubits = [q0]
        super().__init__([[1, 0], [0, -1.0]], qubits)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian

    def __pow__(self, t: float) -> Gate:
        return TZ(t, *self.qubits)


class H(Gate):
    r"""
    A 1-qubit Hadamard gate.

    .. math::
        H() \equiv \frac{1}{\sqrt{2}}
        \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        unitary = np.asarray([[1, 1], [1, -1]]) / sqrt(2)
        qubits = [q0]
        super().__init__(unitary, qubits)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian

    def __pow__(self, t: float) -> Gate:
        return TH(t, *self.qubits)


class S(Gate):
    r"""
    A 1-qubit phase S gate, equivalent to ``PHASE(pi/2)``. The square root
    of the Z gate (up to global phase). Also commonly denoted as the P gate.

    .. math::
        S() \equiv \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

    """
    def __init__(self, q0: Qubit = 0) -> None:
        qubits = [q0]
        super().__init__(np.asarray([[1.0, 0.0], [0.0, 1.0j]]), qubits)

    @property
    def H(self) -> Gate:
        return S_H(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return PHASE(pi / 2 * t, *self.qubits)


class T(Gate):
    r"""
    A 1-qubit T (pi/8) gate, equivalent to ``PHASE(pi/4)``. The forth root
    of the Z gate (up to global phase).

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.cis(pi / 4.0))]]
        qubits = [q0]
        super().__init__(unitary, qubits)

    @property
    def H(self) -> Gate:
        return T_H(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return PHASE(pi / 4 * t, *self.qubits)


class PHASE(Gate):
    r"""
    A 1-qubit parametric phase shift gate

    .. math::
        \text{PHASE}(\theta) \equiv \begin{pmatrix}
         1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0.0], [0.0, bk.cis(ctheta)]]
        qubits = [q0]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        theta = 2. * pi - theta % (2. * pi)
        return PHASE(theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta'] * t
        return PHASE(theta, *self.qubits)


class RX(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate.

    .. math::
        R_X(\theta) =   \begin{pmatrix}
                            \cos(\frac{\theta}{2}) & -i \sin(\theta/2) \\
                            -i \sin(\theta/2) & \cos(\theta/2)
                        \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2), -1.0j * bk.sin(ctheta / 2)],
                   [-1.0j * bk.sin(ctheta / 2), bk.cos(ctheta / 2)]]
        qubits = [q0]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        return RX(-theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta']
        return RX(theta * t, *self.qubits)


class RY(Gate):
    r"""A 1-qubit Pauli-Y parametric rotation gate

    .. math::
        R_Y(\theta) \equiv \begin{pmatrix}
        \cos(\theta / 2) & -\sin(\theta / 2)
        \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[bk.cos(ctheta / 2.0), -bk.sin(ctheta / 2.0)],
                   [bk.sin(ctheta / 2.0), bk.cos(ctheta / 2.0)]]
        qubits = [q0]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        return RY(-theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta']
        return RY(theta * t, *self.qubits)


class RZ(Gate):
    r"""A 1-qubit Pauli-X parametric rotation gate

    .. math::
        R_Z(\theta)\equiv   \begin{pmatrix}
                                \cos(\theta/2) - i \sin(\theta/2) & 0 \\
                                0 & \cos(\theta/2) + i \sin(\theta/2)
                            \end{pmatrix}

    Args:
        theta: Angle of rotation in Bloch sphere
    """
    def __init__(self, theta: float, q0: Qubit = 0) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[bk.exp(-ctheta * 0.5j), 0],
                   [0, bk.exp(ctheta * 0.5j)]]
        qubits = [q0]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        return RZ(-theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta']
        return RZ(theta * t, *self.qubits)


# Standard 2 qubit gates

class CZ(Gate):
    r"""A controlled-Z gate

    Equivalent to ``controlled_gate(Z())`` and locally equivalent to
    ``CAN(1/2,0,0)``

    .. math::
        \text{CZ}() = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                    0&0&1&0 \\ 0&0&0&-1 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]
        params = None
        qubits = [q0, q1]
        super().__init__(unitary, qubits, params)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian


class CNOT(Gate):
    r"""A controlled-not gate

    Equivalent to ``controlled_gate(X())``, and
    locally equivalent to ``CAN(1/2, 0, 0)``

     .. math::
        \text{CNOT}() \equiv \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        unitary = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]
        params = None
        qubits = [q0, q1]
        super().__init__(unitary, qubits, params)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian


class SWAP(Gate):
    r"""A 2-qubit swap gate

    Locally equivalent to ``CAN(1/2,1/2,1/2)``.

    .. math::
        \text{SWAP}() \equiv
            \begin{pmatrix}
            1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1
            \end{pmatrix}

    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        unitary = [[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]]
        params = None
        qubits = [q0, q1]
        super().__init__(unitary, qubits, params)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian


class ISWAP(Gate):
    r"""A 2-qubit iswap gate

    Locally equivalent to ``CAN(1/2,1/2,0)``.

    .. math::
        \text{ISWAP}() \equiv
        \begin{pmatrix} 1&0&0&0 \\ 0&0&i&0 \\ 0&i&0&0 \\ 0&0&0&1 \end{pmatrix}

    """
    def __init__(self, q0: Qubit = 0, q1: Qubit = 1) -> None:
        # Note: array wrapper is to work around an eager mode
        # (not not regular tensorflow) issue.
        # "Can't convert Python sequence with mixed types to Tensor."

        unitary = np.array([[1, 0, 0, 0],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [0, 0, 0, 1]])
        params = None
        qubits = [q0, q1]
        super().__init__(unitary, qubits, params)


class CPHASE00(Gate):
    r"""A 2-qubit 00 phase-shift gate

    .. math::
        \text{CPHASE00}(\theta) \equiv \text{diag}(e^{i \theta}, 1, 1, 1)
    """
    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[bk.exp(1j * ctheta), 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]]
        qubits = [q0, q1]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = - self.params['theta']
        return CPHASE00(theta, *self.qubits)


class CPHASE01(Gate):
    r"""A 2-qubit 01 phase-shift gate

    .. math::
        \text{CPHASE01}(\theta) \equiv \text{diag}(1, e^{i \theta}, 1, 1)
    """
    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, bk.exp(1j * ctheta), 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]]
        qubits = [q0, q1]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = - self.params['theta']
        return CPHASE01(theta, *self.qubits)


class CPHASE10(Gate):
    r"""A 2-qubit 10 phase-shift gate

    .. math::
        \text{CPHASE10}(\theta) \equiv \text{diag}(1, 1, e^{i \theta}, 1)
    """
    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, bk.exp(1j * ctheta), 0],
                   [0, 0, 0, 1.0]]
        qubits = [q0, q1]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = - self.params['theta']
        return CPHASE10(theta, *self.qubits)


class CPHASE(Gate):
    r"""A 2-qubit 11 phase-shift gate

    .. math::
        \text{CPHASE}(\theta) \equiv \text{diag}(1, 1, 1, e^{i \theta})
    """
    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, bk.exp(1j * ctheta)]]
        qubits = [q0, q1]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = - self.params['theta']
        return CPHASE(theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta'] * t
        return CPHASE(theta, *self.qubits)


class PSWAP(Gate):
    r"""A 2-qubit parametric-swap gate, as defined by Quil.
    Interpolates between SWAP (theta=0) and iSWAP (theta=pi/2).

    Locally equivalent to ``CAN(1/2, 1/2, 1/2 - theta/pi)``

    .. math::
        \text{PSWAP}(\theta) \equiv \begin{pmatrix} 1&0&0&0 \\
        0&0&e^{i\theta}&0 \\ 0&e^{i\theta}&0&0 \\ 0&0&0&1 \end{pmatrix}
    """
    def __init__(self, theta: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)

        unitary = [[[[1, 0], [0, 0]], [[0, 0], [bk.exp(ctheta * 1.0j), 0]]],
                   [[[0, bk.exp(ctheta * 1.0j)], [0, 0]], [[0, 0], [0, 1]]]]
        qubits = [q0, q1]
        super().__init__(unitary, qubits, dict(theta=theta))

    @property
    def H(self) -> Gate:
        theta = self.params['theta']
        theta = 2. * pi - theta % (2. * pi)
        return PSWAP(theta, *self.qubits)


class PISWAP(Gate):
    r"""A parametric iswap gate, generated from XY interaction.

    Locally equivalent to CAN(t,t,0), where t = theta / (2 * pi)

    .. math::
        \text{PISWAP}(\theta) \equiv
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(2\theta) & i \sin(2\theta) & 0 \\
                0 & i \sin(2\theta) & \cos(2\theta) & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    """
    def __init__(self, theta: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        ctheta = bk.ccast(theta)
        unitary = [[[[1, 0], [0, 0]],
                    [[0, bk.cos(2*ctheta)], [bk.sin(2*ctheta) * 1j, 0]]],
                   [[[0, bk.sin(2*ctheta) * 1j], [bk.cos(2*ctheta), 0]],
                    [[0, 0], [0, 1]]]]
        params = dict(theta=theta)
        super().__init__(unitary, [q0, q1], params)

    @property
    def H(self) -> Gate:
        theta = - self.params['theta']
        return PISWAP(theta, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta = self.params['theta'] * t
        return PISWAP(theta, *self.qubits)


# Standard 3 qubit gates

class CCNOT(Gate):
    r"""
    A 3-qubit Toffoli gate. A controlled, controlled-not.

    Equivalent to ``controlled_gate(cnot())``

    .. math::
        \text{CCNOT}() \equiv \begin{pmatrix}
                1& 0& 0& 0& 0& 0& 0& 0 \\
                0& 1& 0& 0& 0& 0& 0& 0 \\
                0& 0& 1& 0& 0& 0& 0& 0 \\
                0& 0& 0& 1& 0& 0& 0& 0 \\
                0& 0& 0& 0& 1& 0& 0& 0 \\
                0& 0& 0& 0& 0& 1& 0& 0 \\
                0& 0& 0& 0& 0& 0& 0& 1 \\
                0& 0& 0& 0& 0& 0& 1& 0
            \end{pmatrix}

    """
    def __init__(self,
                 q0: Qubit = 0,
                 q1: Qubit = 1,
                 q2: Qubit = 2) -> None:
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 0]]
        params = None
        qubits = [q0, q1, q2]
        super().__init__(unitary, qubits, params)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian


class CSWAP(Gate):
    r"""
    A 3-qubit Fredkin gate. A controlled swap.

    Equivalent to ``controlled_gate(swap())``

    .. math::
        \text{CSWAP}() \equiv \begin{pmatrix}
                1& 0& 0& 0& 0& 0& 0& 0 \\
                0& 1& 0& 0& 0& 0& 0& 0 \\
                0& 0& 1& 0& 0& 0& 0& 0 \\
                0& 0& 0& 1& 0& 0& 0& 0 \\
                0& 0& 0& 0& 1& 0& 0& 0 \\
                0& 0& 0& 0& 0& 0& 1& 0 \\
                0& 0& 0& 0& 0& 1& 0& 0 \\
                0& 0& 0& 0& 0& 0& 0& 1
            \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0,
                 q1: Qubit = 1, q2: Qubit = 2) -> None:
        unitary = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]]
        params = None
        qubits = [q0, q1, q2]
        super().__init__(unitary, qubits, params)

    @property
    def H(self) -> Gate:
        return copy.copy(self)  # Hermitian


# Other 1-qubit gates

class S_H(Gate):
    r"""
    The inverse of the 1-qubit phase S gate, equivalent to ``PHASE(-pi/2)``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

    """
    def __init__(self, q0: Qubit = 0) -> None:
        super().__init__(np.asarray([[1.0, 0.0], [0.0, -1.0j]]), [q0])

    @property
    def H(self) -> Gate:
        return S(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return PHASE(-pi / 2 * t, *self.qubits)


class T_H(Gate):
    r"""
    The inverse (complex conjugate) of the 1-qubit T (pi/8) gate, equivalent
    to ``PHASE(-pi/4)``.

    .. math::
        \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}
    """
    def __init__(self, q0: Qubit = 0) -> None:
        unitary = [[1.0, 0.0], [0.0, bk.ccast(bk.cis(-pi / 4.0))]]
        super().__init__(unitary, [q0])

    @property
    def H(self) -> Gate:
        return T(*self.qubits)

    def __pow__(self, t: float) -> Gate:
        return PHASE(-pi / 4 * t, *self.qubits)


class RN(Gate):
    r"""A 1-qubit rotation of angle theta about axis (nx, ny, nz)

    .. math::
        R_n(\theta) = \cos \frac{theta}{2} I - i \sin\frac{theta}{2}
            (n_x X+ n_y Y + n_z Z)

    Args:
        theta: Angle of rotation on Block sphere
        (nx, ny, nz): A three-dimensional real unit vector
    """

    def __init__(self,
                 theta: float,
                 nx: float,
                 ny: float,
                 nz: float,
                 q0: Qubit = 0) -> None:
        ctheta = bk.ccast(theta)

        cost = bk.cos(ctheta / 2)
        sint = bk.sin(ctheta / 2)
        unitary = [[cost - 1j * sint * nz, -1j * sint * nx - sint * ny],
                   [-1j * sint * nx + sint * ny, cost + 1j * sint * nz]]

        params = dict(theta=theta, nx=nx, ny=ny, nz=nz)
        super().__init__(unitary, [q0], params)

    @property
    def H(self) -> Gate:
        theta, nx, ny, nz = self.params.values()
        return RN(-theta, nx, ny, nz, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        theta, nx, ny, nz = self.params.values()
        return RN(t * theta, nx, ny, nz, *self.qubits)


class TX(Gate):
    r"""Powers of the 1-qubit Pauli-X gate.

    .. math::
        TX(t) = X^t = e^{i \pi t/2} R_X(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        t = t % 2
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2),
                    phase * -1.0j * bk.sin(ctheta / 2)],
                   [phase * -1.0j * bk.sin(ctheta / 2),
                    phase * bk.cos(ctheta / 2)]]
        super().__init__(unitary, [q0], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return TX(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return TX(t, *self.qubits)


class TY(Gate):
    r"""Powers of the 1-qubit Pauli-Y gate.

    The pseudo-Hadamard gate is TY(3/2), and its inverse is TY(1/2).

    .. math::
        TY(t) = Y^t = e^{i \pi t/2} R_Y(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere

    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        t = t % 2
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.cos(ctheta / 2.0),
                    phase * -bk.sin(ctheta / 2.0)],
                   [phase * bk.sin(ctheta / 2.0),
                    phase * bk.cos(ctheta / 2.0)]]
        # unitary = RY(pi*t).tensor * bk.exp(- 0.5j * t)
        qubits = [q0]
        super().__init__(unitary, qubits, dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return TY(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return TY(t, *self.qubits)


class TZ(Gate):
    r"""Powers of the 1-qubit Pauli-Z gate.

    .. math::
        TZ(t) = Z^t = e^{i \pi t/2} R_Z(\pi t)

    Args:
        t: Number of half turns (quarter cycles) on Block sphere
    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        t = t % 2
        ctheta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * ctheta)
        unitary = [[phase * bk.exp(-ctheta * 0.5j), 0],
                   [0, phase * bk.exp(ctheta * 0.5j)]]
        super().__init__(unitary, [q0], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return TZ(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return TZ(t, *self.qubits)


class TH(Gate):
    r"""
    Powers of the 1-qubit Hadamard gate.

    .. math::
        TH(t) = H^t = e^{i \pi t/2}
        \begin{pmatrix}
            \cos(\tfrac{t}{2}) + \tfrac{i}{\sqrt{2}}\sin(\tfrac{t}{2})) &
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) \\
            \tfrac{i}{\sqrt{2}} \sin(\tfrac{t}{2}) &
            \cos(\tfrac{t}{2}) -\tfrac{i}{\sqrt{2}} \sin(\frac{t}{2})
        \end{pmatrix}
    """
    def __init__(self, t: float, q0: Qubit = 0) -> None:
        theta = bk.ccast(pi * t)
        phase = bk.exp(0.5j * theta)

        unitary = [[phase * bk.cos(theta / 2)
                    - (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    -(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)],
                   [-(phase * 1.0j * bk.sin(theta / 2)) / sqrt(2),
                    phase * bk.cos(theta / 2)
                    + (phase * 1.0j * bk.sin(theta / 2)) / sqrt(2)]]
        super().__init__(unitary, [q0], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return TH(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return TH(t, *self.qubits)


class ZYZ(Gate):
    r"""A Z-Y-Z decomposition of one-qubit rotations in the Bloch sphere

    The ZYZ decomposition of one-qubit rotations is

    .. math::
        \text{ZYZ}(t_0, t_1, t_2)
            = Z^{t_2} Y^{t_1} Z^{t_0}

    This is the unitary group on a 2-dimensional complex vector space, SU(2).

    Ref: See Barenco et al (1995) section 4 (Warning: gates are defined as
    conjugate of what we now use?), or Eq 4.11 of Nielsen and Chuang.

    Args:
        t0: Parameter of first parametric Z gate.
            Number of half turns on Block sphere.
        t1: Parameter of parametric Y gate.
        t2: Parameter of second parametric Z gate.
    """
    def __init__(self, t0: float, t1: float,
                 t2: float, q0: Qubit = 0) -> None:
        ct0 = bk.ccast(pi * t0)
        ct1 = bk.ccast(pi * t1)
        ct2 = bk.ccast(pi * t2)
        ct3 = 0

        unitary = [[bk.cis(ct3 - 0.5 * ct2 - 0.5 * ct0) * bk.cos(0.5 * ct1),
                    -bk.cis(ct3 - 0.5 * ct2 + 0.5 * ct0) * bk.sin(0.5 * ct1)],
                   [bk.cis(ct3 + 0.5 * ct2 - 0.5 * ct0) * bk.sin(0.5 * ct1),
                    bk.cis(ct3 + 0.5 * ct2 + 0.5 * ct0) * bk.cos(0.5 * ct1)]]

        super().__init__(unitary, [q0], dict(t0=t0, t1=t1, t2=t2))

    @property
    def H(self) -> Gate:
        t0, t1, t2 = self.params.values()
        return ZYZ(-t2, -t1, -t0, *self.qubits)


# Other 2-qubit gates


# TODO: Add references and explanation
# DOCME: Comment on sign conventions.
class CAN(Gate):
    r"""A canonical 2-qubit gate

    The canonical decomposition of 2-qubits gates removes local 1-qubit
    rotations, and leaves only the non-local interactions.

    .. math::
        \text{CAN}(t_x, t_y, t_z) \equiv
            \exp\Big\{-i\frac{\pi}{2}(t_x X\otimes X
            + t_y Y\otimes Y + t_z Z\otimes Z)\Big\}

    """
    def __init__(self,
                 tx: float, ty: float, tz: float,
                 q0: Qubit = 0, q1: Qubit = 1) -> None:
        xx = XX(tx)
        yy = YY(ty)
        zz = ZZ(tz)

        gate = yy @ xx
        gate = zz @ gate
        unitary = gate.tensor
        super().__init__(unitary, [q0, q1], dict(tx=tx, ty=ty, tz=tz))

    @property
    def H(self) -> Gate:
        tx, ty, tz = self.params.values()
        return CAN(-tx, -ty, -tz, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        tx, ty, tz = self.params.values()
        return CAN(tx * t, ty * t, tz * t, *self.qubits)


# Backwards compatability
# TODO: Add deprecation warning
class CANONICAL(CAN):
    """Deprecated. Use class CAN instead"""
    pass


class XX(Gate):
    r"""A parametric 2-qubit gate generated from an XX interaction,

    Equivalent to ``CAN(t,0,0)``.

    XX(1/2) is the Mølmer-Sørensen gate.

    Ref: Sørensen, A. & Mølmer, K. Quantum computation with ions in thermal
    motion. Phys. Rev. Lett. 82, 1971–1974 (1999)

    Args:
        t:
    """
    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, -1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [-1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        super().__init__(unitary, [q0, q1], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return XX(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return XX(t, *self.qubits)


class YY(Gate):
    r"""A parametric 2-qubit gate generated from a YY interaction.

    Equivalent to ``CAN(0,t,0)``, and locally equivalent to
    ``CAN(t,0,0)``

    Args:
        t:
    """
    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        theta = bk.ccast(pi * t)
        unitary = [[bk.cos(theta / 2), 0, 0, 1.0j * bk.sin(theta / 2)],
                   [0, bk.cos(theta / 2), -1.0j * bk.sin(theta / 2), 0],
                   [0, -1.0j * bk.sin(theta / 2), bk.cos(theta / 2), 0],
                   [1.0j * bk.sin(theta / 2), 0, 0, bk.cos(theta / 2)]]
        super().__init__(unitary, [q0, q1], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return YY(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return YY(t, *self.qubits)


class ZZ(Gate):
    r"""A parametric 2-qubit gate generated from a ZZ interaction.

    Equivalent to ``CAN(0,0,t)``, and locally equivalent to
    ``CAN(t,0,0)``

    Args:
        t:
    """
    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        theta = bk.ccast(pi * t)
        unitary = [[[[bk.cis(-theta / 2), 0], [0, 0]],
                    [[0, bk.cis(theta / 2)], [0, 0]]],
                   [[[0, 0], [bk.cis(theta / 2), 0]],
                    [[0, 0], [0, bk.cis(-theta / 2)]]]]
        super().__init__(unitary, [q0, q1], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return ZZ(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return ZZ(t, *self.qubits)


class EXCH(Gate):
    r"""A 2-qubit parametric gate generated from an exchange interaction.

    Equivalent to CAN(t,t,t)

    """
    def __init__(self, t: float, q0: Qubit = 0, q1: Qubit = 1) -> None:
        unitary = CAN(t, t, t).tensor
        super().__init__(unitary, [q0, q1], dict(t=t))

    @property
    def H(self) -> Gate:
        t = - self.params['t']
        return EXCH(t, *self.qubits)

    def __pow__(self, t: float) -> Gate:
        t = self.params['t'] * t
        return EXCH(t, *self.qubits)


# TODO: LATEX_OPERATIONS, QUIL_GATESET
GATESET = frozenset([I, X, Y, Z, H, S, T, PHASE, RX, RY, RZ, CZ,
                     CNOT, SWAP, ISWAP, CPHASE00, CPHASE01, CPHASE10,
                     CPHASE, PSWAP, CCNOT, CSWAP, PISWAP,
                     # Extras
                     RN, TX, TY, TZ, TH, ZYZ,
                     CAN, XX, YY, ZZ, EXCH,
                     S_H, T_H])

# TODO: Rename STDGATES to NAME_GATE?
STDGATES = {gate_class.__name__: gate_class for gate_class in GATESET}
