
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow numpy backend
"""


import math
import typing
import string

import numpy as np
from numpy import (  # noqa: F401
    sqrt, pi, conj, transpose, minimum,
    arccos, exp, cos, sin, reshape, size,
    real, imag, matmul, absolute, trace, diag,
    einsum, outer, sum)


TL = np
"""'TensorLibrary'. The actual imported backend python package
"""


DEVICE = 'cpu'
"""Current device"""
# FIXME DOCME


CTYPE = np.complex128
"""The complex datatype used by the backend
"""


FTYPE = np.float64
"""Floating point datatype used by the backend
"""


TENSOR = np.ndarray
"""Datatype of the backend tensors.
"""


BKTensor = typing.Any
"""Type hint for backend tensors"""
# Just used for documentation right now. Type checking numpy arrays
# not really supported yet (Jan 2018)


TensorLike = typing.Any
"""Any python object that can be converted into a backend tensor
"""
# Only used for documentation currently. Type checking numpy arrays and
# similar things not really supported yet. (Jan 2018)


MAX_QUBITS = 32
"""
Maximum number of qubits supported by this backend. Numpy arrays can't
have more than 32 dimensions, which limits us to no more than 32 qubits.
Pytorch has a similar problem, leading to a maximum of 24 qubits
"""


EINSUM_SUBSCRIPTS = string.ascii_lowercase + string.ascii_uppercase
"""
A string of all characters that can be used in einsum subscripts in
sorted order
"""


def gpu_available()->bool:
    """Does the backend support GPU acceleration on current hardware?"""
    return False


def ccast(value: complex) -> TensorLike:
    """Cast value to complex tensor (if necessary)"""
    return value


def fcast(value: float) -> TensorLike:
    """Cast value to float tensor (if necessary)"""
    return value


def astensor(array: TensorLike) -> BKTensor:
    """Converts a numpy array to the backend's tensor object
    """
    array = np.asarray(array, dtype=CTYPE)
    return array


def astensorproduct(array: TensorLike) -> BKTensor:
    """Converts a numpy array to the backend's tensor object, and reshapes
    to [2]*N (So the number of elements must be a power of 2)
    """
    tensor = astensor(array)
    N = int(math.log2(size(tensor)))
    array = tensor.reshape([2]*N)
    return array


def evaluate(tensor: BKTensor) -> TensorLike:
    """:returns: the value of a tensor as an ordinary python object"""
    return tensor


def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    return len(tensor.shape)


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    # Note: Relying on fact that vdot flattens arrays
    return np.vdot(tensor0, tensor1)


def cis(theta: float) -> BKTensor:
    r""":returns: complex exponential

    .. math::
        \text{cis}(\theta) = \cos(\theta)+ i \sin(\theta) = \exp(i \theta)
    """
    return np.exp(theta*1.0j)


def set_random_seed(seed: int) -> None:
    """Reinitialize the random number generator"""
    np.random.seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    """Get item from tensor"""
    return tensor.__getitem__(key)


def productdiag(tensor: BKTensor) -> BKTensor:
    """Returns the matrix diagonal of the product tensor"""  # DOCME: Explain
    N = rank(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = np.diag(tensor)
    tensor = reshape(tensor, [2]*(N//2))
    return tensor


def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: typing.List[int]) -> BKTensor:
    r"""
    Generalization of matrix multiplication to product tensors.

    A state vector in product tensor representation has N dimension, one for
    each contravariant index, e.g. for 3-qubit states
    :math:`B^{b_0,b_1,b_2}`. An operator has K dimensions, K/2 for
    contravariant indices (e.g. ket components) and K/2 for covariant (bra)
    indices, e.g. :math:`A^{a_0,a_1}_{a_2,a_3}` for a 2-qubit gate. The given
    indices of A are contracted against B, replacing the given positions.

    E.g. ``tensormul(A, B, [0,2])`` is equivalent to

    .. math::

        C^{a_0,b_1,a_1} =\sum_{i_0,i_1} A^{a_0,a_1}_{i_0,i_1} B^{i_0,b_1,i_1}

    Args:
        tensor0: A tensor product representation of a gate
        tensor1: A tensor product representation of a gate or state
        indices: List of indices of tensor1 on which to act.
    Returns:
        Resultant state or gate tensor

    """

    # Note: This method is the critical computational core of QuantumFlow
    # We currently have two implementations, one that uses einsum, the other
    # using matrix multiplication
    #
    # numpy:
    #   einsum is much faster particularly for small numbers of qubits
    # tensorflow:
    #   Little different is performance, but einsum would restrict the
    #   maximum number of qubits to 26 (Because tensorflow only allows 26
    #   einsum subscripts at present]
    # torch:
    #   einsum is slower than matmul

    N = rank(tensor1)
    K = rank(tensor0) // 2
    assert K == len(indices)

    out = list(EINSUM_SUBSCRIPTS[0:N])
    left_in = list(EINSUM_SUBSCRIPTS[N:N+K])
    left_out = [out[idx] for idx in indices]
    right = list(EINSUM_SUBSCRIPTS[0:N])
    for idx, s in zip(indices, left_in):
        right[idx] = s

    subscripts = ''.join(left_out + left_in + [','] + right + ['->'] + out)
    # print('>>>', K, N, subscripts)

    tensor = einsum(subscripts, tensor0, tensor1)
    return tensor
