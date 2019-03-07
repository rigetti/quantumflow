
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QuantumFlow pytorch backend

WARNING: EXPERIMENTAL

Regrettably we have to handle complex math manually, since pytorch
does not (yet) support complex tensors.
"""


import os
import math
from math import sqrt                                           # noqa: F401
import typing
import string

import numpy as np
from numpy import exp, cos, sin, minimum, arccos                # noqa: F401

import torch
import torch.cuda
from torch import sum                                           # noqa: F401


try:
    import torch.cuda.is_available as gpu_available
except ImportError:
    def gpu_available() -> bool:
        return False

from .numpybk import ccast, fcast                               # noqa: F401
from .numpybk import TensorLike, BKTensor
from .numpybk import astensor as np_astensor


TL = torch
name = TL.__name__
version = TL.__version__

CTYPE = np.complex128
FTYPE = np.float64

TENSOR = torch.tensor

DEFAULT_DEVICE = 'gpu' if gpu_available() else 'cpu'
DEVICE = os.getenv('QUANTUMFLOW_DEVICE', DEFAULT_DEVICE)
assert DEVICE in {'cpu', 'gpu'}

if DEVICE == 'cpu':
    _DTYPE = torch.DoubleTensor
else:
    _DTYPE = torch.cuda.DoubleTensor

# Maximum tensor dimensions in pytorch is only 24!
MAX_QUBITS = 24


EINSUM_SUBSCRIPTS = string.ascii_lowercase[2:]
# Torch's einsum suscripts must be lowercase letters
# We reserve 'a' for our own use when faking complex numbers


def astensor(array: TensorLike) -> BKTensor:
    if isinstance(array, _DTYPE):
        tensor = array
        return tensor

    array = np_astensor(array)
    array = np.stack((array.real, array.imag))
    tensor = torch.from_numpy(array)
    if DEVICE == 'gpu':
        tensor = tensor.cuda()

    assert isinstance(tensor, _DTYPE)

    return tensor


def astensorproduct(array: TensorLike) -> BKTensor:
    tensor = astensor(array)
    N = int(math.log2(tensor.numel()))
    tensor = torch.reshape(tensor, ([2]*N))
    return tensor


def evaluate(tensor: BKTensor) -> TensorLike:
    """Return the value of a tensor"""
    if isinstance(tensor, _DTYPE):
        if torch.numel(tensor) == 1:
            return tensor.item()
        if tensor.numel() == 2:
            return tensor[0].cpu().numpy() + 1.0j * tensor[1].cpu().numpy()

        return tensor[0].cpu().numpy() + 1.0j * tensor[1].cpu().numpy()
    return tensor


def real(tensor: BKTensor) -> BKTensor:
    tensor = tensor.clone()
    tensor[1] = 0.0
    return tensor


def imag(tensor: BKTensor) -> BKTensor:
    tensor = tensor.clone()
    tensor[0] = tensor[1]
    tensor[1] = 0.0
    return tensor


def size(tensor: BKTensor) -> int:
    return tensor.numel()//2


def absolute(tensor: BKTensor) -> BKTensor:
    if tensor.numel() == 2:
        return math.sqrt(tensor[0]**2 + tensor[1]**2)

    new_real = torch.sqrt(tensor[0]**2 + tensor[1]**2)
    new_imag = torch.zeros(*new_real.shape)
    new_imag = new_imag.type(_DTYPE)

    return torch.stack((new_real, new_imag))


def cis(theta: float) -> BKTensor:
    """ cis(theta) = cos(theta)+ i sin(theta) = exp(i theta) """
    return np.exp(theta*1.0j)


def rank(tensor: BKTensor) -> int:
    """Return the number of dimensions of a tensor"""
    if isinstance(tensor, np.ndarray):
        return len(tensor.shape)

    return len(tensor[0].size())


def inner(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    """Return the inner product between two tensors"""
    N = torch.numel(tensor0[0])
    tensor0_real = tensor0[0].contiguous().view(N)
    tensor0_imag = tensor0[1].contiguous().view(N)
    tensor1_real = tensor1[0].contiguous().view(N)
    tensor1_imag = tensor1[1].contiguous().view(N)

    res = (torch.matmul(tensor0_real, tensor1_real)
           + torch.matmul(tensor0_imag, tensor1_imag),
           torch.matmul(tensor0_real, tensor1_imag)
           - torch.matmul(tensor0_imag, tensor1_real))

    return _DTYPE(res)


def outer(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:

    tensor0 = tensor0.contiguous()
    tensor1 = tensor1.contiguous()

    bits = rank(tensor0) + rank(tensor1)
    num0 = torch.numel(tensor0[0])
    num1 = torch.numel(tensor1[0])
    res = (torch.ger(tensor0[0].view(num0), tensor1[0].view(num1))
           - torch.ger(tensor0[1].view(num0), tensor1[1].view(num1)),
           torch.ger(tensor0[0].view(num0), tensor1[1].view(num1))
           + torch.ger(tensor0[1].view(num0), tensor1[0].view(num1)))
    tensor = torch.stack(res)
    tensor = tensor.resize_([2]*(bits+1))

    return tensor


def conj(tensor: BKTensor) -> BKTensor:
    return torch.stack((tensor[0], - tensor[1]))


def transpose(tensor: BKTensor, perm: typing.Sequence = None) -> BKTensor:
    if perm is None:
        perm = list(reversed(list(range(rank(tensor)))))

    perm = list(map(int, perm))  # Convert darray to integer list if necessary
    return torch.stack((tensor[0].permute(*perm).contiguous(),
                        tensor[1].permute(*perm).contiguous()))


def diag(tensor: BKTensor) -> BKTensor:
    return torch.stack(torch.diag(tensor[0]), torch.diag(tensor[1]))


def reshape(tensor: BKTensor, shape: list) -> BKTensor:
    return torch.stack((tensor[0].view(*shape), tensor[1].view(*shape)))


def matmul(tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    tensor0_real, tensor0_imag = tensor0
    tensor1_real, tensor1_imag = tensor1

    new_real = tensor0_real @ tensor1_real - tensor0_imag @ tensor1_imag
    new_imag = tensor0_real @ tensor1_imag + tensor0_imag @ tensor1_real

    return torch.stack((new_real, new_imag))


def trace(tensor: BKTensor) -> complex:
    new_real = torch.trace(tensor[0])
    new_imag = torch.trace(tensor[1])
    return torch.stack((new_real, new_imag))


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)


def getitem(tensor: BKTensor, key: typing.Any) -> BKTensor:
    return tensor[0].__getitem__(key).cpu().numpy() \
            + 1.0j * tensor[1].__getitem__(key).cpu().numpy()


def productdiag(tensor: BKTensor) -> BKTensor:
    N = rank(tensor)
    tensor = reshape(tensor, [2**(N//2), 2**(N//2)])
    tensor = torch.stack((torch.diag(tensor[0]), torch.diag(tensor[1])))
    tensor = reshape(tensor, [2]*(N//2))
    return tensor


def einsum(subscripts: str, tensor0: BKTensor, tensor1: BKTensor) -> BKTensor:
    t0_re, t0_im = tensor0
    t1_re, t1_im = tensor1

    new_real = torch.einsum(subscripts, [t0_re, t1_re]) \
        - torch.einsum(subscripts, [t0_im, t1_im])
    new_imag = torch.einsum(subscripts, [t0_re, t1_im]) \
        + torch.einsum(subscripts, [t0_im, t1_re])

    return torch.stack((new_real, new_imag))


def tensormul(tensor0: BKTensor, tensor1: BKTensor,
              indices: typing.List[int]) -> BKTensor:
    N = rank(tensor1)
    K = rank(tensor0) // 2
    assert K == len(indices)

    gate = reshape(tensor0, [2**K, 2**K])

    perm = list(indices) + [n for n in range(N) if n not in indices]
    inv_perm = np.argsort(perm)

    tensor = tensor1
    tensor = transpose(tensor, perm)
    tensor = reshape(tensor, [2**K, 2**(N-K)])
    tensor = matmul(gate, tensor)
    tensor = reshape(tensor, [2]*N)
    tensor = transpose(tensor, inv_perm)

    return tensor
