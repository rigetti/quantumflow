
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: utilities

Useful routines not necessarily intended to be part of the public API.
"""

from typing import Any, Sequence, Callable, Set
import warnings
import functools
from fractions import Fraction

import numpy as np
import networkx as nx

import sympy

# from scipy.linalg import fractional_matrix_power as matpow # Matrix power
# from scipy.linalg import sqrtm as matsqrt   # Matrix square root

__all__ = ['invert_map',
           'bitlist_to_int',
           'int_to_bitlist',
           'deprecated',
           'from_graph6',
           'to_graph6',
           'spanning_tree_count',
           'octagonal_tiling_graph',
           'cis',
           'rationalize']


def invert_map(mapping: dict, one_to_one: bool = True) -> dict:
    """Invert a dictionary. If not one_to_one then the inverted
    map will contain lists of former keys as values.
    """
    if one_to_one:
        inv_map = {value: key for key, value in mapping.items()}
    else:
        inv_map = {}
        for key, value in mapping.items():
            inv_map.setdefault(value, set()).add(key)

    return inv_map


def bitlist_to_int(bitlist: Sequence[int]) -> int:
    """Converts a sequence of bits to an integer.

    >>> from quantumflow.utils import bitlist_to_int
    >>> bitlist_to_int([1, 0, 0])
    4
    """
    return int(''.join([str(d) for d in bitlist]), 2)


def int_to_bitlist(x: int, pad: int = None) -> Sequence[int]:
    """Converts an integer to a binary sequence of bits.

    Pad prepends with sufficient zeros to ensures that the returned list
    contains at least this number of bits.

    >>> from quantumflow.utils import int_to_bitlist
    >>> int_to_bitlist(4, 4))
    [0, 1, 0, 0]
    """
    if pad is None:
        form = '{:0b}'
    else:
        form = '{:0' + str(pad) + 'b}'

    return [int(b) for b in form.format(x)]


def deprecated(func: Callable) -> Callable:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def _new_func(*args: Any, **kwargs: Any) -> Any:
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return _new_func


# -- Graphs --

def from_graph6(graph6: str) -> nx.Graph:
    """Convert a string encoded in graph6 format to a networkx graph"""
    return nx.from_graph6_bytes(bytes(graph6, "utf-8"))


def to_graph6(graph: nx.Graph) -> str:
    """Convert a networkx graph to a string in to a graph6 format"""
    # Networkx makes this surprisingly tricky.
    return nx.to_graph6_bytes(graph)[10:-1].decode("utf-8")


def spanning_tree_count(graph: nx.Graph) -> int:
    """Return the number of unique spanning trees of a graph, using
    Kirchhoff's matrix tree theorem.
    """
    laplacian = nx.laplacian_matrix(graph).toarray()
    comatrix = laplacian[:-1, :-1]
    det = np.linalg.det(comatrix)
    count = int(round(det))
    return count


def octagonal_tiling_graph(M: int, N: int) -> nx.Graph:
    """Return the octagonal tiling graph (4.8.8, truncated square tiling,
    truncated quadrille) of MxNx8 nodes

    The 'positions' node attribute gives node coordinates for the octagonal
    tiling. (Nodes are located on a square lattice, and edge lengths are
    uneven)
    """

    grp = nx.Graph()
    octogon = [[(1, 0), (0, 1)],
               [(0, 1), (0, 2)],
               [(0, 2), (1, 3)],
               [(1, 3), (2, 3)],
               [(2, 3), (3, 2)],
               [(3, 2), (3, 1)],
               [(3, 1), (2, 0)],
               [(2, 0), (1, 0)]]
    left = [[(1, 0), (1, -1)], [(2, 0), (2, -1)]]
    up = [[(0, 1), (-1, 1)], [(0, 2), (-1, 2)]]

    for m in range(M):
        for n in range(N):
            edges = octogon
            if n != 0:
                edges = edges + left
            if m != 0:
                edges = edges + up

            for (x0, y0), (x1, y1) in edges:
                grp.add_edge((m*4+x0, n*4+y0), (m*4+x1, n*4+y1))

    positions = {node: node for node in grp}
    nx.set_node_attributes(grp, positions, 'positions')

    return grp


# -- More Math --

def cis(x: float) -> complex:
    r"""
    Implements Euler's formula
    :math:`\text{cis}(x) = e^{i \pi x} = \cos(x) + i \sin(x)`
    """
    return np.cos(x) + 1.0j * np.sin(x)


_DENOMINATORS = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64,
                     128, 256, 512, 1024, 2048, 4096, 8192])


def rationalize(flt: float, denominators: Set[int] = None) -> Fraction:
    """Convert a floating point number to a Fraction with a small
    denominator.

    Args:
        flt:            A floating point number
        denominators:   Collection of standard denominators. Default is
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64, 128, 256, 512,
            1024, 2048, 4096, 8192

    Raises:
        ValueError:     If cannot rationalize float
    """
    if denominators is None:
        denominators = _DENOMINATORS
    frac = Fraction.from_float(flt).limit_denominator()
    if frac.denominator not in denominators:
        raise ValueError('Cannot rationalize')
    return frac


def symbolize(flt: float) -> sympy.Symbol:
    """Attempt to convert a real number into a simpler symbolic
    representation.

    Returns:
        A sympy Symbol. (Convert to string with str(sym) or to latex with
            sympy.latex(sym)
    Raises:
        ValueError:     If cannot simplify float
    """
    try:
        ratio = rationalize(flt)
        res = sympy.simplify(ratio)
    except ValueError:
        ratio = rationalize(flt/np.pi)
        res = sympy.simplify(ratio) * sympy.pi
    return res

# fin
