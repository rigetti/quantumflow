#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utility script to generate random collections of graphs.
"""

import argparse
import networkx as nx
from qcml.utils import to_graph6

__version__ = "0.1.0"
__description__ = 'Generate a random collection of Erdős-Rényi graphs' + \
                  ' (in graph6 format)'


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(
        description=__description__)

    parser.add_argument('--version', action='version', version=__version__)

    parser.add_argument('-d', '--degree', type=float, action='store',
                        help='Degree')

    parser.add_argument('--family', type=str, action='store', default='er',
                        help='Graph family')

    parser.add_argument('N', type=int, action='store', help='Nodes')

    parser.add_argument('S', type=int, action='store', help='Samples')

    parser.add_argument('fout', action='store',
                        metavar='OUT_FILE', help='Write graphs to file')

    opts = vars(parser.parse_args())
    N = opts.pop('N')
    S = opts.pop('S')
    fout = opts.pop('fout')

    degree = opts.pop('degree')
    family = opts.pop('family')
    assert family in {'er', 'reg'}

    if family == 'reg':
        assert degree is not None
        degree = int(degree)

    with open(fout, 'w') as file:
        for _ in range(S):
            if family == 'er':
                graph = nx.gnp_random_graph(N, 0.5)
            elif family == 'reg':
                graph = nx.random_regular_graph(int(degree), N)
            else:
                assert False
            file.write(to_graph6(graph))
            file.write('\n')


if __name__ == "__main__":
    _cli()
