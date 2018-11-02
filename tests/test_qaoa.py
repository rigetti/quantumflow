
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.qaoa
"""

import numpy as np
import networkx as nx

from quantumflow.qaoa import qubo_circuit, graph_cuts


def test_qubo_circuit():

    # Random graph
    graph = nx.gnp_random_graph(4, 0.5)
    circ = qubo_circuit(graph, 4, [10, 11, 12, 13], [20, 21, 22, 23])
    # print(circ)

    # Circuit with edge weights
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.1)
    graph.add_edge(1, 2, weight=0.4)

    circ = qubo_circuit(graph, 2, [1, 1], [2, 2])
    assert len(circ.elements) == 13
    # print(circ)

    # Add node weights
    graph.nodes[0]['weight'] = 4
    circ = qubo_circuit(graph, 2, [1, 1], [2, 2])
    assert len(circ.elements) == 15
    print(circ)


def test_graph_cuts():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    cut = graph_cuts(graph)
    cut = np.resize(cut, (8))
    assert np.allclose(cut, [0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0])

    # Weighted graph
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.1)
    graph.add_edge(1, 2, weight=0.4)
    graph.add_edge(0, 2, weight=0.35)
    cut = graph_cuts(graph)
    cut = np.resize(cut, (8))
    assert np.allclose(cut, [0., 0.75, 0.5, 0.45, 0.45, 0.5, 0.75, 0.])
