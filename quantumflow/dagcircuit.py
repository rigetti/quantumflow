# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Directed Acyclic Graph representations of a Circuit.
"""

from typing import List, Dict, Iterable, Iterator, Generator

import numpy as np
import networkx as nx

from .qubits import Qubit, Qubits
from .states import State, Density
from .ops import Operation, Gate, Channel
from .circuits import Circuit
from .utils import invert_map


__all__ = 'DAGCircuit',


class DAGCircuit(Operation):
    """A Directed Acyclic Graph representation of a Circuit.

    The circuit is converted to a networkx directed acyclic multi-graph,
    stored in the `graph` attribute.

    There are 3 node types, 'in' nodes representing qubits at the start of a
    circuit; operation nodes; and 'out' nodes for qubits at the
    end of a circuit. Edges are directed from 'in' to 'out' via the Operation
    nodes. Each edge is keyed to a qubit.

    A DAGCircuit is considered a mutable object, like Circuit, the other
    composite Operation class.

    DAGCircuit is iterable, yielding all of the operation nodes in
    topological sort order.

    Note: Provisional API
    """
    def __init__(self, elements: Iterable[Operation]) -> None:
        G = nx.MultiDiGraph()

        for elem in elements:
            if isinstance(elem, tuple):  # Filter in and out nodes
                continue
            G.add_node(elem)
            for qubit in elem.qubits:
                qin = ('in', qubit)
                qout = ('out', qubit)
                if not G.has_node(qout):
                    G.add_edge(qin, qout)
                prev = list(G.predecessors(qout))[0]
                G.remove_edge(prev, qout)
                G.add_edge(prev, elem, key=qubit)
                G.add_edge(elem, qout, key=qubit)

        self.graph = G

    @property
    def qubits(self) -> Qubits:
        G = self.graph
        in_nodes = [node for node, deg in G.in_degree() if deg == 0]
        if not in_nodes:
            return ()
        _, qubits = zip(*in_nodes)
        qubits = tuple(sorted(qubits))
        return qubits

    @property
    def qubit_nb(self) -> int:
        return len(self.qubits)

    @property
    def H(self) -> 'DAGCircuit':
        return DAGCircuit(Circuit(self).H)

    def run(self, ket: State) -> State:
        for elem in self:
            ket = elem.run(ket)
        return ket

    def evolve(self, rho: Density) -> Density:
        for elem in self:
            rho = elem.evolve(rho)
        return rho

    def asgate(self) -> Gate:
        return Circuit(self).asgate()

    def aschannel(self) -> Channel:
        return Circuit(self).aschannel()

    def depth(self, local: bool = True) -> int:
        """Return the circuit depth.

        Args:
            local:  If True include local one-qubit gates in depth
                calculation. Else return the multi-qubit gate depth.
        """
        G = self.graph
        if not local:
            def remove_local(dagc: DAGCircuit) \
                    -> Generator[Operation, None, None]:
                for elem in dagc:
                    if dagc.graph.degree[elem] > 2:
                        yield elem
            G = DAGCircuit(remove_local(self)).graph

        return nx.dag_longest_path_length(G) - 1

    def size(self) -> int:
        """Return the number of operations."""
        return self.graph.order() - 2 * self.qubit_nb

    def component_nb(self) -> int:
        """Return the number of independent components that this
        DAGCircuit can be split into."""
        return nx.number_weakly_connected_components(self.graph)

    def components(self) -> List['DAGCircuit']:
        """Split DAGCircuit into independent components"""
        comps = nx.weakly_connected_component_subgraphs(self.graph)
        return [DAGCircuit(comp) for comp in comps]

    def layers(self) -> Circuit:
        """Split DAGCircuit into layers, where the operations within each
        layer operate on different qubits (and therefore commute).

        Returns: A Circuit of Circuits, one Circuit per layer
        """
        node_depth: Dict[Qubit, int] = {}
        G = self.graph

        for elem in self:
            depth = np.max(list(node_depth.get(prev, -1) + 1
                           for prev in G.predecessors(elem)))
            node_depth[elem] = depth

        depth_nodes = invert_map(node_depth, one_to_one=False)

        layers = []
        for nd in range(0, self.depth()):
            elements = depth_nodes[nd]
            circ = Circuit(list(elements))
            layers.append(circ)

        return Circuit(layers)

    def __iter__(self) -> Iterator[Operation]:
        for elem in nx.topological_sort(self.graph):
            if isinstance(elem, Operation):
                yield elem

    # DOCME TESTME
    def next_element(self, elem: Operation, qubit: Qubit = None) -> Operation:
        for _, node, key in self.graph.edges(elem, keys=True):
            if qubit is None or key == qubit:
                return node
        assert False        # pragma: no cover  # Insanity check

    # DOCME TESTME
    def prev_element(self, elem: Operation, qubit: Qubit = None) -> Operation:
        for node, _, key in self.graph.in_edges(elem, keys=True):
            if qubit is None or key == qubit:
                return node
        assert False        # pragma: no cover  # Insanity check

# End class DAGCircuit
