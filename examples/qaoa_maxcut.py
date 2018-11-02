#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""QAOA graph maxcut using tensorflow gradient decent."""

import os
import sys
import argparse
import ast

from numpy.random import normal
import networkx as nx
import tensorflow as tf

# Note that we QuantumFlow sets an interactive session
os.environ['QUANTUMFLOW_BACKEND'] = 'tensorflow'
import quantumflow as qf                                    # noqa: E402
from quantumflow.qaoa import qubo_circuit, graph_cuts       # noqa: E402
from quantumflow.utils import to_graph6                     # noqa: E402
assert qf.backend.BACKEND == 'tensorflow'


__version__ = qf.__version__
__description__ = 'QAOA graph maxcut using tensorflow gradient decent'
__author__ = 'Gavin E. Crooks'

DEFAULT_NODES = 7
DEFAULT_STEPS = 5
LEARNING_RATE = 0.01
MAX_OPT_STEPS = 10000


def maxcut_qaoa(
        graph,
        steps=DEFAULT_STEPS,
        learning_rate=LEARNING_RATE,
        verbose=False):
    """QAOA Maxcut using tensorflow"""

    if not isinstance(graph, nx.Graph):
        graph = nx.from_edgelist(graph)

    init_scale = 0.01
    init_bias = 0.5

    init_beta = normal(loc=init_bias, scale=init_scale, size=[steps])
    init_gamma = normal(loc=init_bias, scale=init_scale, size=[steps])

    beta = tf.get_variable('beta', initializer=init_beta)
    gamma = tf.get_variable('gamma', initializer=init_gamma)

    circ = qubo_circuit(graph, steps, beta, gamma)
    cuts = graph_cuts(graph)
    maxcut = cuts.max()
    expect = circ.run().expectation(cuts)
    loss = - expect

    # === Optimization ===
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = opt.minimize(loss, var_list=[beta, gamma])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        block = 10
        min_difference = 0.0001
        last_ratio = -1
        for step in range(0, MAX_OPT_STEPS, block):
            for _ in range(block):
                sess.run(train)
            ratio = sess.run(expect) / maxcut
            if ratio - last_ratio < min_difference:
                break
            last_ratio = ratio

            if verbose:
                print("# step: {}  ratio: {:.4f}%".format(step, ratio))

        opt_beta = sess.run(beta)
        opt_gamma = sess.run(gamma)

    return ratio, opt_beta, opt_gamma


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(
        description=__description__)

    parser.add_argument('--version', action='version', version=__version__)

    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('-i', '--fin', action='store', dest='fin',
                        default='', metavar='FILE',
                        help='Read parameters from file')

    parser.add_argument('-o', '--fout', action='store', dest='fout',
                        default='', metavar='FILE',
                        help='Write parameters to file')

    parser.add_argument('-N', '--nodes', type=int, dest='nodes',
                        default=DEFAULT_NODES)

    parser.add_argument('-P', '--steps', type=int, dest='steps',
                        default=DEFAULT_STEPS)

    parser.add_argument('--lr', type=float, dest='learning_rate',
                        default=LEARNING_RATE)

    parser.add_argument('graph', nargs='*', metavar='GRAPH', type=str,
                        action='store', default=['random'],
                        help='Graph to maxcut')

    examples = {
        'ring8':
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)],
        'example1':
            [(0, 3), (0, 5), (0, 6), (1, 4), (1, 6), (1, 7), (2, 5), (2, 6),
             (2, 7), (3, 7), (4, 7), (6, 7)],
        'example2':  # 0.93
            [(0, 4), (0, 6), (1, 4), (1, 7), (2, 5), (2, 6), (3, 5), (3, 7),
             (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)],
        'example3':
            [(0, 2), (0, 4), (0, 7), (1, 2), (1, 5), (1, 6), (2, 3), (2, 5),
             (2, 6), (2, 7), (3, 5), (3, 7), (4, 6), (4, 7), (5, 6), (5, 7)],
    }

    # Run command
    opts = vars(parser.parse_args())

    verbose = opts.pop('verbose')

    steps = opts.pop('steps')
    N = opts.pop('nodes')
    learning_rate = opts.pop('learning_rate')

    fin = opts.pop('fin')
    if fin:
        graphs = nx.read_graph6(fin)
    else:
        graphs = []
        for grapharg in opts.pop('graph'):
            if grapharg == 'random':
                graph = nx.gnp_random_graph(N, 0.5)
            elif grapharg in examples:
                graph = examples[grapharg]
            elif grapharg[0] == '[':
                graph = ast.literal_eval(grapharg)
            else:
                graph = nx.from_graph6_bytes(grapharg)

            graphs.append(graph)

    for graph in graphs:
        if not isinstance(graph, nx.Graph):
            graph = nx.from_edgelist(graph)

        ratio, betas, gammas = maxcut_qaoa(graph, steps=steps,
                                           learning_rate=learning_rate,
                                           verbose=verbose)

        # YAML compatible output.
        print('---')
        print('graph:', to_graph6(graph))
        print('edges:', list(map(list, graph.edges())))
        print('ratio: {:.4f}'.format(ratio))
        print('beta: ', list(betas))
        print('gamma:', list(gammas))
        print()
        sys.stdout.flush()

        tf.reset_default_graph()


if __name__ == "__main__":
    _cli()
