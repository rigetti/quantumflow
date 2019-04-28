#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Canonical decompositions of pairs of 2-qubit gates."""

import os
import argparse

from matplotlib import pyplot, rcParams
from mpl_toolkits.mplot3d import Axes3D


os.environ['QUANTUMFLOW_BACKEND'] = 'numpy'
import quantumflow as qf                                    # noqa: E402

__version__ = qf.__version__
__description__ = """Canonical decompositions of pairs of 2-qubit gates.

We construct 2-qubit gate sandwiches, where two random 1-qubit gates
are sandwiched between two 2-qubit gates, which are specified by their
canonical coordinates). We then decompose to find the composite gate's
canonical coordinates within the Weyl chamber. A 3D view of the data
is constructed and shown with matplotlib.

See arXiv:1904.10541 "Fixed-Depth Two-Qubit Circuits and the Monodromy
Polytope" by Eric C. Peterson, Gavin E. Crooks, and Robert S. Smith.
For an introduction to the Weyl Chamber of 2-qubit gates, see
http://threeplusone.com/weyl and http://threeplusone.com/gates

Canonical Coordinates of 2-qubit Gates
--------------------------------------
The intrinsic non-local character of a 2-qubit gate can be captured by
the gate canonical coordinates with the Weyl chamber.

0    0    0   Identity   [Also (1, 0, 0)]
.5   0    0   CNOT / CZ
.5   .5   0   ISWAP / DCNOT
.5   .5   .5  SWAP

.25  0    0   sqrt(CNOT)  [Also (.75, 0, 0)]
.25  .25  0   sqrt(ISWAP) [Also (.75, .25, 0)]
.25  .25  .25 sqrt(SWAP)
.75  .25  .25 Adjoint of sqrt(SWAP)

.5   .25  0   B
.5   .25  .25 ECP
.375 .375 0 DB (Dagwood Bumstead)
.5   .5   .25 QFT (Parametric swap, half way between ISWAP and SWAP)


Interesting Sandwiches
----------------------

* CZ-CZ sandwich
Special orthongal gates. Bottom of Weyl chamber.
    ./weyl.py 0.5 0 0   0.5 0 0

* CZ-ISWAP sandwich
Improper orthongal gates. Plane connecting CZ, ISWAP, and SWAP.
    ./weyl.py 0.5 0 0   0.5 0.5 0

* sqrt(ISWAP) sandwich
Tetrahedron with the ECP gate as top point.
    ./weyl.py 0.25 0.25 0   0.25 0.25 0

* Dagwood Bumstead (DB) sandwich
Of all gates in the XY family, the Dagwood Bumstead gate makes the
biggest sandwich. (3/4 of all 2-qubit gates)
    ./weyl.py .375 .375 0  .375 .375 0

"""
__author__ = 'Gavin E. Crooks'


SAMPLES = 1024 * 2


def sandwich_decompositions(coords0, coords1, samples=SAMPLES):
    """Create composite gates, decompose, and return a list
    of canonical coordinates"""
    decomps = []
    for _ in range(samples):
        circ = qf.Circuit()
        circ += qf.CANONICAL(*coords0, 0, 1)
        circ += qf.random_gate([0])
        circ += qf.random_gate([1])
        circ += qf.CANONICAL(*coords1, 0, 1)
        gate = circ.asgate()

        coords = qf.canonical_coords(gate)
        decomps.append(coords)

    return decomps


def display_weyl(decomps):
    """Construct and display 3D plot of canonical coordinates"""
    tx, ty, tz = list(zip(*decomps))

    rcParams['axes.labelsize'] = 24
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(tx, ty, tz)
    ax.plot((1,), (1,), (1,))
    ax.plot((0, 1, 1/2, 0, 1/2, 1, 1/2, 1/2),
            (0, 0, 1/2, 0, 1/2, 0, 1/2, 1/2),
            (0, 0, 0, 0, 1/2, 0, 0, 1/2))
    ax.plot((0, 1/2, 1, 1/2, 1/2),
            (0, 1/4, 0, 1/4, 1/2),
            (0, 1/4, 0, 1/4, 0))

    points = [(0, 0, 0), (1/4, 0, 0), (1/2, 0, 0), (3/4, 0, 0), (1, 0, 0),
              (1/4, 1/4, 0), (1/2, 1/4, 0), (3/4, 1/4, 0), (1/2, 1/2, 0),
              (1/4, 1/4, 1/4), (1/2, 1/4, 1/4), (3/4, 1/4, 1/4),
              (1/2, 1/2, 1/4), (1/2, 1/2, 1/2)]

    ax.scatter(*zip(*points))
    eps = 0.04
    ax.text(0, 0, 0-2*eps, 'I', ha='center')
    ax.text(1, 0, 0-2*eps, 'I', ha='center')
    ax.text(1/2, 1/2, 0-2*eps, 'iSWAP', ha='center')
    ax.text(1/2, 1/2, 1/2+eps, 'SWAP', ha='center')
    ax.text(1/2, 0, 0-2*eps, 'CNOT', ha='center')

    # More coordinate labels
    # ax.text(1/4-eps, 1/4, 1/4, '$\sqrt{SWAP}$', ha='right')
    # ax.text(3/4+eps, 1/4, 1/4, '$\sqrt{SWAP}^\dagger$', ha='left')
    # ax.text(1/4, 0, 0-2*eps, '$\sqrt{{CNOT}}$', ha='center')
    # ax.text(3/4, 0, 0-2*eps, '$\sqrt{{CNOT}}$', ha='center')
    # ax.text(1/2, 1/4, 0-2*eps, 'B', ha='center')
    # ax.text(1/2, 1/4, 1/4+eps, 'ECP', ha='center')
    # ax.text(1/4, 1/4, 0-2*eps, '$\sqrt{iSWAP}$', ha='center')
    # ax.text(3/4, 1/4, 0-2*eps, '$\sqrt{iSWAP}$', ha='center')
    # ax.text(1/2, 1/2+eps, 1/4, 'PSWAP(1/2)', ha='left')

    ax.set_xlim(0, 1)
    ax.set_ylim(-1/4, 3/4)
    ax.set_zlim(-1/4, 3/4)

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    pyplot.show()


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('coords', nargs=6,
                        metavar='COORD',
                        type=float,
                        action='store',
                        help='Coordinates of initial and final gates'
                        )

    # Parse
    opts = vars(parser.parse_args())
    coords = opts.pop('coords')

    # Decompose and display
    decomps = sandwich_decompositions(coords[0:3], coords[3:6])
    display_weyl(decomps)


if __name__ == "__main__":
    _cli()
