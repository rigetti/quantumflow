# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Visualizations of quantum circuits,
"""

from typing import Any
import os
import subprocess
import tempfile

import PIL
import sympy

from .qubits import Qubits
from .gates import P0, P1
from .stdgates import (I,  # X, Y, Z, H, T, S,
                       T_H, S_H, RX, RY, RZ, TX, TY,
                       TZ, CNOT, CZ, SWAP, ISWAP, CCNOT, CSWAP,
                       XX, YY, ZZ)
from .stdops import Reset, Measure
from .utils import symbolize
from .circuits import Circuit
from .dagcircuit import DAGCircuit


__all__ = ('LATEX_GATESET',
           'circuit_to_latex',
           'render_latex',
           'circuit_to_image')


# TODO: Should be set of types to match GATESET in stdgates?
LATEX_GATESET = ['I', 'X', 'Y', 'Z', 'H', 'T', 'S', 'T_H', 'S_H', 'RX', 'RY',
                 'RZ', 'TX', 'TY', 'TZ', 'CNOT', 'CZ', 'SWAP', 'ISWAP',
                 'CCNOT', 'CSWAP', 'XX', 'YY', 'ZZ', 'P0', 'P1', 'RESET']

# TODO: Gates not yet supported by latex: PISWAP, PHASE, CANONICAL, CPHASE ...
# Possibly convert unsupported gates to displayable gates.


def circuit_to_latex(circ: Circuit,
                     qubits: Qubits = None,
                     document: bool = True) -> str:
    """
    Create an image of a quantum circuit in LaTeX.

    Can currently draw X, Y, Z, H, T, S, T_H, S_H, RX, RY, RZ, TX, TY, TZ,
    CNOT, CZ, SWAP, ISWAP, CCNOT, CSWAP, XX, YY, ZZ, P0 and P1 gates,
    and the RESET operation.

    Args:
        circ:       A quantum Circuit
        qubits:     Optional qubit list to specify qubit order
        document:   If false, just the qcircuit latex is returned. Else the
                    circuit image is wrapped in a standalone LaTeX document
                    ready for typesetting.
    Returns:
        A LaTeX string representation of the circuit.

    Raises:
        NotImplementedError: For unsupported gates.

    Refs:
        LaTeX Qcircuit package
            (https://arxiv.org/pdf/quant-ph/0406003).
    """
    if qubits is None:
        qubits = circ.qubits
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    layers = _display_layers(circ, qubits)

    layer_code = []
    code = [r'\lstick{\ket{' + str(q) + '}}' for q in qubits]
    layer_code.append(code)

    for layer in layers.elements:
        code = [r'\qw'] * N
        assert isinstance(layer, Circuit)
        for gate in layer:
            idx = [qubit_idx[q] for q in gate.qubits]

            name = gate.name
            if isinstance(gate, I):
                pass
            elif(len(idx) == 1) and name in ['X', 'Y', 'Z', 'H', 'T', 'S']:
                code[idx[0]] = r'\gate{' + gate.name + '}'
            elif isinstance(gate, S_H):
                code[idx[0]] = r'\gate{S^\dag}'
            elif isinstance(gate, T_H):
                code[idx[0]] = r'\gate{T^\dag}'
            elif isinstance(gate, RX):
                theta = _latex_format(gate.params['theta'])
                code[idx[0]] = r'\gate{R_x(%s)}' % theta
            elif isinstance(gate, RY):
                theta = _latex_format(gate.params['theta'])
                code[idx[0]] = r'\gate{R_y(%s)}' % theta
            elif isinstance(gate, RZ):
                theta = _latex_format(gate.params['theta'])
                code[idx[0]] = r'\gate{R_z(%s)}' % theta
            elif isinstance(gate, TX):
                t = _latex_format(gate.params['t'])
                code[idx[0]] = r'\gate{X^{%s}}' % t
            elif isinstance(gate, TY):
                t = _latex_format(gate.params['t'])
                code[idx[0]] = r'\gate{Y^{%s}}' % t
            elif isinstance(gate, TZ):
                t = _latex_format(gate.params['t'])
                code[idx[0]] = r'\gate{Z^{%s}}' % t
            elif isinstance(gate, CNOT):
                code[idx[0]] = r'\ctrl{' + str(idx[1] - idx[0]) + '}'
                code[idx[1]] = r'\targ'
            elif isinstance(gate, XX):
                t = _latex_format(gate.params['t'])
                d = idx[1] - idx[0]
                code[idx[0]] = r'\gate{X\!X^{%s}} \qwx[%s]' % (t, d)
                code[idx[1]] = r'\gate{X\!X^{%s}}' % t
            elif isinstance(gate, YY):
                t = _latex_format(gate.params['t'])
                d = idx[1] - idx[0]
                code[idx[0]] = r'\gate{Y\!Y^{%s}} \qwx[%s]' % (t, d)
                code[idx[1]] = r'\gate{Y\!Y^{%s}}' % t
            elif isinstance(gate, ZZ):
                t = _latex_format(gate.params['t'])
                d = idx[1] - idx[0]
                code[idx[0]] = r'\gate{Z\!Z^{%s}} \qwx[%s]' % (t, d)
                code[idx[1]] = r'\gate{Z\!Z^{%s}}' % t
            elif isinstance(gate, CZ):
                code[idx[0]] = r'\ctrl{' + str(idx[1] - idx[0]) + '}'
                code[idx[1]] = r'\ctrl{' + str(idx[0] - idx[1]) + '}'
            elif isinstance(gate, SWAP):
                code[idx[0]] = r'\qswap \qwx[' + str(idx[1] - idx[0]) + ']'
                code[idx[1]] = r'\qswap'
            elif isinstance(gate, ISWAP):
                code[idx[0]] = r'\sgate{\scriptstyle \text{iSWAP}}{' \
                               + str(idx[1] - idx[0]) + '}'
                code[idx[1]] = r'\gate{\scriptstyle \text{iSWAP}}'
            elif isinstance(gate, CCNOT):
                code[idx[0]] = r'\ctrl{' + str(idx[1]-idx[0]) + '}'
                code[idx[1]] = r'\ctrl{' + str(idx[2]-idx[1]) + '}'
                code[idx[2]] = r'\targ'
            elif isinstance(gate, CSWAP):
                code[idx[0]] = r'\ctrl{' + str(idx[1]-idx[0]) + '}'
                code[idx[1]] = r'\qswap \qwx[' + str(idx[2] - idx[1]) + ']'
                code[idx[2]] = r'\qswap'
            elif isinstance(gate, P0):
                code[idx[0]] = r'\push{\ket{0}\!\!\bra{0}} \qw'
            elif isinstance(gate, P1):
                code[idx[0]] = r'\push{\ket{1}\!\!\bra{1}} \qw'
            elif isinstance(gate, Reset):
                for i in idx:
                    code[i] = r'\push{\rule{0.1em}{0.5em}\, \ket{0}\,} \qw'
            elif isinstance(gate, Measure):
                code[idx[0]] = r'\meter'
            else:
                raise NotImplementedError(str(gate))

        layer_code.append(code)

    code = [r'\qw'] * N
    layer_code.append(code)

    latex_lines = [''] * N

    for line, wire in enumerate(zip(*layer_code)):
        latex = '& ' + ' & '.join(wire)
        if line < N - 1:  # Not last line
            latex += r' \\'
        latex_lines[line] = latex

    latex_code = _QCIRCUIT % '\n'.join(latex_lines)

    if document:
        latex_code = _DOCUMENT_HEADER + latex_code + _DOCUMENT_FOOTER

    return latex_code


_DOCUMENT_HEADER = r"""
\documentclass[border={20pt 4pt 20pt 4pt}]{standalone}
\usepackage[braket, qm]{qcircuit}
\usepackage{amsmath}
\begin{document}
"""

_QCIRCUIT = r"""\Qcircuit @C=1em @R=.7em {
%s
}"""

_DOCUMENT_FOOTER = r"""
\end{document}
"""


def _display_layers(circ: Circuit, qubits: Qubits) -> Circuit:
    """Separate a circuit into groups of gates that do not visually overlap"""
    N = len(qubits)
    qubit_idx = dict(zip(qubits, range(N)))
    gate_layers = DAGCircuit(circ).layers()

    layers = []
    lcirc = Circuit()
    layers.append(lcirc)
    unused = [True] * N

    for gl in gate_layers:
        assert isinstance(gl, Circuit)
        for gate in gl:
            indices = [qubit_idx[q] for q in gate.qubits]

            if not all(unused[min(indices):max(indices)+1]):
                # New layer
                lcirc = Circuit()
                layers.append(lcirc)
                unused = [True] * N

            unused[min(indices):max(indices)+1] = \
                [False] * (max(indices) - min(indices) + 1)
            lcirc += gate

    return Circuit(layers)


def render_latex(latex: str) -> PIL.Image:
    """
    Convert a single page LaTeX document into an image.

    To display the returned image, `img.show()`


    Required external dependencies: `pdflatex` (with `qcircuit` package),
    and `poppler` (for `pdftocairo`).

    Args:
        A LaTeX document as a string.

    Returns:
        A PIL Image

    Raises:
        OSError: If an external dependency is not installed.
    """
    tmpfilename = 'circ'
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename)
        with open(tmppath + '.tex', 'w') as latex_file:
            latex_file.write(latex)

        subprocess.run(["pdflatex",
                        "-halt-on-error",
                        "-output-directory={}".format(tmpdirname),
                        "{}".format(tmpfilename+'.tex')],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL,
                       check=True)

        subprocess.run(['pdftocairo',
                        '-singlefile',
                        '-png',
                        '-q',
                        tmppath + '.pdf',
                        tmppath])
        img = PIL.Image.open(tmppath + '.png')

    return img


def circuit_to_image(circ: Circuit,
                     qubits: Qubits = None) -> PIL.Image:
    """Create an image of a quantum circuit.

    A convenience function that calls circuit_to_latex() and render_latex().

    Args:
        circ:       A quantum Circuit
        qubits:     Optional qubit list to specify qubit order

    Returns:
        Returns: A PIL Image (Use img.show() to display)

    Raises:
        NotImplementedError: For unsupported gates.
        OSError: If an external dependency is not installed.
    """
    latex = circuit_to_latex(circ, qubits)
    img = render_latex(latex)
    return img


# ==== UTILITIES ====

def _latex_format(obj: Any) -> str:
    """Format an object as a latex string."""
    if isinstance(obj, float):
        try:
            return sympy.latex(symbolize(obj))
        except ValueError:
            return "{0:.4g}".format(obj)

    return str(obj)

# fin
