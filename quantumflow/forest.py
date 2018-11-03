
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantumflow: Interface to pyQuil and the Rigetti Forest
"""

from typing import Sequence, Dict

import networkx as nx
import PIL
import numpy as np

from .cbits import Register
from .qubits import Qubit
from .states import State
from .stdgates import STDGATES
from .ops import Gate
from .stdops import Measure
from .circuits import Circuit
from .programs import Program
from .parser import parse_quil
from .visualization import circuit_to_latex, render_latex

from pyquil.api._quantum_computer import _get_qvm_compiler_based_on_endpoint

from . import pyquil

__all__ = ['get_virtual_qc',
           'get_compiler',
           'NullCompiler',
           'circuit_to_pyquil',
           'pyquil_to_circuit',
           'qvm_run_and_measure',
           'wavefunction_to_state',
           'state_to_wavefunction',
           'QUILGATES',
           'QuantumFlowQVM',
           'pyquil_to_image']

QUILGATES = {'I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'PHASE',
             'RX', 'RY', 'RZ', 'CZ', 'CNOT', 'SWAP',
             'ISWAP', 'CPHASE00', 'CPHASE01', 'CPHASE10',
             'CPHASE', 'PSWAP', 'CCNOT', 'CSWAP', 'PISWAP'}
"""Names of Quil compatible gates"""


def get_virtual_qc(qubit_nb: int,
                   device: pyquil.NxDevice = None,
                   connection: pyquil.ForestConnection = None,
                   compiler: pyquil.AbstractCompiler = None,
                   noisy: bool = False,
                   name: str = None) -> pyquil.QuantumComputer:
    """
    Return a virtual quantum computer.

    Args:
        qubit_nb: Number of qubits
        device: An NxDevice. Defaults to a completely connected machine with
            gates between all qubit pairs.
        connection: Optional preexisting Forest connection
        compiler: If not specified, then we default to either a local or
            remote Rigetti quilc compiler
        noisy: Optionally add standard noise to the simulation.
        name: A name for this virtual quantum computer. Defaults to
            'NNq-generic-qvm', where NN is the number of qubits.
    """
    if name is None:
        name = '{}q-generic-qvm'.format(qubit_nb)

    if device is None:
        topology = nx.complete_graph(qubit_nb)
        device = pyquil.NxDevice(topology)

    if connection is None:
        connection = pyquil.ForestConnection()

    if noisy:
        gates = pyquil.gates_in_isa(device.get_isa())
        noise_model = pyquil.decoherence_noise_with_asymmetric_ro(gates)
    else:
        noise_model = None

    qvm = pyquil.QVM(connection=connection, noise_model=noise_model)

    if compiler is None:
        compiler = get_compiler(qubit_nb, device, connection)

    vqc = pyquil.QuantumComputer(name=name,
                                 qam=qvm,
                                 device=device,
                                 compiler=compiler)

    return vqc


def get_compiler(qubit_nb: int,
                 device: pyquil.NxDevice = None,
                 connection: pyquil.ForestConnection = None) \
        -> pyquil.AbstractCompiler:
    """
    Return a connection to `quilc`, the quil compiler.
    """
    if device is None:
        topology = nx.complete_graph(qubit_nb)
        device = pyquil.NxDevice(topology)

    if connection is None:
        connection = pyquil.ForestConnection()

    endpoint = connection.compiler_endpoint
    return _get_qvm_compiler_based_on_endpoint(endpoint, device)


class NullCompiler(pyquil.AbstractCompiler):
    """A null pyQuil compiler. Passes programs through unchanged"""
    def get_version_info(self) -> dict:
        return {}

    def quil_to_native_quil(self, program: pyquil.Program) -> pyquil.Program:
        return program

    def native_quil_to_executable(self, nq_program: pyquil.Program) \
            -> pyquil.Program:
        return nq_program


def qvm_run_and_measure(circ: Circuit, trials: int = 1) \
        -> Dict[Qubit, np.ndarray]:  # pragma: no cover
    # Tests only run if qvm and quilc are installed locally.
    N = circ.qubit_nb

    vqc = get_virtual_qc(N, compiler=NullCompiler())

    prog = circuit_to_pyquil(circ)
    results = vqc.run_and_measure(prog, trials)

    return results


def pyquil_to_image(program: pyquil.Program) -> PIL.Image:  # pragma: no cover
    """Returns an image of a pyquil circuit.

    See circuit_to_latex() for more details.
    """
    circ = pyquil_to_circuit(program)
    latex = circuit_to_latex(circ)
    img = render_latex(latex)
    return img


def circuit_to_pyquil(circuit: Circuit) -> pyquil.Program:
    """Convert a QuantumFlow circuit to a pyQuil program"""
    prog = pyquil.Program()

    for elem in circuit.elements:
        if isinstance(elem, Gate) and elem.name in QUILGATES:
            params = list(elem.params.values()) if elem.params else []
            prog.gate(elem.name, params, elem.qubits)
        elif isinstance(elem, Measure):
            for q in elem.qubits:
                prog.measure(q)
        else:
            # FIXME: more informative error message
            raise ValueError('Cannot convert operation to pyquil')

    return prog


def pyquil_to_circuit(program: pyquil.Program) -> Circuit:
    """Convert a protoquil pyQuil program to a QuantumFlow Circuit"""

    circ = Circuit()
    for inst in program.instructions:
        # print(type(inst))
        if isinstance(inst, pyquil.Declare):            # Ignore
            continue
        if isinstance(inst, pyquil.Halt):               # Ignore
            continue
        if isinstance(inst, pyquil.Pragma):             # TODO Barrier?
            continue
        elif isinstance(inst, pyquil.Measurement):
            circ += Measure(inst.qubit.index)
        # elif isinstance(inst, pyquil.ResetQubit):     # TODO
        #     continue
        elif isinstance(inst, pyquil.Gate):
            defgate = STDGATES[inst.name]
            gate = defgate(*inst.params)
            qubits = [q.index for q in inst.qubits]
            gate = gate.relabel(qubits)
            circ += gate
        else:
            raise ValueError('Program is not protoquil')

    return circ


def state_to_wavefunction(state: State) -> pyquil.Wavefunction:
    """Convert a QuantumFlow state to a pyQuil Wavefunction"""
    # TODO: qubits?
    amplitudes = state.vec.asarray()

    # pyQuil labels states backwards.
    amplitudes = amplitudes.transpose()
    amplitudes = amplitudes.reshape([amplitudes.size])
    return pyquil.Wavefunction(amplitudes)


def wavefunction_to_state(wfn: pyquil.Wavefunction) -> State:
    """Convert a pyQuil Wavefunction to a QuantumFlow State"""
    # TODO: qubits?
    return State(wfn.amplitudes.transpose())


# Note QAM status: 'connected', 'loaded', 'running', 'done'

class QuantumFlowQVM(pyquil.api.QAM):
    """A Quantum Virtual Machine that runs pyQuil programs."""

    def __init__(self) -> None:
        super().__init__()
        self.program: pyquil.Program
        self.status = 'connected'
        self._prog: Program
        self._ket: State

    def load(self, binary: pyquil.Program) -> 'QuantumFlowQVM':
        """
        Load a pyQuil program, and initialize QVM into a fresh state.

        Args:
            binary: A pyQuil program
        """

        assert self.status in ['connected', 'done']
        prog = parse_quil(str(binary))

        self._prog = prog
        self.program = binary
        self.status = 'loaded'

        return self

    def write_memory(self, *, region_name: str,
                     offset: int = 0, value: int = None) -> 'QuantumFlowQVM':
        # assert self.status in ['loaded', 'done']
        raise NotImplementedError()

    def run(self) -> 'QuantumFlowQVM':
        """Run a previously loaded program"""
        assert self.status in ['loaded']
        self.status = 'running'
        self._ket = self._prog.run()
        # Should set state to 'done' after run complete.
        # Makes no sense to keep status at running. But pyQuil's
        # QuantumComputer calls wait() after run, which expects state to be
        # 'running', and whose only effect to is to set state to 'done'
        return self

    def wait(self) -> 'QuantumFlowQVM':
        assert self.status == 'running'
        self.status = 'done'
        return self

    # Note: The star is here to prevent positional arguments (apparently)
    def read_from_memory_region(self, *, region_name: str,
                                offsets: Sequence[int] = None) \
            -> Sequence[int]:
        assert self.status == 'done'

        # Unclear what offsets is meant to do. Some pyquil code seems to think
        # its synonymous with classical_addresses, but other bits claim it's
        # an unsupported weird new feature of quil's memory model. Ignore.
        if offsets is not None:
            raise NotImplementedError('Offsets not yet supported')

        reg = Register(region_name)
        assert self._ket is not None
        memory = self._ket.memory
        result = [value for addr, value in memory.items()
                  if addr.register == reg]

        return result

    def wavefunction(self) -> pyquil.Wavefunction:
        """
        Return the wavefunction of a completed program.
        """
        assert self.status == 'done'
        assert self._ket is not None
        wavefn = state_to_wavefunction(self._ket)
        return wavefn
