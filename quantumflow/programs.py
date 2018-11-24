
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: Programs and program instructions

In QF, Circuits contain only a list of Operations which are executed in
sequence, whereas Programs can contain non-linear control flow.
"""

# Callable and State imported for typing pragmas

from typing import List, Generator, Dict, Union, Tuple
from abc import ABC  # Abstract Base Class

# import numpy as np

# import sympy
from sympy import Symbol as Parameter

from .cbits import Addr, Register
from .qubits import Qubits
from .states import zero_state, State, Density
# from .ops import Gate
from .stdgates import STDGATES
# from .utils import symbolize
# from .backend import TensorLike


__all__ = ['Instruction', 'Program', 'DefCircuit', 'Wait',
           'Nop', 'Halt',
           'Label', 'Jump', 'JumpWhen', 'JumpUnless',
           'Pragma', 'Include', 'Call',
           'Declare',
           # 'Convert',
           'Load', 'Store',
           'Parameter',
           # 'DefGate', 'quil_parameter', 'eval_parameter'
           ]


# Private register used to store program state
_prog_state_ = Register('_prog_state_')
PC = _prog_state_['pc']
NAMEDGATES = _prog_state_['namedgates']
TARGETS = _prog_state_['targets']
WAIT = _prog_state_['wait']

HALTED = -1             # Program counter of finished programs


class Instruction(ABC):
    """
    An program instruction a hybrid quantum-classical program. Represents
    such operations as control flow and declarations.
    """

    _qubits: Qubits = ()

    @property
    def qubits(self) -> Qubits:
        """Return the qubits that this operation acts upon"""
        return self._qubits

    @property
    def qubit_nb(self) -> int:
        """Return the total number of qubits"""
        return len(self.qubits)

    @property
    def name(self) -> str:
        """Return the name of this operation"""
        return self.__class__.__name__.upper()

    def quil(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.quil()

    def run(self, ket: State) -> State:
        """Apply the action of this operation upon a pure state"""
        raise NotImplementedError()          # pragma: no cover

    def evolve(self, rho: Density) -> Density:
        # For purely classical Instructions the action of run() and evolve()
        # is the same
        res = self.run(rho)
        assert isinstance(res, Density)
        return res


class Program(Instruction):
    """A Program for a hybrid quantum computer, following the Quil
    quantum instruction set architecture.
    """

    def __init__(self,
                 instructions: List[Instruction] = None,
                 name: str = None,
                 params: dict = None) -> None:
        if instructions is None:
            instructions = []
        self.instructions = instructions

    def quil(self) -> str:
        items = [str(i) for i in self.instructions]
        items.append("")
        res = "\n".join(items)
        return res

    @property
    def qubits(self) -> Qubits:
        allqubits = [instr.qubits for instr in self.instructions]   # Gather
        qubits = [qubit for q in allqubits for qubit in q]          # Flatten
        qubits = list(set(qubits))                                  # Unique
        qubits.sort()                                               # Sort
        return qubits

    def __iadd__(self, other: Instruction) -> 'Program':
        """Append an instruction to the end of the program"""
        self.instructions.append(other)
        return self

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, key: int) -> Instruction:
        return self.instructions[key]

    def __iter__(self) -> Generator[Instruction, None, None]:
        for inst in self.instructions:
            yield inst

    def _initilize(self, state: State) -> State:
        """Initialize program state. Called by program.run() and .evolve()"""

        targets = {}
        for pc, instr in enumerate(self):
            if isinstance(instr, Label):
                targets[instr.target] = pc

        state = state.update({PC: 0,
                              TARGETS: targets,
                              NAMEDGATES: STDGATES.copy()})
        return state

    # FIXME: can't nest programs?
    def run(self, ket: State = None) -> State:
        """Compiles and runs a program. The optional program argument
        supplies the initial state and memory. Else qubits and classical
        bits start from zero states.
        """
        if ket is None:
            qubits = self.qubits
            ket = zero_state(qubits)

        ket = self._initilize(ket)

        pc = 0
        while pc >= 0 and pc < len(self):
            instr = self.instructions[pc]
            ket = ket.update({PC: pc + 1})
            ket = instr.run(ket)
            pc = ket.memory[PC]

        return ket

    # DOCME
    # TESTME
    def evolve(self, rho: Density = None) -> Density:
        if rho is None:
            rho = zero_state(self.qubits).asdensity()

        rho1 = self._initilize(rho)
        assert isinstance(rho1, Density)     # Make type checker happy
        rho = rho1

        pc = 0
        while pc >= 0 and pc < len(self):
            instr = self.instructions[pc]
            rho = rho.update({PC: pc + 1})
            rho = instr.evolve(rho)
            pc = rho.memory[PC]

        return rho


# TODO: Rename DefProgram? SubProgram?, Subroutine???
# FIXME: Not implemented
class DefCircuit(Program):
    """Define a parameterized sub-circuit of instructions."""

    # Note that Quil's DEFCIRCUIT can include arbitrary instructions,
    # whereas QuantumFlows Circuit contains only quantum gates. (Called
    # protoquil in pyQuil)

    def __init__(self,
                 name: str,
                 params: Dict[str, float],
                 qubits: Qubits = None,
                 instructions: List[Instruction] = None) \
            -> None:
        # DOCME: Not clear how params is meant to work
        super().__init__(instructions)
        if qubits is None:
            qubits = []
        self.progname = name
        self.params = params
        self._qubits = qubits

    @property
    def qubits(self) -> Qubits:
        return self._qubits

    def quil(self) -> str:
        if self.params:
            fparams = '(' + ','.join(map(str, self.params)) + ')'
        else:
            fparams = ""

        if self.qubits:
            # FIXME: Not clear what type self.qubits is expected to be?
            # These are named qubits?
            # Fix, test, and remove pragma
            fqubits = ' ' + ' '.join(map(str, self.qubits))  # pragma: no cover
        else:
            fqubits = ''

        result = '{} {}{}{}:\n'.format(self.name, self.progname,
                                       fparams, fqubits)

        for instr in self.instructions:
            result += "    "
            result += str(instr)
            result += "\n"

        return result


class Wait(Instruction):
    """Returns control to higher level by calling Program.wait()"""
    def run(self, ket: State) -> State:
        # FIXME: callback
        return ket


class Nop(Instruction):
    """No operation"""
    def run(self, ket: State) -> State:
        return ket


class Halt(Instruction):
    """Halt program and exit"""
    def run(self, ket: State) -> State:
        ket = ket.update({PC: HALTED})
        return ket


class Load(Instruction):
    """ The LOAD instruction."""

    def __init__(self, target: Addr, left: str, right: Addr) -> None:
        self.target = target
        self.left = left
        self.right = right

    def quil(self) -> str:
        return '{} {} {} {}'.format(self.name, self.target,
                                    self.left, self.right)

    def run(self, ket: State) -> State:
        raise NotImplementedError()         # pragma: no cover


class Store(Instruction):
    """ The STORE instruction."""

    def __init__(self, target: str, left: Addr,
                 right: Union[int, Addr]) -> None:
        self.target = target
        self.left = left
        self.right = right

    def quil(self) -> str:
        return '{} {} {} {}'.format(self.name, self.target,
                                    self.left, self.right)

    def run(self, ket: State) -> State:
        raise NotImplementedError()         # pragma: no cover


class Label(Instruction):
    """Set a jump target."""
    def __init__(self, target: str) -> None:
        self.target = target

    def quil(self) -> str:
        return '{} @{}'.format(self.name, self.target)

    def run(self, ket: State) -> State:
        return ket


class Jump(Instruction):
    """Unconditional jump to target label"""
    def __init__(self, target: str) -> None:
        self.target = target

    def quil(self) -> str:
        return '{} @{}'.format(self.name, self.target)

    def run(self, ket: State) -> State:
        dest = ket.memory[TARGETS][self.target]
        return ket.update({PC: dest})


class JumpWhen(Instruction):
    """Jump to target label if a classical bit is one."""
    def __init__(self, target: str, condition: Addr) -> None:
        self.target = target
        self.condition = condition

    def quil(self) -> str:
        return '{} @{} {}'.format(self.name, self.target, self.condition)

    def run(self, ket: State) -> State:
        if ket.memory[self.condition]:
            dest = ket.memory[TARGETS][self.target]
            return ket.update({PC: dest})
        return ket

    @property
    def name(self) -> str:
        return "JUMP-WHEN"


class JumpUnless(Instruction):
    """Jump to target label if a classical bit is zero."""
    def __init__(self, target: str, condition: Addr) -> None:
        # super().__init__()
        self.target = target
        self.condition = condition

    def quil(self) -> str:
        return '{} @{} {}'.format(self.name, self.target, self.condition)

    def run(self, ket: State) -> State:
        if not ket.memory[self.condition]:
            dest = ket.memory[TARGETS][self.target]
            return ket.update({PC: dest})
        return ket

    @property
    def name(self) -> str:
        return "JUMP-UNLESS"


class Pragma(Instruction):
    """
    A PRAGMA instruction.

    This is printed in QUIL as::
        PRAGMA <command> <arg>* "<freeform>"?
    """
    def __init__(self,
                 command: str,
                 args: List[float] = None,
                 freeform: str = None) -> None:
        self.command = command
        self.args = args
        self.freeform = freeform

    def quil(self) -> str:
        ret = ["PRAGMA {}".format(self.command)]
        if self.args:
            ret.extend(str(a) for a in self.args)
        if self.freeform:
            ret.append('"{}"'.format(self.freeform))
        return ' '.join(ret)

    def run(self, ket: State) -> State:
        return ket


class Include(Instruction):
    """Include additional file of quil instructions.

    (Currently recorded, but not acted upon)
    """
    def __init__(self, filename: str, program: Program = None) -> None:
        # DOCME: What is program argument for? How is this meant to work?
        self.filename = filename
        self.program = program

    def quil(self) -> str:
        return '{} "{}"'.format(self.name, self.filename)

    def run(self, ket: State) -> State:
        raise NotImplementedError()         # pragma: no cover


class Call(Instruction):
    """Pass control to a named gate or circuit"""
    def __init__(self,
                 name: str,
                 params: List[Parameter],
                 qubits: Qubits) -> None:
        self.gatename = name
        self.params = params
        self._qubits = qubits

    def quil(self) -> str:
        if self.qubits:
            fqubits = " "+" ".join([str(qubit) for qubit in self.qubits])
        else:
            fqubits = ""
        if self.params:
            fparams = "(" + ", ".join(str(p) for p in self.params) \
                + ")"
        else:
            fparams = ""
        return "{}{}{}".format(self.gatename, fparams, fqubits)

    def run(self, ket: State) -> State:
        namedgates = ket.memory[NAMEDGATES]
        if self.gatename not in namedgates:
            raise RuntimeError('Unknown named gate')

        gateclass = namedgates[self.gatename]
        gate = gateclass(*self.params)
        gate = gate.relabel(self.qubits)

        ket = gate.run(ket)
        return ket

# FIXME
# class DefGate(Instruction):
#     """Define a new quantum gate."""
#     def __init__(self,
#                  gatename: str,
#                  matrix: TensorLike,
#                  params: List[Parameter] = None) -> None:
#         self.gatename = gatename
#         self.matrix = matrix
#         self.params = params

#     def quil(self) -> str:
#         if self.params:
#             fparams = '(' + ','.join(map(str, self.params)) + ')'
#         else:
#             fparams = ''

#         result = '{} {}{}:\n'.format(self.name, self.gatename, fparams)

#         for row in self.matrix:
#             result += "    "
#             # FIXME: Hack to format complex numbers as quil expects
#             result += ", ".join(str(col).replace('*I', 'i')
#                                         .replace('j', 'i')
#                                         .replace('1.0i', 'i') for col in row)
#             result += "\n"

#         return result

#     def _create_gate(self, *gateparams: float) -> Gate:
#         if not self.params:
#             matrix = np.asarray(self.matrix)
#             gate = Gate(matrix, name=self.gatename)
#         else:
#             params = {str(p): complex(g) for p, g
#                       in zip(self.params, gateparams)}
#             K = len(self.matrix)
#             matrix = np.zeros(shape=(K, K), dtype=np.complex)
#             for i in range(K):
#                 for j in range(K):
#                     value = eval_parameter(self.matrix[i][j], params)
#                     matrix[i, j] = value
#             matrix = np.asarray(matrix)
#             gate = Gate(matrix, name=self.gatename)
#         return gate

#     def run(self, ket: State) -> State:
#         namedgates = ket.memory[NAMEDGATES].copy()
#         namedgates[self.gatename] = self._create_gate
#         ket = ket.update({NAMEDGATES: namedgates})
#         return ket


class Declare(Instruction):
    """Declare classical memory"""
    def __init__(self, memory_name: str,
                 memory_type: str,
                 memory_size: int,
                 shared_region: str = None,
                 offsets: List[Tuple[int, str]] = None) -> None:

        self.memory_name = memory_name
        self.memory_type = memory_type
        self.memory_size = memory_size
        self.shared_region = shared_region
        self.offsets = offsets

    def quil(self) -> str:
        ql = ['DECLARE']
        ql += [self.memory_name]
        ql += [self.memory_type]
        if self.memory_size != 1:
            ql += ['[{}]'.format(self.memory_size)]

        if self.shared_region is not None:
            ql += ['SHARING']
            ql += [self.shared_region]

            # if self.offsets:
            #     for loc, name in self.offsets:
            #         ql += ['OFFSET', str(loc), name]

        return ' '.join(ql)

    def run(self, ket: State) -> State:
        reg = Register(self.memory_name, self.memory_type)
        mem = {reg[idx]: 0 for idx in range(self.memory_size)}
        return ket.update(mem)


# ==== UTILITIES ====

# def _param_format(obj: Any) -> str:
#     """Format an object as a latex string."""
#     if isinstance(obj, float):
#         try:
#             return str(symbolize(obj))
#         except ValueError:
#             return "{}".format(obj)

#     return str(obj)


# def quil_parameter(symbol: str) -> Parameter:
#     """Create a quil parameter.

#     Args:
#         symbol: The variable name. If the first character is not '%' then a
#             percent sign will be prepended.

#     Returns:
#         A sympy symbol object representing the quil variable.
#     """
#     if symbol[0] != '%':
#         symbol = '%' + symbol
#     return sympy.symbols(symbol)


# def eval_parameter(param: Any, params: Dict[str, Any]) -> complex:
#     """Evaluate a quil parameter (a sympy Symbol). Ordinary
#     python numbers will be passed through unchanged.
#     """
#     return sympy.N(param, subs=params)
