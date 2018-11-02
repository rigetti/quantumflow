
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: quil parser
"""
# Kudos: Adapted from pyquil._parser written by Steven Heidel


from typing import TextIO, Any, List

import numpy as np

import sympy

from antlr4 import ParseTreeWalker, InputStream, CommonTokenStream, FileStream

from pyquil._parser.gen3.QuilLexer import QuilLexer
from pyquil._parser.gen3.QuilListener import QuilListener
from pyquil._parser.gen3.QuilParser import QuilParser as QP
from pyquil._parser.PyQuilListener import CustomErrorListener

from . import programs as inst
from . import stdops as ops
from .cbits import Register

__all__ = ['parse_quil', 'parse_quilfile', 'QUIL_RESERVED_WORDS']


QUIL_RESERVED_WORDS = ['DEFGATE', 'DEFCIRCUIT', 'MEASURE', 'LABEL', 'HALT',
                       'JUMP', 'JUMP-WHEN', 'JUMP-UNLESS', 'RESET', 'WAIT',
                       'NOP', 'INCLUDE', 'PRAGMA', 'DECLARE', 'NEG', 'NOT',
                       'AND', 'IOR', 'XOR', 'MOVE', 'EXCHANGE', 'CONVERT',
                       'ADD', 'SUB', 'MUL', 'DIV', 'EQ', 'GT', 'GE', 'LT',
                       'LE', 'LOAD', 'STORE', 'TRUE', 'FALSE', 'OR']


def parse_quil(quil: str) -> inst.Program:
    """
    Parse a Quil program and return a Program.

    To convert a pyQuil program to a QuantumFlow Program, first convert to
    Quil, `quantumflow_program = qf.parse_quil(str(pyquil_program))`
    """
    input_stream = InputStream(quil)
    return _parse(input_stream)


def parse_quilfile(filename: str) -> inst.Program:
    """Parse a file of quil code, and return a Program"""
    input_stream = FileStream(filename)
    return _parse(input_stream, filename)


def _parse(input_stream: TextIO, filename: str = None) -> inst.Program:
    lexer = QuilLexer(input_stream)
    stream = CommonTokenStream(lexer)

    # Step 2: Run the Parser
    parser = QP(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(CustomErrorListener())
    tree = parser.quil()

    # Step 3: Run the Listener
    qflistener = QFListener(filename=filename)
    walker = ParseTreeWalker()
    walker.walk(qflistener, tree)

    return inst.Program(qflistener.prog)


class QFListener(QuilListener):
    """QFListener quil listener"""
    def __init__(self, filename: str = None) -> None:
        super().__init__()
        self.prog: List = []
        self.stack: List = []
        self.filename = filename

    def top(self, N: int) -> list:
        """Pop top N items from stack"""
        if N == 0:
            return []
        items = self.stack[-N:]
        self.stack = self.stack[:-N]
        return items

    def exitQuil(self, ctx: QP.QuilContext) -> None:
        assert not self.stack  # Insanity check that stack is empty at end

    def exitAllInstr(self, ctx: QP.AllInstrContext) -> None:
        pass

    def exitInstr(self, ctx: QP.InstrContext) -> None:
        pass

    def exitGate(self, ctx: QP.GateContext) -> None:
        # Note: Gate -> Call
        K = len(ctx.qubit())
        bits = self.top(K)
        param_nb = len(ctx.param())
        params = self.top(param_nb)
        name = self.stack.pop()
        self.prog += [inst.Call(name, params, bits)]

    def exitName(self, ctx: QP.NameContext) -> None:
        self.stack += [ctx.IDENTIFIER().getText()]

    def exitQubit(self, ctx: QP.QubitContext) -> None:
        self.stack += [int(ctx.getText())]

    def exitParam(self, ctx: QP.ParamContext) -> None:
        pass        # Handled by various exit???Exp expression methods

    def exitDefGate(self, ctx: QP.DefGateContext) -> None:
        matrix = self.stack.pop()
        param_nb = len(ctx.variable())  # parameter -> variable
        params = self.top(param_nb)
        name = self.stack.pop()
        self.prog += [inst.DefGate(name, matrix, params)]

    def exitVariable(self, ctx: QP.VariableContext) -> None:
        self.stack += [inst.quil_parameter(ctx.IDENTIFIER().getText())]

    def exitMatrix(self, ctx: QP.MatrixContext) -> None:
        K = len(ctx.matrixRow())
        matrix = self.top(K)
        self.stack += [matrix]

    def exitMatrixRow(self, ctx: QP.MatrixRowContext) -> None:
        K = len(ctx.expression())
        row = self.top(K)
        self.stack += [row]

    def enterDefCircuit(self, ctx: QP.DefCircuitContext) -> None:
        pass  # pragma: no cover

    def exitDefCircuit(self, ctx: QP.DefCircuitContext) -> None:
        msg = "DEFCIRCUIT not supported"  # pragma: no cover
        raise NotImplementedError(msg)    # pragma: no cover

    def exitCircuit(self, ctx: QP.CircuitContext) -> None:
        msg = "DEFCIRCUIT not supported"  # pragma: no cover
        raise NotImplementedError(msg)    # pragma: no cover

    def exitMeasure(self, ctx: QP.MeasureContext) -> None:
        addr = self.stack.pop() if ctx.addr() else None
        qubit = self.stack.pop()
        self.prog += [ops.Measure(qubit, addr)]

    def exitAddr(self, ctx: QP.AddrContext) -> None:
        if ctx.IDENTIFIER() is None:
            # TODO: Deprecation warning
            region = 'ro'
        else:
            region = ctx.IDENTIFIER().getText()

        offset = int(ctx.INT().getText()) if ctx.INT() is not None else 0

        addr = Register(region)[offset]
        self.stack += [addr]

    def exitDefLabel(self, ctx: QP.DefLabelContext) -> None:
        label = self.stack.pop()
        self.prog += [inst.Label(label)]

    def exitLabel(self, ctx: QP.LabelContext) -> None:
        self.stack += [ctx.IDENTIFIER().getText()]

    def exitHalt(self, ctx: QP.HaltContext) -> None:
        self.prog += [inst.Halt()]

    def exitJump(self, ctx: QP.JumpContext) -> None:
        target = self.stack.pop()
        self.prog += [inst.Jump(target)]

    def exitJumpWhen(self, ctx: QP.JumpWhenContext) -> None:
        condition = self.stack.pop()
        target = self.stack.pop()
        self.prog += [inst.JumpWhen(target, condition)]

    def exitJumpUnless(self, ctx: QP.JumpUnlessContext) -> None:
        condition = self.stack.pop()
        target = self.stack.pop()
        self.prog += [inst.JumpUnless(target, condition)]

    def exitResetState(self, ctx: QP.ResetStateContext) -> None:
        if ctx.qubit():
            qubit = self.stack.pop()
            # FIXME: Replace with RESET operation?
            self.prog += [ops.Reset(qubit)]    # TESTME
        else:
            self.prog += [ops.Reset()]

    def exitWait(self, ctx: QP.WaitContext) -> None:
        self.prog += [inst.Wait()]

    def exitClassicalUnary(self, ctx: QP.ClassicalUnaryContext) -> None:
        addr = self.stack.pop()
        if ctx.TRUE():
            self.prog += [ops.TRUE(addr)]
        elif ctx.FALSE():
            self.prog += [ops.FALSE(addr)]
        elif ctx.NOT():
            self.prog += [ops.Not(addr)]
        elif ctx.NEG():
            self.prog += [ops.Neg(addr)]
        else:
            assert False    # pragma: no cover

    def exitLogicalBinaryOp(self, ctx: QP.LogicalBinaryOpContext) -> None:
        right = self.stack.pop()
        left = self.stack.pop()

        if ctx.AND():
            self.prog += [ops.And(left, right)]
        elif ctx.IOR():
            self.prog += [ops.Ior(left, right)]
        elif ctx.XOR():
            self.prog += [ops.Xor(left, right)]
        elif ctx.OR():
            self.prog += [ops.Or(left, right)]
        else:
            assert False    # pragma: no cover

    # TESTME
    # TODO
    def exitArithmeticBinaryOp(self,
                               ctx: QP.ArithmeticBinaryOpContext) -> None:

        right = self.stack.pop()
        left = self.stack.pop()
        if ctx.ADD():
            self.prog += [ops.Add(left, right)]
        elif ctx.SUB():
            self.prog += [ops.Sub(left, right)]
        elif ctx.MUL():
            self.prog += [ops.Mul(left, right)]
        elif ctx.DIV():
            self.prog += [ops.Div(left, right)]
        else:
            assert False    # pragma: no cover

    def exitMove(self, ctx: QP.MoveContext) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        self.prog += [ops.Move(left, right)]

    def exitExchange(self, ctx: QP.ExchangeContext) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        self.prog += [ops.Exchange(left, right)]

    # TODO
    # def exitConvert(self, ctx: QP.ConvertContext) -> None:
    #     right = self.stack.pop()
    #     left = self.stack.pop()
    #     self.prog += [inst.Convert(left, right)]

    def exitLoad(self, ctx: QP.LoadContext) -> None:
        right = self.stack.pop()
        left = ctx.IDENTIFIER().getText()
        target = self.stack.pop()
        self.prog += [inst.Load(target, left, right)]

    def exitStore(self, ctx: QP.LoadContext) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        target = ctx.IDENTIFIER().getText()
        self.prog += [inst.Store(target, left, right)]

    # TESTME
    # TODO
    def exitClassicalComparison(self,
                                ctx: QP.ClassicalComparisonContext) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        target = self.stack.pop()

        if ctx.EQ():
            self.prog += [ops.EQ(target, left, right)]
        elif ctx.GT():
            self.prog += [ops.GT(target, left, right)]
        elif ctx.GE():
            self.prog += [ops.GE(target, left, right)]
        elif ctx.LT():
            self.prog += [ops.LT(target, left, right)]
        elif ctx.LE():
            self.prog += [ops.LE(target, left, right)]
        else:
            assert False        # pragma: no cover

    # TODO
    # TESTME
    def exitMemoryDescriptor(self, ctx: QP.MemoryDescriptorContext) -> None:
        name = ctx.IDENTIFIER(0).getText()
        memory_type = ctx.IDENTIFIER(1).getText()
        memory_size = int(ctx.INT().getText()) if ctx.INT() else 1

        if ctx.SHARING():
            shared_region = ctx.IDENTIFIER(2).getText()
            offsets = [(int(offset_ctx.INT().getText()),
                        offset_ctx.IDENTIFIER().getText())
                       for offset_ctx in ctx.offsetDescriptor()]
        else:
            shared_region = None
            offsets = []

        self.prog += [inst.Declare(name, memory_type, memory_size,
                                   shared_region=shared_region,
                                   offsets=offsets)]

    def exitNop(self, ctx: QP.NopContext) -> None:
        self.prog += [inst.Nop()]

    def exitInclude(self, ctx: QP.IncludeContext) -> None:
        filename = ctx.STRING().getText()[1:-1]
        self.prog += [inst.Include(filename, None)]

    def exitPragma(self, ctx: QP.PragmaContext) -> None:
        command = ctx.IDENTIFIER().getText()
        N = len(ctx.pragma_name())
        args = [self.stack.pop() for _ in range(N)]
        args.reverse()

        freeform = ctx.STRING().getText()[1:-1] if ctx.STRING() else None
        self.prog += [inst.Pragma(command, args, freeform)]

    def exitPragma_name(self, ctx: QP.Pragma_nameContext) -> None:
        if ctx.IDENTIFIER():
            self.stack += [ctx.IDENTIFIER().getText()]
        elif ctx.INT():
            self.stack += [int(ctx.INT().getText())]

    def exitNumberExp(self, ctx: QP.NumberExpContext) -> None:
        pass        # Handled by exitNumber

    def exitPowerExp(self, ctx: QP.PowerExpContext) -> None:
        arg1 = self.stack.pop()
        arg0 = self.stack.pop()
        self.stack += [arg0 ** arg1]

    def exitMulDivExp(self, ctx: QP.MulDivExpContext) -> None:
        arg1 = self.stack.pop()
        arg0 = self.stack.pop()
        if ctx.TIMES():
            self.stack += [arg0 * arg1]
        elif ctx.DIVIDE():
            self.stack += [arg0 / arg1]
        else:
            assert False  # pragma: no cover

    def exitParenthesisExp(self, ctx: QP.ParenthesisExpContext) -> None:
        pass        # Handled by sub-expressions

    def exitVariableExp(self, ctx: QP.VariableExpContext) -> None:
        pass        # Handled by exitVariable

    def exitSignedExp(self, ctx: QP.SignedExpContext) -> None:
        self.stack += [self.stack.pop() * self.stack.pop()]

    def exitAddSubExp(self, ctx: QP.AddSubExpContext) -> None:
        arg1 = self.stack.pop()
        arg0 = self.stack.pop()
        if ctx.PLUS():
            self.stack += [arg0 + arg1]
        elif ctx.MINUS():
            self.stack += [arg0 - arg1]
        else:
            assert False  # pragma: no cover

    def exitFunctionExp(self, ctx: QP.FunctionExpContext) -> None:
        arg = self.stack.pop()
        fname = self.stack.pop()

        funcs = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'sqrt': sqrt,       # sympy function. See below.
            'exp': sympy.exp,
            'cis': cis          # sympy function. See below.
            }
        func = funcs[fname]
        self.stack += [func(arg)]

    def exitFunction(self, ctx: QP.FunctionContext) -> None:
        self.stack += [ctx.getText()]

    def exitSign(self, ctx: QP.SignContext) -> None:
        if ctx.MINUS():
            self.stack += [-1]
        elif ctx.PLUS():
            self.stack += [+1]

    def exitNumber(self, ctx: QP.NumberContext) -> None:
        if ctx.I():
            self.stack += [1.0j]
        elif ctx.PI():
            self.stack += [np.pi]

    def exitImaginaryN(self, ctx: QP.ImaginaryNContext) -> None:
        self.stack += [1.0j * self.stack.pop()]

    def exitRealN(self, ctx: QP.RealNContext) -> None:
        if ctx.FLOAT():
            self.stack += [float(ctx.getText())]
        elif ctx.INT():
            self.stack += [int(ctx.getText())]

# End class QFListener


class cis(sympy.Function):
    """Private sympy function for the quil cis function"""
    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg.is_Number:
            return sympy.cos(arg) + 1.0j * sympy.sin(arg)
        return None  # TODO: Correct?


class sqrt(sympy.Function):
    """Private sympy function for the quil sqrt function. Prevents
    automatic simplification from sqrt(x) to x ** 0.5"""
    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg.is_Number:
            return sympy.sqrt(arg)
        return None  # TODO: Correct?

# Fin
