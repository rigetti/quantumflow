
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow: References to classical memory
"""

from typing import Any, Hashable
from functools import total_ordering


__all__ = ['Register', 'Addr']

DTYPE = {'BIT', 'REAL', 'INT', 'OCTET', 'ANY'}

DEFAULT_REGION = 'ro'


# DOCME
@total_ordering
class Register:
    """Create addresses space for a register of classical memory"""
    def __init__(self, name: str = DEFAULT_REGION, dtype: str = 'BIT') -> None:
        assert dtype in DTYPE
        self.name = name
        self.dtype = dtype

    def __getitem__(self, key: Hashable) -> 'Addr':
        return Addr(self, key)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Register):
            return NotImplemented
        return (self.dtype, self.name) < (other.dtype, other.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Register):
            return NotImplemented
        return (self.dtype, self.name) == (other.dtype, other.name)

    def __hash__(self) -> int:
        return hash((self.dtype, self.name))

    def __repr__(self) -> str:
        return ('Register' + '(' + repr(self.name)
                + ', ' + repr(self.dtype) + ')')


@total_ordering
class Addr:
    """An address for an item of classical memory"""
    def __init__(self, register: 'Register', key: Hashable) -> None:
        self.register = register
        self.key = key

    @property
    def dtype(self) -> str:
        return self.register.dtype

    def __str__(self) -> str:
        return '{}[{}]'.format(self.register.name, self.key)

    def __repr__(self) -> str:
        return repr(self.register)+'['+repr(self.key)+']'

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Addr):
            return NotImplemented
        return (self.register, self.key) < (other.register, other.key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Addr):
            return NotImplemented
        return (self.register, self.key) == (other.register, other.key)

    def __hash__(self) -> int:
        return hash((self.register, self.key))
