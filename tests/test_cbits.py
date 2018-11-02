
# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import pytest

import quantumflow as qf


def test_register():
    ro = qf.Register()
    assert ro.name == 'ro'
    assert str(ro) == "Register('ro', 'BIT')"


def test_register_ordered():
    assert qf.Register() == qf.Register('ro')
    assert qf.Register('a') < qf.Register('b')
    assert qf.Register('a') != qf.Register('b')
    assert qf.Register('c') != 'foobar'

    with pytest.raises(TypeError):
        qf.Register('c') < 'foobar'


def test_addr():
    c = qf.Register('c')
    c0 = c[0]
    assert c0.register.name == 'c'
    assert c0.key == 0
    assert c0.register.dtype == 'BIT'

    assert str(c0) == 'c[0]'
    assert repr(c0) == "Register('c', 'BIT')[0]"


def test_addr_ordered():
    key = qf.Register('c')[0]
    d = dict({key: '1234'})
    assert d[key] == '1234'

    assert qf.Register('c')[0] == qf.Register('c')[0]
    assert qf.Register('c')[0] != qf.Register('c')[1]
    assert qf.Register('d')[0] != qf.Register('c')[0]

    assert qf.Register('c')[0] != 'foobar'
    assert qf.Register('c')[0] < qf.Register('c')[1]

    with pytest.raises(TypeError):
        qf.Register('c')[0] < 'foobar'
