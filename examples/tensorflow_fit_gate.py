#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow examples
"""

import sys
import os

os.environ['QUANTUMFLOW_BACKEND'] = 'tensorflow'
import quantumflow as qf                                # noqa: E402
import quantumflow.backend as bk                        # noqa: E402


def fit_zyz(target_gate):
    """
    Tensorflow example. Given an arbitrary one-qubit gate, use gradient
    descent to find corresponding parameters of a universal ZYZ gate.
    """

    assert bk.BACKEND == 'tensorflow'

    tf = bk.TL
    steps = 4000

    t = tf.get_variable('t', [3])
    gate = qf.ZYZ(t[0], t[1], t[2])

    ang = qf.fubini_study_angle(target_gate.vec, gate.vec)
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train = opt.minimize(ang, var_list=[t])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for step in range(steps):
            sess.run(train)
            loss = sess.run(ang)
            sys.stdout.write('\r')
            sys.stdout.write("step: {} gate_angle: {}".format(step, loss))
            if loss < 0.0001:
                break
        print()
        return sess.run(t)


if __name__ == "__main__":
    def main():
        """CLI"""
        print(fit_zyz.__doc__)

        print('Fitting randomly selected 1-qubit gate...')
        target_gate = qf.random_gate(1)
        t = fit_zyz(target_gate)
        print('Fitted parameters:', t)
    main()
