#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Examples: Fit a 1-qubit gate using gradient descent,
using tensorflow 2.0
"""

import os
import tensorflow as tf

os.environ['QUANTUMFLOW_BACKEND'] = 'tensorflow2'
import quantumflow as qf                    # noqa: E402
import quantumflow.backend as bk            # noqa: E402


def fit_zyz(target_gate):
    """
    Tensorflow 2.0 example. Given an arbitrary one-qubit gate, use
    gradient descent to find corresponding parameters of a universal ZYZ
    gate.
    """

    steps = 1000

    dev = '/gpu:0' if bk.DEVICE == 'gpu' else '/cpu:0'

    with tf.device(dev):
        t = tf.Variable(tf.random.normal([3]))

        def loss_fn():
            """Loss"""
            gate = qf.ZYZ(t[0], t[1], t[2])
            ang = qf.fubini_study_angle(target_gate.vec, gate.vec)
            return ang

        opt = tf.optimizers.Adam(learning_rate=0.001)
        opt.minimize(loss_fn, var_list=[t])

        for step in range(steps):
            opt.minimize(loss_fn, var_list=[t])
            loss = loss_fn()
            print(step, loss.numpy())
            if loss < 0.01:
                break
        else:
            print("Failed to coverge")

    return bk.evaluate(t)


if __name__ == "__main__":
    def main():
        """CLI"""
        print(fit_zyz.__doc__)

        print('Fitting randomly selected 1-qubit gate...')
        target = qf.random_gate(1)
        params = fit_zyz(target)
        print('Fitted parameters:', params)

    main()
