#!/usr/bin/env python

# Copyright 2016-2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QuantumFlow Examples
"""

import sys
import os
import numpy as np

os.environ['QUANTUMFLOW_BACKEND'] = 'eager'
import quantumflow as qf                    # noqa: E402
import quantumflow.backend as bk            # noqa: E402


def fit_zyz(target_gate):
    """
    Tensorflow eager mode example. Given an arbitrary one-qubit gate, use
    gradient descent to find corresponding parameters of a universal ZYZ
    gate.
    """
    assert bk.BACKEND == 'eager'

    tf = bk.TL
    tfe = bk.tfe

    steps = 4000

    dev = '/gpu:0' if bk.DEVICE == 'gpu' else '/cpu:0'

    with tf.device(dev):
        t = tfe.Variable(np.random.normal(size=[3]), name='t')

        def loss_fn():
            """Loss"""
            gate = qf.ZYZ(t[0], t[1], t[2])
            ang = qf.fubini_study_angle(target_gate.vec, gate.vec)
            return ang

        loss_and_grads = tfe.implicit_value_and_gradients(loss_fn)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        # train = opt.minimize(ang, var_list=[t])

        for step in range(steps):
            loss, grads_and_vars = loss_and_grads()
            sys.stdout.write('\r')
            sys.stdout.write("step: {:3d} loss: {:10.9f}".format(step,
                                                                 loss.numpy()))
            if loss < 0.0001:
                break
            opt.apply_gradients(grads_and_vars)

        print()
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
