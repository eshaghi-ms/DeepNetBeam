#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:50:38 2020

@author: cosmin
"""
import jax.flatten_util

def jax_tfp_function_factory(model, params, *args):
    """A factory to create a function required by tensorflow_probability.substrates.jax.optimizer.lbfgs_minimize.
    Based on the example from https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        *args: arguments to be passed to model.get_grads method        
        
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """    
    #params = model.get_params(model.opt_state)
    init_params_1d, unflatten_params = jax.flatten_util.ravel_pytree(params)

    # now create a function that will be returned by this factory    
    def f(params_1d):
        """A function that can be used by tfp.substrates.jax.optimizer.lbfgs_minimize

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D (device) array.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        params = unflatten_params(params_1d)
        loss_value, grads = model.get_loss_and_grads(params, *args)
        grads, _ = jax.flatten_util.ravel_pytree(grads)
        #jax.debug.print("Train loss: {}", loss_value)
        # print out iteration & loss
        #f.iter += 1
        #if f.iter%1 == 0:
        #     print("Iter:", f.iter, "train loss:", loss_value)
        
        # # store loss value so we can retrieve later
        # f.history.append(loss_value)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = 0
    f.history = []
    f.init_params_1d = init_params_1d

    return f
