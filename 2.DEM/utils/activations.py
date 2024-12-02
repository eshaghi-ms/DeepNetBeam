#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom activation functions

@author: cosmin
"""
import jax.numpy as jnp
import jax

def hat(x):
    return jnp.where(x>-1, jnp.where(x>0, jnp.where(x>1, 0., 1-x), x+1), 0.)

def relusq(x):
    return jax.nn.relu(x)**2

def quadspline(x):
    return jnp.where(x>-1.5, jnp.where(x>-0.5, jnp.where(x>0.5, jnp.where(x>1.5, 0., (1.5-x)**2/2), 
                                                         (x+1.5)*(0.5-x)/2+(1.5-x)*(x+0.5)/2),
                                       (x+1.5)**2/2), 0.)
    