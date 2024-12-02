#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:54:43 2023

@author: cosmin
"""
import numpy as np
def generate_quad_pts_weights_1d(x_min=0, x_max=1, num_elem=10, num_gauss_pts=4):
    """
    Generates the Gauss points and weights on a 1D interval (x_min, x_max), split
    into num_elem equal-length subintervals, each with num_gauss_pts quadature
    points per element.
    
    Note: the sum of the weights should equal to the length of the domain
    Parameters
    ----------
    x_min : (scalar)
        lower bound of the 1D domain.
    x_max : (scalar)
        upper bound of the 1D domain.
    num_elem : (integer)
        number of subdivision intervals or elements.
    num_gauss_pts : (integer)
        number of Gauss points in each element
    Returns
    -------
    pts : (1D array)
        coordinates of the integration points.
    weights : (1D array)
        weights corresponding to each point.
    """
    x_pts = np.linspace(x_min, x_max, num=num_elem+1)
    pts = np.zeros(num_elem*num_gauss_pts)
    weights = np.zeros(num_elem*num_gauss_pts)
    pts_ref, weights_ref = np.polynomial.legendre.leggauss(num_gauss_pts)
    for i in range(num_elem):
        x_min_int = x_pts[i]
        x_max_int = x_pts[i+1]        
        jacob_int = (x_max_int-x_min_int)/2
        pts_int = jacob_int*pts_ref + (x_max_int+x_min_int)/2
        weights_int = jacob_int * weights_ref
        pts[i*num_gauss_pts:(i+1)*num_gauss_pts] = pts_int
        weights[i*num_gauss_pts:(i+1)*num_gauss_pts] = weights_int        
        
    return pts, weights