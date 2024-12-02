#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:53:01 2023

@author: mohammadsadegh
"""


import time
import matplotlib.pyplot as plt
import matplotlib as mpl

# jax libraries
import jax
import jax.numpy as jnp
from jax import random
#from jax.config import config

# utils libraries
from utils.jax_tfp_loss import jax_tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Porous_DEM_BC
from utils.bfgs import minimize as bfgs_minimize

#make figures bigger on HiDPI monitors
mpl.rcParams['figure.dpi'] = 200
#config.update("jax_enable_x64", True)

    

def Porous(beam_length, beam_width, pressure, E, nu, state, porosity, porosity_state, BCstate, penalty_weight, numElem, num_epoch, layers):
    #define the input and output data set

    domainCorners = jnp.array([[0., 0.], [0, beam_width], [beam_length, 0.], [beam_length, beam_width]])
    geomDomain = Quadrilateral(domainCorners)

    model_data = dict()
    model_data["E"] = E
    model_data["nu"] = nu
    model_data["state"] = state

    model_data["e0"] = porosity
    model_data["h"] = beam_width
    model_data["L"] = beam_length
    model_data["porousitystate"] = porosity_state
    model_data["BCstate"] = BCstate

    Penalty_weight = penalty_weight

    numElemU = numElem[0]
    numElemV = numElem[1]
    numGauss = numElem[2]
    xPhys, yPhys, Wint = jnp.array(geomDomain.getQuadIntPts(numElemU, numElemV, numGauss))
    data_type = "float32"

    Xint = jnp.concatenate((xPhys,yPhys),axis=1).astype(data_type)
    Wint = jnp.array(Wint).astype(data_type)

    # prepare boundary points in the fromat Xbnd = [Xcoord, Ycoord, norm_x, norm_y] and
    # Wbnd for boundary integration weights and
    # Ybnd = [trac_x, trac_y], where Xcoord, Ycoord are the x and y coordinates of the point,
    # norm_x, norm_y are the x and y components of the unit normals
    # trac_x, trac_y are the x and y components of the traction vector at each point
                   
    #boundary for x=beam_length, include both the x and y directions
    xPhysBnd, yPhysBnd, xNorm, yNorm, Wbnd = jnp.array(geomDomain.getQuadEdgePts(numElemU, numGauss, 3))
    Xbnd = jnp.concatenate((xPhysBnd, yPhysBnd, xNorm, yNorm), axis=1).astype(data_type)
    Wbnd = jnp.array(Wbnd).astype(data_type)
    #inert = beam_width**3/12
    Ybnd_x = jnp.zeros_like(xPhysBnd).astype(data_type)
    Ybnd_y = (jnp.zeros_like(xPhysBnd) - pressure).astype(data_type)
    Ybnd = jnp.concatenate((Ybnd_x, Ybnd_y), axis=1)
    
    if model_data["BCstate"]=="C-C":
        #Dirichlet boundary in x = 0 and x = L
        xPhysDL, yPhysDL, xNDL, yNDL, WbDL = jnp.array(geomDomain.getQuadEdgePts(numElemV, numGauss, 4))
        XbDL = jnp.concatenate((xPhysDL, yPhysDL, xNDL, yNDL), axis=1).astype(data_type)
        WbDL = jnp.array(WbDL).astype(data_type)

        xPhysDR, yPhysDR, xNDR, yNDR, WbDR = jnp.array(geomDomain.getQuadEdgePts(numElemV, numGauss, 2))
        XbDR = jnp.concatenate((xPhysDR, yPhysDR, xNDR, yNDR), axis=1).astype(data_type)
        WbDR = jnp.array(WbDR).astype(data_type)

        YbD_x = jnp.zeros_like(xPhysDL).astype(data_type)
        YbD_y = jnp.zeros_like(xPhysDL).astype(data_type)

        YbDL = jnp.concatenate((YbD_x, YbD_y), axis=1)
        YbDR = jnp.concatenate((YbD_x, YbD_y), axis=1)

        XbD = jnp.concatenate((XbDR, XbDL), axis=0)
        WbD = jnp.concatenate((WbDR, WbDL), axis=0)
        YbD = jnp.concatenate((YbDR, YbDL), axis=0)
    elif model_data["BCstate"]=="C-H":
        #Dirichlet boundary in x = 0
        xPhysDL, yPhysDL, xNDL, yNDL, WbDL = jnp.array(geomDomain.getQuadEdgePts(numElemV, numGauss, 4))
        XbDL = jnp.concatenate((xPhysDL, yPhysDL, xNDL, yNDL), axis=1).astype(data_type)
        WbDL = jnp.array(WbDL).astype(data_type)

        YbD_x_L = jnp.zeros_like(xPhysDL).astype(data_type)
        YbD_y_L = jnp.zeros_like(xPhysDL).astype(data_type)

        YbDL = jnp.concatenate((YbD_x_L, YbD_y_L), axis=1)

        #Dirichlet boundary in x = L
        xPhysDR = jnp.array([[beam_length]])
        yPhysDR = jnp.array([[beam_width/2]])
        xNDR = jnp.array([[1.0]])
        yNDR = jnp.array([[1.0]])
        WbDR = jnp.array([[1.0]])

        XbDR = jnp.concatenate((xPhysDR, yPhysDR, xNDR, yNDR), axis=1).astype(data_type)
        WbDR = jnp.array(WbDR).astype(data_type)

        YbD_x_R = jnp.zeros_like(xPhysDR).astype(data_type)
        YbD_y_R = jnp.zeros_like(xPhysDR).astype(data_type)
        YbDR = jnp.concatenate((YbD_x_R, YbD_y_R), axis=1)

        XbD = jnp.concatenate((XbDR, XbDL), axis=0)
        WbD = jnp.concatenate((WbDR, WbDL), axis=0)
        YbD = jnp.concatenate((YbDR, YbDL), axis=0)
    elif model_data["BCstate"]=="H-H":
        #Dirichlet boundary in x = 0
        xPhysDL = jnp.array([[0.0]])
        yPhysDL = jnp.array([[beam_width/2]])
        xNDL = jnp.array([[1.0]])
        yNDL = jnp.array([[1.0]])
        WbDL = jnp.array([[1.0]])

        XbDL = jnp.concatenate((xPhysDL, yPhysDL, xNDL, yNDL), axis=1).astype(data_type)
        WbDL = jnp.array(WbDL).astype(data_type)

        YbD_x_L = jnp.zeros_like(xPhysDL).astype(data_type)
        YbD_y_L = jnp.zeros_like(xPhysDL).astype(data_type)

        YbDL = jnp.concatenate((YbD_x_L, YbD_y_L), axis=1)

        #Dirichlet boundary in x = L
        xPhysDR = jnp.array([[beam_length]])
        yPhysDR = jnp.array([[beam_width/2]])
        xNDR = jnp.array([[1.0]])
        yNDR = jnp.array([[1.0]])
        WbDR = jnp.array([[1.0]])

        XbDR = jnp.concatenate((xPhysDR, yPhysDR, xNDR, yNDR), axis=1).astype(data_type)
        WbDR = jnp.array(WbDR).astype(data_type)

        YbD_x_R = jnp.zeros_like(xPhysDR).astype(data_type)
        YbD_y_R = jnp.zeros_like(xPhysDR).astype(data_type)
        YbDR = jnp.concatenate((YbD_x_R, YbD_y_R), axis=1)

        XbD = jnp.concatenate((XbDR, XbDL), axis=0)
        WbD = jnp.concatenate((WbDR, WbDL), axis=0)
        YbD = jnp.concatenate((YbDR, YbDL), axis=0)
    


        
    #define the model 
    #layers = [2, 20, 20, 20, 2]
    key = random.PRNGKey(42)

    print_epoch = 200

    pred_model = Porous_DEM_BC(key, layers, num_epoch[0], print_epoch, model_data, data_type, Penalty_weight)
    params = pred_model.get_params(pred_model.opt_state)


    #training
    t0 = time.time()
    print("Training (ADAM)...")

    pred_model.train(Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

    t1 = time.time()
    print("Time taken (ADAM)", t1-t0, "seconds")

    params = pred_model.get_params(pred_model.opt_state)

    print("Training (TFP-BFGS)...")

    #train_op2 = "TFP-BFGS"
    loss_func = jax_tfp_function_factory(pred_model, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

    initial_pos = loss_func.init_params_1d
    tolerance = 1e-5
    current_loss, _ = loss_func(initial_pos)
    num_bfgs_iterations = 0
    while True:
        results = bfgs_minimize(loss_func, initial_position = initial_pos,
                            max_iterations=num_epoch[1], tolerance=1e-16)
        initial_pos = results.position
        num_bfgs_iterations += results.num_iterations
        print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)
        if current_loss < results.objective_value-tolerance:
            current_loss = results.objective_value
        else:
            break
    _, unflatten_params = jax.flatten_util.ravel_pytree(params)
    params = unflatten_params(results.position)

    t2 = time.time()

    print("Time taken (TFP-BFGS)", t2-t1, "seconds")
    print("Time taken (all)", t2-t0, "seconds")


    numPtsUTest = 2*numElemU*numGauss
    numPtsVTest = 2*numElemV*numGauss
    xPhysTest, yPhysTest = jnp.array(geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1]))
    XTest = jnp.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
    YTest = pred_model(XTest, params)  

    YTest_x = YTest[:,0]
    YTest_y = YTest[:,1]

    
    eps_xx, eps_yy, eps_xy = pred_model.kinematicEq(params, xPhysTest, yPhysTest)
    stress_xx, stress_yy, stress_xy = pred_model.constitutiveEq(params, xPhysTest, yPhysTest)
    
    loss = pred_model.loss_log
    loss = [arr.ravel() for arr in loss]
    loss = jnp.concatenate(loss)

    return xPhysTest, yPhysTest, YTest_x, YTest_y,  eps_xx, eps_yy, eps_xy, stress_xx, stress_yy, stress_xy, loss
