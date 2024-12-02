#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:39:59 2024
      
Use Deep Energy Method
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
from utils.Solvers import Porous_DEM_BC_Nonlinear
from utils.bfgs import minimize as bfgs_minimize

       

#make figures bigger on HiDPI monitors
mpl.rcParams['figure.dpi'] = 200
#config.update("jax_enable_x64", True)

#define the input and output data set
beam_length = 2.
beam_width = 0.1
pressure = 1e-1 #1e4
domainCorners = jnp.array([[0., 0.], [0, beam_width], [beam_length, 0.], [beam_length, beam_width]])
geomDomain = Quadrilateral(domainCorners)

model_data = dict()
model_data["E"] = 200e3
model_data["nu"] = 0.333
model_data["state"] = "plane stress"

model_data["e0"] = 0.8
model_data["h"] = beam_width
model_data["L"] = beam_length
model_data["porousitystate"] = "state1"


Penalty_weight = 1e9

numElemU = 20
numElemV = 10
numGauss = 4
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
inert = beam_width**3/12
Ybnd_x = jnp.zeros_like(xPhysBnd).astype(data_type)
Ybnd_y = (jnp.zeros_like(xPhysBnd) - pressure).astype(data_type)
Ybnd = jnp.concatenate((Ybnd_x, Ybnd_y), axis=1)

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


plt.scatter(xPhys, yPhys, s=0.1)
plt.scatter(xPhysBnd, yPhysBnd, s=1, c='red')
plt.title("Boundary and interior integration points")
plt.show()
    
#define the model 
layers = [2, 20, 20, 20, 2]
key = random.PRNGKey(42)

num_epoch = 2000
print_epoch = 100

pred_model = Porous_DEM_BC_Nonlinear(key, layers, num_epoch, print_epoch, model_data, data_type, Penalty_weight)
params = pred_model.get_params(pred_model.opt_state)

#training
t0 = time.time()
print("Training (ADAM)...")

pred_model.train(Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")

params = pred_model.get_params(pred_model.opt_state)

print("Training (TFP-BFGS)...")

train_op2 = "TFP-BFGS"
loss_func = jax_tfp_function_factory(pred_model, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

initial_pos = loss_func.init_params_1d
tolerance = 1e-5
current_loss, _ = loss_func(initial_pos)
num_bfgs_iterations = 0
while True:
    results = bfgs_minimize(loss_func, initial_position = initial_pos,
                        max_iterations=1000, tolerance=1e-16)
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


print("Testing...")
numPtsUTest = 2*numElemU*numGauss
numPtsVTest = 2*numElemV*numGauss
xPhysTest, yPhysTest = jnp.array(geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1]))
XTest = jnp.concatenate((xPhysTest,yPhysTest),axis=1).astype(data_type)
YTest = pred_model(XTest, params)  

xPhysTest2D = jnp.resize(XTest[:,0], (numPtsVTest, numPtsUTest))
yPhysTest2D = jnp.resize(XTest[:,1], (numPtsVTest, numPtsUTest))
YTest2D_x = jnp.resize(YTest[:,0], (numPtsVTest, numPtsUTest))
YTest2D_y = jnp.resize(YTest[:,1], (numPtsVTest, numPtsUTest))

plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_x, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed x-displacement")
plt.axis('equal')
plt.show()

plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D_y, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Computed y-displacement")
plt.axis('equal')
plt.show()


