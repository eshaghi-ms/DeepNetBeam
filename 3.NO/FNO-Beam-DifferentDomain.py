#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator for a 2D elasticity problem on a FGM beam with 
random distribution of Elasticity modulus under different random tensions

Problem statement:
    \Omega = (0,0.1)x(0,2) 
    Fixed BC: x = 0 and x = 2
    Traction \tau = GRF at y=0.1 in the vertical direction   
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
import time
import os
import math

from utils.postprocessing_NO import plot_pred_timo
from utils.fno_2d  import FNO2d_domain
from utils.fno_utils import count_params, LpLoss, train_fno_domain
from utils.Solver import elastic_beam


torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0) 
################################################################
#  configurations
################################################################
#beam_length = 2.
#beam_width = 0.1

model_data = dict()
model_data["E"] = 200 #200e3
model_data["nu"] = 0.333
#model_data["beam_length"] = beam_length
#model_data["beam_width"] = beam_width

ntrain = 10000
ntest = 200

batch_size = 200
learning_rate = 0.001

epochs = 2000
step_size = 1
gamma = 0.5

modes = 12
width = 32

#if s//2 + 1 < modes:
#    raise ValueError("Warning: modes should be bigger than (s//2+1)")

PATH_t = "database/traction_10201x128.npy"
PATH_m = "database/material_10201x32.npy"
PATH_d = 'database/disp2D_10201x128x32x2.npy'
PATH_dom = 'database/domain_10201x2.npy'

if os.path.exists('persistent_variable.txt'):
    with open('persistent_variable.txt', 'r') as file:
        persistent_variable = int(file.read())
else:
    persistent_variable = 0

persistent_variable += 1

with open('persistent_variable.txt', 'w') as file:
    file.write(str(persistent_variable))

SaveFolder = f"Run_{persistent_variable}/"

if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)
    
with open( SaveFolder + 'data.txt', 'w') as file:
    file.write("ntrain = " + str(ntrain) + "\n")
    file.write("ntest = " + str(ntest) + "\n")
    file.write("batch_size = " + str(batch_size) + "\n")
    file.write("learning_rate = " + str(learning_rate) + "\n")
    file.write("epochs = " + str(epochs) + "\n")
    file.write("modes = " + str(modes) + "\n")
    file.write("width = " + str(width) + "\n")
    file.write("--------------------------------\n")

t_start = time.time()
#################################################################
# generate the data
#################################################################

# load data
traction = np.load(PATH_t)
material = np.load(PATH_m)
disp2D = np.load(PATH_d)
domain = np.load(PATH_dom)

numPtsU = disp2D.shape[1]
numPtsV = disp2D.shape[2]
N_traction = traction.shape[1]
N_material = material.shape[1]

model_data["numPtsU"] = numPtsU
model_data["numPtsV"] = numPtsV
model_data["N_traction"] = N_traction
model_data["N_material"] = N_material

num_refinements = int(math.log(numPtsU, 2) - 4)

# pick n number of data
random_indices = np.random.choice(disp2D.shape[0], size=disp2D.shape[0], replace=False)
traction = traction[random_indices]
material = material[random_indices]
disp2D = disp2D[random_indices]
domain = domain[random_indices]

traction_train = traction[0:ntrain, :]
material_train = material[0:ntrain, :]
disp2D_train = disp2D[0:ntrain, :,:,:]
domain_train = domain[0:ntrain,:]


traction_test = traction[ntrain:ntrain+ntest, :]
material_test = material[ntrain:ntrain+ntest, :]
disp2D_test = disp2D[ntrain:ntrain+ntest, :,:,:]
domain_test = domain[ntrain:ntrain+ntest,:]

# convert input to torch array
traction_train = torch.from_numpy(traction_train.reshape(ntrain, numPtsU, 1, 1)).float().to(device)
material_train = torch.from_numpy(material_train.reshape(ntrain, 1, numPtsV, 1)).float().to(device)
disp2D_train = torch.from_numpy(disp2D_train).float().to(device)
domain_train = torch.from_numpy(domain_train).float().to(device)


traction_test = torch.from_numpy(traction_test.reshape(ntest, numPtsU, 1, 1)).float().to(device)
material_test = torch.from_numpy(material_test.reshape(ntest, 1, numPtsV, 1)).float().to(device)
disp2D_test = torch.from_numpy(disp2D_test).float().to(device)
domain_test = torch.from_numpy(domain_test).float().to(device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(traction_train, material_train, domain_train, 
                                  disp2D_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(traction_test, material_test, domain_test, 
                                  disp2D_test), batch_size=batch_size, shuffle=False)

# model
model = FNO2d_domain(modes, modes, width, model_data).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen-t_start)
################################################################
# training
################################################################
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma)
myloss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_mse_log, train_l2_log, test_l2_log = train_fno_domain(model, train_loader, 
                                                     test_loader, myloss, optimizer,
                                                     scheduler, device, epochs, batch_size)
print("Training time: ", default_timer()-t1)
# plot the convergence of the losses
# plt.semilogy(train_mse_log, label='Train MSE')
plt.semilogy(train_l2_log, label = 'Train L2')
plt.semilogy(test_l2_log, label = 'Test L2')
plt.legend()
plt.savefig( SaveFolder + 'TrainingTrend.png', dpi=300)
# plt.show()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(traction_test, material_test, domain_test, 
                                             disp2D_test), batch_size=1, shuffle=False)
pred = torch.zeros(disp2D_test.shape)
index = 0
x1_test=torch.zeros(traction_test.shape)
x2_test=torch.zeros(material_test.shape)
x3_test=torch.zeros(domain_test.shape)

y_test=torch.zeros(disp2D_test.shape)
test_l2_set = []

with torch.no_grad():
    for x1, x2, x3, y in test_loader:
        x1, x2, x3 ,y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        x1_test[index]=x1
        x2_test[index]=x2
        x3_test[index]=x3
        y_test[index]=y
        out = model(x1,x2,x3)
        pred[index] = out
        test_l2 = myloss(out.view(1, -1), y.view(1, -1)).item()
        test_l2_set.append(test_l2)
        print(index, test_l2)
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())

################################################################
# evaluation
################################################################

#PLotting a random function from the test data generated by GRF
index = 1 #random.randrange(0, ntest)

f = x1_test[index, :, :, :].cpu().numpy()
m = x2_test[index, :, :, :].cpu().numpy()
d = x3_test[index, :].cpu().numpy()

u_exact = y_test[index, :, :, 0].cpu().numpy()
v_exact = y_test[index, :, :, 1].cpu().numpy()

u_pred = pred[index, :, :, 0].cpu().numpy()
v_pred = pred[index, :, :, 1].cpu().numpy()

beam_length = d[0]
beam_width = d[1]

x_test_plot = np.linspace(0, beam_length, numPtsU).astype('float64')
y_test_plot = np.linspace(0, beam_width, numPtsV).astype('float64')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)
x_plot_grid = x_plot_grid.transpose()
y_plot_grid = y_plot_grid.transpose()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, "GRF - u", SaveFolder+"GRF/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, "GRF - v", SaveFolder+"GRF/")

np.save(SaveFolder+"GRF/" + "traction.npy", f)
np.save(SaveFolder+"GRF/" + "u_exact.npy", u_exact)
np.save(SaveFolder+"GRF/" + "v_exact.npy", v_exact)
np.save(SaveFolder+"GRF/" + "u_pred.npy", u_pred)
np.save(SaveFolder+"GRF/" + "v_pred.npy", v_pred)
np.save(SaveFolder+"GRF/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+"GRF/" + "y_plot_grid.npy", y_plot_grid)

# testing f=const.
f_name = "const-dist1"

f = np.ones(x_test_plot.shape)*1e-1
f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

Emod = 200#200e3
nu =  0.333
e0 = 0.5
beam_length = 2.
beam_width = 0.1
d_tensor = torch.tensor([beam_length, beam_width]).unsqueeze(0)

x_test_plot = np.linspace(0, beam_length, numPtsU).astype('float64')
y_test_plot = np.linspace(0, beam_width, numPtsV).astype('float64')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)
x_plot_grid = x_plot_grid.transpose()
y_plot_grid = y_plot_grid.transpose()

new_model_data = dict()
new_model_data["E"] = Emod
new_model_data["nu"] = nu
new_model_data["beam_length"] = beam_length
new_model_data["beam_width"] = beam_width
new_model_data["numPtsU"] = numPtsU
new_model_data["numPtsV"] = numPtsV
new_model_data["N_traction"] = N_traction
new_model_data["N_material"] = N_material

yPhys = y_test_plot - beam_width/2
elasticity_modulus = Emod*(1-e0*np.cos(np.pi*(yPhys/beam_width)))
E_tensor = elasticity_modulus.reshape(1, numPtsV, 1)
E_tensor = torch.from_numpy(E_tensor).float().to(device).unsqueeze(0)


disp_IGA, t_IGA = elastic_beam(f, elasticity_modulus, new_model_data)

t = time.time()
disp_net = model(f_tensor, E_tensor, d_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")
    

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)

"""
l = beam_length
x = x_test_plot
# ---------------------------
# testing f=|sin(x)|
f_name = "sin"

f = abs(np.sin(x*4*np.pi/l))

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")
    

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)

# ---------------------------
# testing f=|cos(x)|
f_name = "cos"

f = abs(np.cos(x*4*np.pi/l))

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)


# ---------------------------
# testing f=linear
f_name = "linear"

f = -x/l+1

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)


# ---------------------------
# testing f=square
f_name = "square"

f = -4*(x/l)**2 + 4*(x/l)

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)


# ---------------------------
# testing f=exponential
f_name = "exponential"

f = -(np.exp(-x)-1)*(np.exp(l-x)-1)/l**2

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)



# ---------------------------
# testing f=reciprocal
f_name = "reciprocal"

f = -(x*(x-l))/(x+l/2)**2

f_tensor = f.reshape(numPtsU, 1, 1)
f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

disp_IGA, t_IGA = elastic_beam(f, model_data, num_refinements)
t = time.time()
disp_net = model(f_tensor)
t_net = time.time() - t

u_exact = disp_IGA[:, :, 0]
v_exact = disp_IGA[:, :, 1]
u_pred = disp_net[0, :, :, 0].detach().cpu().numpy()
v_pred = disp_net[0, :, :, 1].detach().cpu().numpy()

plot_pred_timo(x_plot_grid, y_plot_grid, u_exact, u_pred, f_name + " - u", SaveFolder + f_name + "/")
plot_pred_timo(x_plot_grid, y_plot_grid, v_exact, v_pred, f_name + " - v", SaveFolder + f_name + "/")

with open( SaveFolder + 'data.txt', 'a') as file:
    file.write("-------------" + f_name + "-------------\n")
    file.write("time_IGA = " + str(t_IGA) + "\n")
    file.write("time_net = " + str(t_net) + "\n")

np.save(SaveFolder+f_name + "/" + "traction.npy", f)
np.save(SaveFolder+f_name + "/" + "u_exact.npy", u_exact)
np.save(SaveFolder+f_name + "/" + "v_exact.npy", v_exact)
np.save(SaveFolder+f_name + "/" + "u_pred.npy", u_pred)
np.save(SaveFolder+f_name + "/" + "v_pred.npy", v_pred)
np.save(SaveFolder+f_name + "/" + "x_plot_grid.npy", x_plot_grid)
np.save(SaveFolder+f_name + "/" + "y_plot_grid.npy", y_plot_grid)
"""
