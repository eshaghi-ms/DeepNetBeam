#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:18:45 2023


@author: mohammadsadegh
"""

import os
import PorousFunctions


beam_width = 0.1
pressure = 1e-1
E = 200e3
nu = 0.333
state = "plane stress" # {"plane strain", "plane stress"}


penalty_weight = 1e9

numElem = [20, 10, 4]
num_epoch = [2000, 1000]
layers = [2, 20, 20, 20, 2]

#porosity_stateS = ["state1", "state2"]
#BCstateS = ["C-C", "C-H", "H-H"]
#porosityS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#beam_lengthS = [0.1, 0.2, 0.3 ,0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

porosity_stateS = ["state1"]
BCstateS = ["C-C"]
porosityS = [0]
beam_lengthS = [0.1]


try:
    os.mkdir('Data')
except FileExistsError:
    pass    


for porosity_state in porosity_stateS:
    for BCstate in BCstateS:
        for porosity in porosityS:
            for beam_length in beam_lengthS:
                
                print(f"porosity_state = {porosity_state}")
                print(f"BCstate = {BCstate}")
                print(f"porosity = {porosity}")
                print(f"beam_length = {beam_length}")
                
                xPhys, yPhys, u, w, eps_xx, eps_yy, eps_xy, stress_xx, stress_yy, stress_xy, loss = \
                PorousFunctions.Porous(beam_length, beam_width, pressure, E, nu, state, porosity, \
                                       porosity_state, BCstate, penalty_weight, numElem, num_epoch, layers)
                    
                file_path = "Data/" + BCstate + "_" + porosity_state + "_" + str(porosity) + "_" + str(beam_length) 
                
                with open((file_path + 'xPhys.txt'), "w") as file:
                    for value in xPhys.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'yPhys.txt'), "w") as file:
                    for value in yPhys.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'u.txt'), "w") as file:
                    for value in u.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'w.txt'), "w") as file:
                    for value in w.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'eps_xx.txt'), "w") as file:
                    for value in eps_xx.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'eps_yy.txt'), "w") as file:
                    for value in eps_yy.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'eps_xy.txt'), "w") as file:
                    for value in eps_xy.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'stress_xx.txt'), "w") as file:
                    for value in stress_xx.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'stress_yy.txt'), "w") as file:
                    for value in stress_yy.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'stress_xy.txt'), "w") as file:
                    for value in stress_xy.flatten():
                        file.write(str(value) + "\n")
                        
                with open((file_path + 'loss.txt'), "w") as file:
                    for value in loss.flatten():
                        file.write(str(value) + "\n")
                        
                