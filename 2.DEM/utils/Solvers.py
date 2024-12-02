

import itertools

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit, value_and_grad
from jax.example_libraries import optimizers
from tqdm import trange
from functools import partial


class Poisson2D:
    def __init__(self, rng_key, layers, activation=jax.nn.swish):
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    
    # OK
    
    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
    
    def u(self, params, x, y):
        # if isinstance(x, jax._src.interpreters.ad.JVPTracer):
        X = jnp.array([[x, y]])
        X = jnp.reshape(X,(1,2))
        # else:
        #     X = jnp.concatenate((x,y),axis=1).astype(data_type)
        u_val = jnp.squeeze(self.apply(params, X))
        return u_val
    
    def u_vec(self, params, x, y):
        u_val = vmap(self.u, (None, 0, 0))(params, x, y)
        return u_val
    
    def dudx(self, params, x, y):
        dudx = jnp.squeeze(grad(self.u, 1)(params, x, y))
        return dudx
    
    def dudy(self, params, x, y):
        dudy = jnp.squeeze(grad(self.u, 2)(params, x, y))
        return dudy
    
    def dudx_vec(self, params, x, y):
        dudx_vec = vmap(self.dudx, (None, 0, 0))(params, x, y)
        return dudx_vec
    
    def dudy_vec(self, params, x, y):
        dudy_vec = vmap(self.dudy, (None, 0, 0))(params, x, y)
        return dudy_vec
            
    def d2udx2(self, params, x, y):
        d2udx2 = grad(self.dudx, 1)(params, x, y)
        return d2udx2
    
    def d2udy2(self, params, x, y):
        d2udy2 = grad(self.dudy, 2)(params, x, y)
        return d2udy2
    
    def d2udx2_vec(self, params, x, y):
        d2udx2_vec = vmap(self.d2udx2, (None, 0, 0))(params, x, y)
        return d2udx2_vec
    
    def d2udy2_vec(self, params, x, y):
        d2udy2_vec = vmap(self.d2udy2, (None, 0, 0))(params, x, y)
        return d2udy2_vec
    
    def loss(self, params, inputs, targets, bnd_inputs, bnd_targets):
        #preds = self.apply(params, inputs)
        x = inputs[:,0:1]
        y = inputs[:,1:2]
        
        d2udx2_pred = self.d2udx2_vec(params, x, y)
        d2udy2_pred = self.d2udy2_vec(params, x, y)
        f_val = -(d2udx2_pred + d2udy2_pred)
        int_residual = f_val - targets
        
        bnd_pred = self.apply(params, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        
        int_loss = jnp.mean(jnp.square(int_residual))
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        loss = int_loss + bnd_loss
        
        return loss
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.loss)(params, *args)
        return loss, grads
    
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, inputs, targets, bnd_inputs, bnd_targets):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, inputs, targets, bnd_inputs, bnd_targets)
        return self.opt_update(i, g, opt_state)
    
    def train(self, inputs, targets, bnd_inputs, bnd_targets, n_iter = 100):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       inputs, targets, bnd_inputs, bnd_targets)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, inputs, targets, bnd_inputs,
                                       bnd_targets)

                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})


class Elasticity2D_DEM: 
    def __init__(self, rng_key, layers, num_epoch, print_epoch, model_data, data_type, activation=jax.nn.swish):
        super(Elasticity2D_DEM, self).__init__()
        self.model_layers = layers
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.Emod = model_data["E"]
        self.nu = model_data["nu"]
        if model_data["state"]=="plane strain":
            self.Emat = self.Emod/((1+self.nu)*(1-2*self.nu))*\
                                    jnp.array([[1-self.nu, self.nu, 0], 
                                               [self.nu, 1-self.nu, 0], 
                                               [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = self.Emod/(1-self.nu**2)*jnp.array([[1, self.nu, 0], 
                                                            [self.nu, 1, 0], 
                                                            [0, 0, (1-self.nu)/2]],dtype=data_type)
            
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []
        
    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    def __call__(self, X, params):
        return jnp.squeeze(self.disp_vec(params, X[:,0:1], X[:,1:2]))
        
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    def disp(self, params, xPhys, yPhys):
        X = jnp.array([[xPhys, yPhys]])
        X = jnp.reshape(X,(1,2))
        X = self.apply(params, X)
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)
        disp = jnp.concatenate([u_val, v_val],1)                       
        return disp
    
    def disp_vec(self, params, x, y):
        disp_vec = vmap(self.disp, (None, 0, 0))(params, x, y)
        return jnp.squeeze(disp_vec)
    
    def grad_disp(self, params, x, y):
        return jax.jacfwd(self.disp, (1, 2))(params, x, y)
    
    def grad_disp_vec(self, params, x, y):
        grad_disp_vec = vmap(self.grad_disp, (None, 0, 0))(params, x, y)
        return grad_disp_vec
    
    # Compute the strains
    def kinematicEq(self, params, xPhys, yPhys):
        disp_dx, disp_dy = self.grad_disp_vec(params, xPhys, yPhys)
        disp_dx = jnp.squeeze(disp_dx)
        disp_dy = jnp.squeeze(disp_dy)

        eps_xx_val = jnp.expand_dims(disp_dx[:,0], axis=-1)
        eps_yy_val = jnp.expand_dims(disp_dy[:,1], axis=-1)
        eps_xy_val = jnp.expand_dims((disp_dx[:,1] + disp_dy[:,0]), axis=-1)

        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    def constitutiveEq(self, params, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(params, xPhys, yPhys)      
        eps_val = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val],1)
        stress_val = jnp.matmul(eps_val, self.Emat)
        stress_xx_val = stress_val[:,0:1]
        stress_yy_val = stress_val[:,1:2]
        stress_xy_val = stress_val[:,2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val
        
    #Custom loss function
    def get_all_losses(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd):
        # calculate the interior loss
        xPhys_int = Xint[:,0:1]
        yPhys_int = Xint[:,1:2]                
        sigma_xx_int, sigma_yy_int, sigma_xy_int = self.constitutiveEq(params, xPhys_int, yPhys_int)
        eps_xx_int, eps_yy_int, eps_xy_int = self.kinematicEq(params, xPhys_int, yPhys_int)
        loss_int = jnp.sum(1/2*(eps_xx_int*sigma_xx_int + eps_yy_int*sigma_yy_int + \
                                      eps_xy_int*sigma_xy_int)*Wint)
        
        # calculate the boundary loss corresponding to the Neumann boundary
        xPhys_bnd = Xbnd[:, 0:1]
        yPhys_bnd = Xbnd[:, 1:2]  
        disp = self.disp_vec(params, xPhys_bnd, yPhys_bnd)    
        u_val_bnd = jnp.expand_dims(disp[:,0], axis=-1)
        v_val_bnd = jnp.expand_dims(disp[:,1], axis=-1)            
        loss_bnd = jnp.sum((u_val_bnd*Ybnd[:, 0:1] + v_val_bnd*Ybnd[:, 1:2])*Wbnd)
        return loss_int, loss_bnd
    
    def get_loss(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd):
        loss_int, loss_bnd = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd)
        return loss_int - loss_bnd
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.get_loss)(params, *args)
        return loss, grads
           
      
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, Xint, Wint, Xbnd, Wbnd, Ybnd):
        params = self.get_params(opt_state)

        g = grad(self.get_loss)(params, Xint, Wint, Xbnd, Wbnd, Ybnd)
        return self.opt_update(i, g, opt_state)
    
    def train(self, Xint, Wint, Xbnd, Wbnd, Ybnd):
        pbar = trange(self.num_epoch)

        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       Xint, Wint, Xbnd, Wbnd, Ybnd)

            if it % self.print_epoch == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.get_loss(params, Xint, Wint, Xbnd, Wbnd, Ybnd)

                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
    

class Elasticity2D_DEM_BC: 
    def __init__(self, rng_key, layers, num_epoch, print_epoch, model_data, data_type, Penalty_weight, activation=jax.nn.swish):
        super(Elasticity2D_DEM_BC, self).__init__()
        self.model_layers = layers
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.Emod = model_data["E"]
        self.nu = model_data["nu"]
        self.Penalty_weight = Penalty_weight
        if model_data["state"]=="plane strain":
            self.Emat = self.Emod/((1+self.nu)*(1-2*self.nu))*\
                                    jnp.array([[1-self.nu, self.nu, 0], 
                                               [self.nu, 1-self.nu, 0], 
                                               [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = self.Emod/(1-self.nu**2)*jnp.array([[1, self.nu, 0], 
                                                            [self.nu, 1, 0], 
                                                            [0, 0, (1-self.nu)/2]],dtype=data_type)
            
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []
        
    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    def __call__(self, X, params):
        return jnp.squeeze(self.disp_vec(params, X[:,0:1], X[:,1:2]))
        
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    def disp(self, params, xPhys, yPhys):
        X = jnp.array([[xPhys, yPhys]])
        X = jnp.reshape(X,(1,2))
        X = self.apply(params, X)
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)
        disp = jnp.concatenate([u_val, v_val],1)                       
        return disp
    
    def disp_vec(self, params, x, y):
        disp_vec = vmap(self.disp, (None, 0, 0))(params, x, y)
        return jnp.squeeze(disp_vec)
    
    def grad_disp(self, params, x, y):
        return jax.jacfwd(self.disp, (1, 2))(params, x, y)
    
    def grad_disp_vec(self, params, x, y):
        grad_disp_vec = vmap(self.grad_disp, (None, 0, 0))(params, x, y)
        return grad_disp_vec
    
    # Compute the strains
    def kinematicEq(self, params, xPhys, yPhys):
        disp_dx, disp_dy = self.grad_disp_vec(params, xPhys, yPhys)
        disp_dx = jnp.squeeze(disp_dx)
        disp_dy = jnp.squeeze(disp_dy)

        eps_xx_val = jnp.expand_dims(disp_dx[:,0], axis=-1)
        eps_yy_val = jnp.expand_dims(disp_dy[:,1], axis=-1)
        eps_xy_val = jnp.expand_dims((disp_dx[:,1] + disp_dy[:,0]), axis=-1)

        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    def constitutiveEq(self, params, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(params, xPhys, yPhys)      
        eps_val = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val],1)
        stress_val = jnp.matmul(eps_val, self.Emat)
        stress_xx_val = stress_val[:,0:1]
        stress_yy_val = stress_val[:,1:2]
        stress_xy_val = stress_val[:,2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val
        
    #Custom loss function
    def get_all_losses(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        # calculate the interior loss
        xPhys_int = Xint[:,0:1]
        yPhys_int = Xint[:,1:2]                
        sigma_xx_int, sigma_yy_int, sigma_xy_int = self.constitutiveEq(params, xPhys_int, yPhys_int)
        eps_xx_int, eps_yy_int, eps_xy_int = self.kinematicEq(params, xPhys_int, yPhys_int)
        loss_int = jnp.sum(1/2*(eps_xx_int*sigma_xx_int + eps_yy_int*sigma_yy_int + \
                                      eps_xy_int*sigma_xy_int)*Wint)
        
        # calculate the boundary loss corresponding to the Neumann boundary
        xPhys_bnd = Xbnd[:, 0:1]
        yPhys_bnd = Xbnd[:, 1:2]  
        disp = self.disp_vec(params, xPhys_bnd, yPhys_bnd)    
        u_val_bnd = jnp.expand_dims(disp[:,0], axis=-1)
        v_val_bnd = jnp.expand_dims(disp[:,1], axis=-1)            
        loss_bnd = jnp.sum((u_val_bnd*Ybnd[:, 0:1] + v_val_bnd*Ybnd[:, 1:2])*Wbnd)
        
        # calculate the boundary loss corresponding to the Dirichlet boundary
        xPhys_bD = XbD[:, 0:1]
        yPhys_bD = XbD[:, 1:2]  
        disp_bD = self.disp_vec(params, xPhys_bD, yPhys_bD)   
        loss_bD = jnp.mean(jnp.square(disp_bD-YbD)*WbD)
        
        return loss_int, loss_bnd, loss_bD
    
    def get_loss(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        #Penalty_weight = 1e9
        loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return loss_int - loss_bnd + self.Penalty_weight*loss_bD
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.get_loss)(params, *args)
        return loss, grads
           
      
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        params = self.get_params(opt_state)

        g = grad(self.get_loss)(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return self.opt_update(i, g, opt_state)
    
    def train(self, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        pbar = trange(self.num_epoch)

        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

            if it % self.print_epoch == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.get_loss(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value, 'loss_int': loss_int, 'loss_bnd': loss_bnd, 'loss_bD': loss_bD})
                
                
class Porous_DEM_BC: 
    def __init__(self, rng_key, layers, num_epoch, print_epoch, model_data, data_type, Penalty_weight, activation=jax.nn.swish):
        super(Porous_DEM_BC, self).__init__()
        self.model_layers = layers
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.E0 = model_data["E"]
        self.nu = model_data["nu"]
        self.state = model_data["state"]
        self.e0 = model_data["e0"]
        self.h = model_data["h"]
        self.porousitystate = model_data["porousitystate"]
        
        self.Penalty_weight = Penalty_weight
        
        if model_data["state"]=="plane strain":
            self.Emat = 1/((1+self.nu)*(1-2*self.nu))*\
                                    jnp.array([[1-self.nu, self.nu, 0], 
                                               [self.nu, 1-self.nu, 0], 
                                               [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = 1/(1-self.nu**2)*jnp.array([[1, self.nu, 0], 
                                                            [self.nu, 1, 0], 
                                                            [0, 0, (1-self.nu)/2]],dtype=data_type)
                    
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []
        
        
    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    def __call__(self, X, params):
        return jnp.squeeze(self.disp_vec(params, X[:,0:1], X[:,1:2]))
        
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    def disp(self, params, xPhys, yPhys):
        X = jnp.array([[xPhys, yPhys]])
        X = jnp.reshape(X,(1,2))
        X = self.apply(params, X)
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)
        disp = jnp.concatenate([u_val, v_val],1)                       
        return disp
    
    def disp_vec(self, params, x, y):
        disp_vec = vmap(self.disp, (None, 0, 0))(params, x, y)
        return jnp.squeeze(disp_vec)
    
    def grad_disp(self, params, x, y):
        return jax.jacfwd(self.disp, (1, 2))(params, x, y)
    
    def grad_disp_vec(self, params, x, y):
        grad_disp_vec = vmap(self.grad_disp, (None, 0, 0))(params, x, y)
        return grad_disp_vec
    
    def PorosityDist(self, yPhys):
        yPhys = yPhys - self.h/2
        if self.porousitystate=="state1":
            E = self.E0*(1-self.e0*jnp.cos(jnp.pi*(yPhys/self.h)))

        elif self.porousitystate =="state2":
            E = self.E0*(1-self.e0*jnp.cos((jnp.pi/2.0*(yPhys/self.h)) + jnp.pi/4))

        elif self.porousitystate =="state3":
            E = self.E0*(1-1/self.e0-1/self.e0*(2/jnp.pi*(jnp.sqrt(1-self.e0)-2/jnp.pi+1))**2)
        return E
    
    # Compute the strains
    def kinematicEq(self, params, xPhys, yPhys):
        disp_dx, disp_dy = self.grad_disp_vec(params, xPhys, yPhys)
        disp_dx = jnp.squeeze(disp_dx)
        disp_dy = jnp.squeeze(disp_dy)

        eps_xx_val = jnp.expand_dims(disp_dx[:,0], axis=-1)
        eps_yy_val = jnp.expand_dims(disp_dy[:,1], axis=-1)
        eps_xy_val = jnp.expand_dims((disp_dx[:,1] + disp_dy[:,0]), axis=-1)

        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    def constitutiveEq(self, params, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(params, xPhys, yPhys)      
        eps = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val],1)
                
        E_dist = self.PorosityDist(yPhys)
        expanded_E = jnp.expand_dims(E_dist, axis=-1)

        expanded_Emat = jnp.expand_dims(self.Emat, axis=0) 
        EPorous = jnp.multiply(expanded_E, expanded_Emat)
        
        eps_val_expand = jnp.expand_dims(eps, axis=-1)

        stress_val = jnp.matmul(EPorous, eps_val_expand)
        
        #stress_val = jnp.matmul(eps_val, self.Emat)
        stress_xx_val = jnp.squeeze(stress_val[:,0:1],axis=2)
        stress_yy_val = jnp.squeeze(stress_val[:,1:2],axis=2)
        stress_xy_val = jnp.squeeze(stress_val[:,2:3],axis=2)
        return stress_xx_val, stress_yy_val, stress_xy_val
        
    #Custom loss function
    def get_all_losses(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        # calculate the interior loss
        xPhys_int = Xint[:,0:1]
        yPhys_int = Xint[:,1:2]                
        sigma_xx_int, sigma_yy_int, sigma_xy_int = self.constitutiveEq(params, xPhys_int, yPhys_int)
        eps_xx_int, eps_yy_int, eps_xy_int = self.kinematicEq(params, xPhys_int, yPhys_int)
        loss_int = jnp.sum(1/2*(eps_xx_int*sigma_xx_int + eps_yy_int*sigma_yy_int + \
                                      eps_xy_int*sigma_xy_int)*Wint)
        
        # calculate the boundary loss corresponding to the Neumann boundary
        xPhys_bnd = Xbnd[:, 0:1]
        yPhys_bnd = Xbnd[:, 1:2]  
        disp = self.disp_vec(params, xPhys_bnd, yPhys_bnd)    
        u_val_bnd = jnp.expand_dims(disp[:,0], axis=-1)
        v_val_bnd = jnp.expand_dims(disp[:,1], axis=-1)            
        loss_bnd = jnp.sum((u_val_bnd*Ybnd[:, 0:1] + v_val_bnd*Ybnd[:, 1:2])*Wbnd)
        
        # calculate the boundary loss corresponding to the Dirichlet boundary
        xPhys_bD = XbD[:, 0:1]
        yPhys_bD = XbD[:, 1:2]  
        disp_bD = self.disp_vec(params, xPhys_bD, yPhys_bD)   
        loss_bD = jnp.mean(jnp.square(disp_bD-YbD)*WbD)
        
        return loss_int, loss_bnd, loss_bD
    
    def get_loss(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        #Penalty_weight = 1e9
        loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return loss_int - loss_bnd + self.Penalty_weight*loss_bD
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.get_loss)(params, *args)
        return loss, grads
           
      
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        params = self.get_params(opt_state)

        g = grad(self.get_loss)(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return self.opt_update(i, g, opt_state)
    
    def train(self, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        pbar = trange(self.num_epoch)

        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

            if it % self.print_epoch == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.get_loss(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss(Porous)': loss_value, 'loss_int': loss_int, 'loss_bnd': loss_bnd, 'loss_bD': loss_bD})

class Porous_DEM_BC_Nonlinear: 
    def __init__(self, rng_key, layers, num_epoch, print_epoch, model_data, data_type, Penalty_weight, activation=jax.nn.swish):
        super(Porous_DEM_BC_Nonlinear, self).__init__()
        self.model_layers = layers
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.data_type = data_type
        self.E0 = model_data["E"]
        self.nu = model_data["nu"]
        self.state = model_data["state"]
        self.e0 = model_data["e0"]
        self.h = model_data["h"]
        self.porousitystate = model_data["porousitystate"]
        
        self.Penalty_weight = Penalty_weight
        
        if model_data["state"]=="plane strain":
            self.Emat = 1/((1+self.nu)*(1-2*self.nu))*\
                                    jnp.array([[1-self.nu, self.nu, 0], 
                                               [self.nu, 1-self.nu, 0], 
                                               [0, 0, (1-2*self.nu)/2]],dtype=data_type)
        elif model_data["state"]=="plane stress":
            self.Emat = 1/(1-self.nu**2)*jnp.array([[1, self.nu, 0], 
                                                            [self.nu, 1, 0], 
                                                            [0, 0, (1-self.nu)/2]],dtype=data_type)
                    
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []
        
        
    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    def __call__(self, X, params):
        return jnp.squeeze(self.disp_vec(params, X[:,0:1], X[:,1:2]))
        
    def dirichletBound(self, X, xPhys, yPhys):
        u_val = X[:,0:1]
        v_val = X[:,1:2]
        return u_val, v_val
          
    # Running the model
    def disp(self, params, xPhys, yPhys):
        X = jnp.array([[xPhys, yPhys]])
        X = jnp.reshape(X,(1,2))
        X = self.apply(params, X)
        u_val, v_val = self.dirichletBound(X, xPhys, yPhys)
        disp = jnp.concatenate([u_val, v_val],1)                       
        return disp
    
    def disp_vec(self, params, x, y):
        disp_vec = vmap(self.disp, (None, 0, 0))(params, x, y)
        return jnp.squeeze(disp_vec)
    
    def grad_disp(self, params, x, y):
        return jax.jacfwd(self.disp, (1, 2))(params, x, y)
    
    def grad_disp_vec(self, params, x, y):
        grad_disp_vec = vmap(self.grad_disp, (None, 0, 0))(params, x, y)
        return grad_disp_vec
    
    def PorosityDist(self, yPhys):
        yPhys = yPhys - self.h/2
        if self.porousitystate=="state1":
            E = self.E0*(1-self.e0*jnp.cos(jnp.pi*(yPhys/self.h)))

        elif self.porousitystate =="state2":
            E = self.E0*(1-self.e0*jnp.cos((jnp.pi/2.0*(yPhys/self.h)) + jnp.pi/4))

        elif self.porousitystate =="state3":
            E = self.E0*(1-1/self.e0-1/self.e0*(2/jnp.pi*(jnp.sqrt(1-self.e0)-2/jnp.pi+1))**2)
        return E
    
    # Compute the strains
    def kinematicEq(self, params, xPhys, yPhys):
        disp_dx, disp_dy = self.grad_disp_vec(params, xPhys, yPhys)
        disp_dx = jnp.squeeze(disp_dx)
        disp_dy = jnp.squeeze(disp_dy)

        eps_xx_val = jnp.expand_dims(disp_dx[:,0] + 1/2*disp_dy[:,0]**2, axis=-1)
        eps_yy_val = jnp.expand_dims(disp_dy[:,1], axis=-1)
        eps_xy_val = jnp.expand_dims((disp_dx[:,1] + disp_dy[:,0]), axis=-1)

        return eps_xx_val, eps_yy_val, eps_xy_val
    
    # Compute the stresses
    def constitutiveEq(self, params, xPhys, yPhys):
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(params, xPhys, yPhys)      
        eps = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val],1)
                
        E_dist = self.PorosityDist(yPhys)
        expanded_E = jnp.expand_dims(E_dist, axis=-1)

        expanded_Emat = jnp.expand_dims(self.Emat, axis=0) 
        EPorous = jnp.multiply(expanded_E, expanded_Emat)
        
        eps_val_expand = jnp.expand_dims(eps, axis=-1)

        stress_val = jnp.matmul(EPorous, eps_val_expand)
        
        #stress_val = jnp.matmul(eps_val, self.Emat)
        stress_xx_val = jnp.squeeze(stress_val[:,0:1],axis=2)
        stress_yy_val = jnp.squeeze(stress_val[:,1:2],axis=2)
        stress_xy_val = jnp.squeeze(stress_val[:,2:3],axis=2)
        return stress_xx_val, stress_yy_val, stress_xy_val
        
    #Custom loss function
    def get_all_losses(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        # calculate the interior loss
        xPhys_int = Xint[:,0:1]
        yPhys_int = Xint[:,1:2]                
        sigma_xx_int, sigma_yy_int, sigma_xy_int = self.constitutiveEq(params, xPhys_int, yPhys_int)
        eps_xx_int, eps_yy_int, eps_xy_int = self.kinematicEq(params, xPhys_int, yPhys_int)
        loss_int = jnp.sum(1/2*(eps_xx_int*sigma_xx_int + eps_yy_int*sigma_yy_int + \
                                      eps_xy_int*sigma_xy_int)*Wint)
        
        # calculate the boundary loss corresponding to the Neumann boundary
        xPhys_bnd = Xbnd[:, 0:1]
        yPhys_bnd = Xbnd[:, 1:2]  
        disp = self.disp_vec(params, xPhys_bnd, yPhys_bnd)    
        u_val_bnd = jnp.expand_dims(disp[:,0], axis=-1)
        v_val_bnd = jnp.expand_dims(disp[:,1], axis=-1)            
        loss_bnd = jnp.sum((u_val_bnd*Ybnd[:, 0:1] + v_val_bnd*Ybnd[:, 1:2])*Wbnd)
        
        # calculate the boundary loss corresponding to the Dirichlet boundary
        xPhys_bD = XbD[:, 0:1]
        yPhys_bD = XbD[:, 1:2]  
        disp_bD = self.disp_vec(params, xPhys_bD, yPhys_bD)   
        loss_bD = jnp.mean(jnp.square(disp_bD-YbD)*WbD)
        
        return loss_int, loss_bnd, loss_bD
    
    def get_loss(self, params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        #Penalty_weight = 1e9
        loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return loss_int - loss_bnd + self.Penalty_weight*loss_bD
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.get_loss)(params, *args)
        return loss, grads
           
      
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        params = self.get_params(opt_state)

        g = grad(self.get_loss)(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
        return self.opt_update(i, g, opt_state)
    
    def train(self, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD):
        pbar = trange(self.num_epoch)

        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)

            if it % self.print_epoch == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.get_loss(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                loss_int, loss_bnd, loss_bD = self.get_all_losses(params, Xint, Wint, Xbnd, Wbnd, Ybnd, XbD, WbD, YbD)
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss(Porous)': loss_value, 'loss_int': loss_int, 'loss_bnd': loss_bnd, 'loss_bD': loss_bD})