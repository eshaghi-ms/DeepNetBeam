"""
2D linear elasticity example
Solve the equilibrium equation -\nabla \cdot \sigma(x,y) = f(x,y) for x, y \in \Omega 
with the strain-displacement equation:
    \epsilon = 1/2(\nabla u + \nabla u^T)
and the constitutive law:
    \sigma = 2*\mu*\epsilon + \lambda*(\nabla\cdot u)I,
where \mu and \lambda are Lame constants, I is the identity tensor.
Dirichlet boundary conditions: u(x)=\hat{u} for x\in\Gamma_D
Neumann boundary conditions: \sigma n = \hat{t} for x\in \Gamma_N,
where n is the normal vector.
For this example:
    \Omega is a rectangle with corners at  (0,0) and (8,2)
   Dirichlet boundary conditions for x=0:
           u(x,y) = P/(6*E*I)*y*((2+nu)*(y^2-W^2/4))
           v(x,y) = -P/(6*E*I)*(3*nu*y^2*L)
    and parabolic traction at x=8
           p(x,y) = P*(y^2 - y*W)/(2*I)
    where P=2 is the maxmimum traction
          E = 1e3 is Young's modulus
          nu = 0.25 is the Poisson ratio
          I = W^3/12 is second moment of area of the cross-section

Use PINNs
implemented in Pytorch
"""

import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
import time
import matplotlib as mpl

#check if GPU is available

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")
    torch.set_default_tensor_type('torch.FloatTensor')

#make figures bigger on HiDPI monitors
mpl.rcParams['figure.dpi'] = 300

# fix random seeds
npr.seed(42)
torch.manual_seed(42)
# define material parameters
E = 1000.
nu = 0.25

# dimensions of the beam
Length = 8.
Width = 2.

# define domain and collocation points
x_dom = 0., Length, 81
y_dom = 0., Width, 41
# create points
lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
dom = np.zeros((x_dom[2]*y_dom[2],2))
c = 0
for x in np.nditer(lin_x):
    tb = y_dom[2]*c
    te = tb + y_dom[2]
    c += 1
    dom[tb:te,0] = x
    dom[tb:te,1] = lin_y

# define constitutive matrix
C = torch.tensor([[E/(1-nu**2), E*nu/(1-nu**2), 0],[E*nu/(1-nu**2), E/(1-nu**2), 0],[0, 0, E/(2*(1+nu))]])

# define boundary conditions
# penalty parameters for BCs
bc_d_tol = 1e2
bc_n_tol = 1e0
# Dirichlet x, y, dir, val
bc_d1 = []    
bc_d2 = []
bc_d2_pts_idx = np.where(dom[:,0] == 0)
bc_d2_pts = dom[bc_d2_pts_idx,:][0]

inert=Width**3/12
P = 2
pei=P/(6*E*inert)

def compExactDisplacement(x, y):
    y_temp = y - Width/2  # move (0,0) to below left corner
    x_disp = pei*y_temp*((6*Length-3*x)*x+(2+nu)*(y_temp**2-Width**2/4));
    y_disp =-pei*(3*nu*y_temp**2*(Length-x)+(4+5*nu)*Width**2*x/4+(3*Length-x)*x**2);
    return x_disp, y_disp

for (x, y) in bc_d2_pts:    
    x_disp, y_disp = compExactDisplacement(x, y)
    bc_d2.append((np.array([x, y]), 1, y_disp))
    bc_d1.append((np.array([x, y]), 0, x_disp))


# von Neumann boundary conditions
# applied force on the right edge
bc_n1 = []
bc_n1_pts_idx = np.where(dom[:,0] == Length)
bc_n1_pts = dom[bc_n1_pts_idx,:][0]
for (x, y) in bc_n1_pts:
    y_temp = y - Width/2  #move (0,0) to below left corner 
    trac = P*(y_temp**2-Width**2/4)/2/inert
    bc_n1.append((np.array([x, y]), 0, trac))

# zero force on the bottom edge
bc_n2 = []
bc_n2_pts_idx = np.where(dom[:,1] == 0)
bc_n2_pts = dom[bc_n2_pts_idx,:][0]
for (x, y) in bc_n2_pts:    
    trac = 0
    bc_n2.append((np.array([x, y]), 1, trac))
    
# zero force on the top edge
bc_n3 = []
bc_n3_pts_idx = np.where(dom[:,1] == Width)
bc_n3_pts = dom[bc_n3_pts_idx,:][0]
for (x, y) in bc_n3_pts:    
    trac = 0
    bc_n3.append((np.array([x, y]), 1, trac)) 
    

# convert numpy BCs to torch
def ConvBCsToTensors(bc_d):
    size_in_1 = len(bc_d)
    size_in_2 = len(bc_d[0][0])
    bc_in = torch.empty(size_in_1, size_in_2, device=dev)
    c = 0
    for bc in bc_d:
        bc_in[c,:] = torch.from_numpy(bc[0])
        c += 1
    return bc_in

# mean squared loss function   
def MyLossSum(tinput):
    return torch.sum(tinput) / tinput.data.nelement()
    
def MyLossSquaredSum(tinput, target):
    return torch.sum((tinput - target) ** 2) / tinput.data.nelement()    

# mean squared loss function
def mse_loss(tinput, target):
    return torch.sum((tinput - target) ** 2) / tinput.data.nelement()

# Pytorch neural network
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y1 = torch.tanh(self.linear1(x.float()))
        y = self.linear4(y1)
        return y
    
# D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 2, 100, 2

# Construct our model by instantiating the class defined above
model = MultiLayerNet(D_in, H, D_out)
model.float().to(dev)

# prepare inputs and outputs for training the model

# inputs
x = torch.from_numpy(dom).float()
x = x.to(dev)
x.requires_grad_(True)
N = x.size()[0]
# get tensor inputs and outputs for boundary conditions
# Dirichlet
# boundary 1
bc_d1_x = ConvBCsToTensors(bc_d1)
bc_d1_y = torch.from_numpy(np.asarray([i[2] for i in bc_d1])).float().to(dev)
# boundary 2
bc_d2_x = ConvBCsToTensors(bc_d2)
bc_d2_y = torch.from_numpy(np.asarray([i[2] for i in bc_d2])).float().to(dev)
# von Neumann
# applied forces
bc_n1_x = ConvBCsToTensors(bc_n1)
bc_n1_x.requires_grad_(True)
bc_n1_y = torch.from_numpy(np.asarray([i[2] for i in bc_n1])).float().to(dev)

bc_n2_x = ConvBCsToTensors(bc_n2)
bc_n2_x.requires_grad_(True)
bc_n2_y = torch.from_numpy(np.asarray([i[2] for i in bc_n2])).float().to(dev)

bc_n3_x = ConvBCsToTensors(bc_n3)
bc_n3_x.requires_grad_(True)
bc_n3_y = torch.from_numpy(np.asarray([i[2] for i in bc_n3])).float().to(dev)

# prepare inputs for testing the model
# define domain
x_dom_test = 0., Length, 81
y_dom_test = 0., Width, 21
# create points
x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])

# Plate in plane stress
def KinematicEquation(primal_pred, x):
    d_primal_x_pred = grad(primal_pred[:,0].unsqueeze(1),x,torch.ones(x.size()[0], 1, device=dev),create_graph=True,retain_graph=True)[0]
    d_primal_y_pred = grad(primal_pred[:,1].unsqueeze(1),x,torch.ones(x.size()[0], 1, device=dev),create_graph=True,retain_graph=True)[0]
    eps_x = d_primal_x_pred[:, 0].unsqueeze(1)
    eps_y = d_primal_y_pred[:, 1].unsqueeze(1)
    eps_xy = d_primal_x_pred[:, 1].unsqueeze(1) + d_primal_y_pred[:, 0].unsqueeze(1)
    eps = torch.cat((eps_x, eps_y, eps_xy), 1)
    return eps


def ConstitutiveEquation(eps_pred, C):
    sig = torch.mm(eps_pred, C)
    return sig

def BalanceEquation(sig_pred, x):
    d_sig_x_pred = grad(sig_pred[:,0].unsqueeze(1),x,torch.ones(x.size()[0], 1, device=dev),create_graph=True,retain_graph=True)[0]
    d_sig_y_pred = grad(sig_pred[:,1].unsqueeze(1),x,torch.ones(x.size()[0], 1, device=dev),create_graph=True,retain_graph=True)[0]
    d_sig_xy_pred = grad(sig_pred[:,2].unsqueeze(1),x,torch.ones(x.size()[0], 1, device=dev),create_graph=True,retain_graph=True)[0]
    dsig_dx = d_sig_x_pred[:, 0].unsqueeze(1) + d_sig_xy_pred[:, 1].unsqueeze(1)
    dsig_dy = d_sig_y_pred[:, 1].unsqueeze(1) + d_sig_xy_pred[:, 0].unsqueeze(1)
    return torch.cat((dsig_dx, dsig_dy), 1)

# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20)
 
domain_area = Length*Width
start_time = time.time()

loss_values = []
  
for t in range(250):
    # Zero gradients, perform a backward pass, and update the weights.  
    
    def closure():
        it_time = time.time()
        # prediction of primary variables
        primal_pred = model(x)
        primal_pred.float()
        # evaluate kinematic equations
        eps_pred = KinematicEquation(primal_pred, x)
        # evaluate constitutive equations
        sig_pred = ConstitutiveEquation(eps_pred, C)  
        
        # evaluate balance equations - y_pred for training the model in the domain
        y_pred = BalanceEquation(sig_pred, x)
        #print(y_pred)
        y = torch.zeros_like(y_pred)
        dom_crit = MyLossSquaredSum (y_pred,y)
        #print(dom_crit)               
        # treat boundary conditions
        # Dirichlet boundary conditions
        # boundary 1 x - direction
        bc_d1_pred = model(bc_d1_x)
        bc_d1_pred.float()
        bc_d1_crit = MyLossSquaredSum(bc_d1_pred[:,bc_d1[0][1]],bc_d1_y)
        # boundary 2 y - direction
        bc_d2_pred = model(bc_d2_x)
        bc_d2_pred.float()
        bc_d2_crit = MyLossSquaredSum(bc_d2_pred[:,bc_d2[0][1]],bc_d2_y)
        # von Neumann boundary conditions
        # boundary 1 (right edge)
        bc_n1_primal = model(bc_n1_x)
        bc_n1_primal.float()
        # print('bc_n1_primal:', bc_n1_primal)
        bc_n1_eps = KinematicEquation(bc_n1_primal, bc_n1_x)
        # print('bc_n1_eps: ', bc_n1_eps)
        bc_n1_sig = ConstitutiveEquation(bc_n1_eps, C)
        # print('bc_n1_sig: ', bc_n1_sig)
        bc_n1_pred = torch.cat((bc_n1_sig[:, 0], bc_n1_sig[:, 2]))
        # print('bc_n1_pred: ', bc_n1_pred)
        bc_n1_crit = MyLossSquaredSum(bc_n1_pred, torch.cat((torch.zeros_like(bc_n1_y),bc_n1_y)))
        
        # print(torch.cat((torch.zeros_like(bc_n1_y),bc_n1_y)))
        
        # boundary 2 (bottom edge)
        bc_n2_primal = model(bc_n2_x)
        bc_n2_eps = KinematicEquation(bc_n2_primal, bc_n2_x)
        bc_n2_sig = ConstitutiveEquation(bc_n2_eps, C)
        bc_n2_pred = torch.cat((bc_n2_sig[:, 1], bc_n2_sig[:, 2]))
        bc_n2_crit = MyLossSquaredSum(bc_n2_pred, torch.cat((torch.zeros_like(bc_n2_y),bc_n2_y)))
        
        # boundary 3 (top edge)
        bc_n3_primal = model(bc_n3_x)
        bc_n3_eps = KinematicEquation(bc_n3_primal, bc_n3_x)
        bc_n3_sig = ConstitutiveEquation(bc_n3_eps, C)
        bc_n3_pred = torch.cat((bc_n3_sig[:, 1], bc_n3_sig[:, 2]))
        bc_n3_crit = MyLossSquaredSum(bc_n3_pred, torch.cat((torch.zeros_like(bc_n3_y),bc_n3_y)))

        # Compute and print loss
        neumann_loss =  (bc_n1_crit + bc_n2_crit + bc_n3_crit)*bc_n_tol
        boundary_loss = ( bc_d1_crit + bc_d2_crit )  * bc_d_tol + neumann_loss
        loss = dom_crit + boundary_loss
        optimizer.zero_grad()
        loss.backward()
        loss_values.append(loss.item())
        print('Iter: %d Loss: %.9e Interior: %.9e Boundary: %.9e Neumann: %.9e Time: %.3e'
              %(t+1, loss.item(), dom_crit.item(), boundary_loss.item(), neumann_loss.item(), time.time()-it_time))
        return loss
    optimizer.step(closure)

elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

# plot results
oShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
oShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceUx = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceUy = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))

# magnification factors for plotting the deformed shape
x_fac = 1
y_fac = 1

# compute the approximate displacements at plot points
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        t_tensor = torch.tensor([x,y]).unsqueeze(0)
        tRes = model(t_tensor).detach().cpu().numpy()[0]
        oShapeX[i][j] = x
        oShapeY[i][j] = y
        surfaceUx[i][j] = tRes[0]
        surfaceUy[i][j] = tRes[1]
        defShapeX[i][j] = x +  tRes[0] * x_fac
        defShapeY[i][j] = y + tRes[1] * y_fac

def plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY):
    fig, axes = plt.subplots(nrows=2)
    cs1 = axes[0].contourf(defShapeX, defShapeY, surfaceUx, 255, cmap=cm.jet)
    cs2 = axes[1].contourf(defShapeX, defShapeY, surfaceUy, 255, cmap=cm.jet)
    fig.colorbar(cs1, ax=axes[0])
    fig.colorbar(cs2, ax=axes[1])
    axes[0].set_title("Displacement in x")
    axes[1].set_title("Displacement in y")
    fig.tight_layout()
    for tax in axes:
        tax.set_xlabel('$x$')
        tax.set_ylabel('$y$')
    plt.show()
    
plotDeformedDisp(surfaceUx, surfaceUy, defShapeX, defShapeY)

#compute the error in the approximate displacements at plot points
oShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
oShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceErrUx = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceErrUy = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceExUx = np.zeros((x_dom_test[2], y_dom_test[2]))
surfaceExUy = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeX = np.zeros((x_dom_test[2], y_dom_test[2]))
defShapeY = np.zeros((x_dom_test[2], y_dom_test[2]))


err_norm = 0
ex_norm = 0
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        t_tensor = torch.tensor([x,y]).unsqueeze(0)
        tRes = model(t_tensor).detach().cpu().numpy()[0]
        oShapeX[i][j] = x
        oShapeY[i][j] = y
        
        x_disp, y_disp = compExactDisplacement(x, y)    
        
        #print(tRes[0], x_disp, tRes[1], y_disp)
        surfaceErrUx[i][j] = x_disp-tRes[0]
        surfaceErrUy[i][j] = y_disp-tRes[1]
        
        surfaceExUx[i][j] = x_disp
        surfaceExUy[i][j] = y_disp
        
        err_norm += (x_disp-tRes[0])**2 + (y_disp-tRes[1])**2
        ex_norm += x_disp**2 + y_disp**2
        

error_u = np.sqrt(err_norm/ex_norm)   
print("Error plots")
plotDeformedDisp(surfaceErrUx, surfaceErrUy, oShapeX, oShapeY)

print("Relative L2 error: ", error_u)

# plot the errors on the left boundary
y_space = y_space[np.newaxis]
plt.plot(y_space.T, surfaceUx[0], y_space.T, surfaceExUx[0])
plt.show()
plt.plot(y_space.T, surfaceErrUx[0])
plt.show()


plt.plot(y_space.T, surfaceUy[0], y_space.T, surfaceExUy[0])
plt.show()
plt.plot(y_space.T, surfaceErrUy[0])
plt.show()

print('Loss convergence')
plt.semilogy(loss_values)
