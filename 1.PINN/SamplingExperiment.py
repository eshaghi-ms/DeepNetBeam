"""
The code is related to sampling experiment in the section 3.1. Cantilever Beam

For sampling strategy of choosing the collocation points for
the enforcement of physical constraints, we have investigated
six methods as follows and the examples of 400 points
generated in [0,2]×[0,8] using the methods are shown in Fig.2.

•1. Equispaced Uniform Grid (Grid): Residual points are chosen at evenly spaced intervals across the
computational domain, forming a uniform grid.
•2. Gaussian Quadrature Points (Quadrature): Residual points are selected based on the famous n-point
Gaussian quadrature rule, which is constructed to yield an exact result for the integral of polynomials of degree
2n −1 or less by a suitable choice of the nodes (in the current work n = 4).
•3. Latin Hypercube Sampling (LHS)[63]: A Monte Carlo method that generates random samples within
intervals based on equal probability, ensuring that the samples are normally distributed within each range.
•4. Halton Sequence (Halton)[64]: Samples are generated by reversing or flipping the base conversion of
numbers using prime bases.
•5. Hammersley Sequence (Hammersley)[65]: Similar to the Halton sequence, but with points in the first
dimension spaced equidistantly.
•6. Sobol Sequence (Sobol)[66]: A base-2 digital sequence that distributes points in a highly uniform
manner.

"""
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import scipy.optimize

from utils.tfp_loss import tfp_function_factory
from utils.scipy_loss import scipy_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Elasticity2D_coll_dist
from utils.Plotting import plot_pts, plot_convergence_semilog
# make figures bigger on HiDPI monitors
import matplotlib as mpl
from skopt.sampler import Sobol, Halton, Hammersly, Lhs

mpl.rcParams['figure.dpi'] = 200
np.random.seed(42)
tf.random.set_seed(42)


class Elast_TimoshenkoBeam(Elasticity2D_coll_dist):
    """
    Class including the boundary conditions for the Timoshenko beam problem
    """

    def __init__(self, layers, train_op, num_epoch, print_epoch, model_data, data_type):
        super().__init__(layers, train_op, num_epoch, print_epoch, model_data, data_type)

    # @tf.function
    def dirichletBound(self, X, xPhys, yPhys):
        # multiply by x,y for strong imposition of boundary conditions
        u_val = X[:, 0:1]
        v_val = X[:, 1:2]
        self.W = 2.0
        self.L = 8.0
        self.I = self.W ** 3 / 12
        self.P = 2.0
        self.pei = self.P / (6 * self.Emod * self.I)

        y_temp = yPhys - self.W / 2
        u_left = self.pei * y_temp * ((2 + self.nu) * (y_temp ** 2 - self.W ** 2 / 4))
        v_left = -self.pei * (3 * self.nu * y_temp ** 2 * self.L)

        u_val = xPhys * u_val + u_left
        v_val = xPhys * v_val + v_left

        return u_val, v_val


# define the exact displacements
def exact_disp(x, y, model_data):
    E = model_data["E"]
    nu = model_data["nu"]
    beam_width = model_data["beam_width"]
    beam_length = model_data["beam_length"]
    pressure = model_data["pressure"]
    inert = beam_width ** 3 / 12
    pei = pressure / (6 * E * inert)
    y_temp = y - beam_width / 2  # move (0,0) to below left corner
    x_disp = pei * y_temp * ((6 * beam_length - 3 * x) * x + (2 + nu) * (y_temp ** 2 - beam_width ** 2 / 4))
    y_disp = -pei * (3 * nu * y_temp ** 2 * (beam_length - x) + (4 + 5 * nu) * beam_width ** 2 * x / 4 + (
            3 * beam_length - x) * x ** 2)
    return x_disp, y_disp


def main(method):

    # define the input and output data set
    beam_length = 8.
    beam_width = 2.
    domainCorners = np.array([[0., 0.], [0, beam_width], [beam_length, 0.], [beam_length, beam_width]])
    geomDomain = Quadrilateral(domainCorners)

    numPtsU = 40
    numPtsV = 10
    data_type = "float64"

    if method == 'Grid':
        xPhys, yPhys = geomDomain.getUnifIntPts(numPtsU,numPtsV,[0,0,0,0])
    elif method == 'Quadrature':
        xPhys, yPhys, Wint = geomDomain.getQuadIntPts(numPtsU//4, numPtsV//4, 4)
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        samplers = {"Sobol": Sobol(), "Halton": Halton(), "Hammersley": Hammersly(), "LHS": Lhs()}
        n_samples = numPtsV * numPtsU
        bounds = [(0, beam_length), (0, beam_width)]
        sampler = samplers[method]
        samples = np.array(sampler.generate(bounds, n_samples))
        xPhys, yPhys = samples[:, 0:1], samples[:, 1:2]

    Xint = np.concatenate((xPhys, yPhys), axis=1).astype(data_type)
    Yint = np.zeros_like(Xint).astype(data_type)

    # bottom boundary, include both x and y directions
    xPhysBndB, yPhysBndB, xNormB, yNormB = geomDomain.getUnifEdgePts(numPtsU, numPtsV, [1, 0, 0, 0])
    dirB0 = np.zeros_like(xPhysBndB)
    dirB1 = np.ones_like(xPhysBndB)
    XbndB0 = np.concatenate((xPhysBndB, yPhysBndB, xNormB, yNormB, dirB0), axis=1).astype(data_type)
    XbndB1 = np.concatenate((xPhysBndB, yPhysBndB, xNormB, yNormB, dirB1), axis=1).astype(data_type)

    # boundary for x=beam_length, include both the x and y directions
    xPhysBndC, yPhysBndC, xNormC, yNormC = geomDomain.getUnifEdgePts(numPtsU, numPtsV, [0, 1, 0, 0])
    dirC0 = np.zeros_like(xPhysBndC)
    dirC1 = np.ones_like(xPhysBndC)
    XbndC0 = np.concatenate((xPhysBndC, yPhysBndC, xNormC, yNormC, dirC0), axis=1).astype(data_type)
    XbndC1 = np.concatenate((xPhysBndC, yPhysBndC, xNormC, yNormC, dirC1), axis=1).astype(data_type)

    # boundary for y=beam_width, include both the x and y direction
    xPhysBndD, yPhysBndD, xNormD, yNormD = geomDomain.getUnifEdgePts(numPtsU, numPtsV, [0, 0, 1, 0])
    dirD0 = np.zeros_like(xPhysBndD)
    dirD1 = np.ones_like(xPhysBndD)
    XbndD0 = np.concatenate((xPhysBndD, yPhysBndD, xNormD, yNormD, dirD0), axis=1).astype(data_type)
    XbndD1 = np.concatenate((xPhysBndD, yPhysBndD, xNormD, yNormD, dirD1), axis=1).astype(data_type)

    # concatenate all the boundaries
    Xbnd = np.concatenate((XbndB0, XbndB1, XbndC0, XbndC1, XbndD0, XbndD1), axis=0)

    # plot the collocation points

    plot_pts(Xint, Xbnd[:, 0:2], 1, method)

    pressure = 2
    model_data = dict()
    model_data["E"] = 1e3  # 1e3
    model_data["nu"] = 0.25
    model_data["state"] = "plane stress"
    model_data["beam_length"] = beam_length
    model_data["beam_width"] = beam_width
    model_data["pressure"] = pressure

    # define loading
    YbndB0 = np.zeros_like(xPhysBndB).astype(data_type)
    YbndB1 = np.zeros_like(xPhysBndB).astype(data_type)
    YbndC0 = np.zeros_like(xPhysBndC).astype(data_type)
    inert = beam_width ** 3 / 12
    YbndC1 = (pressure * (yPhysBndC ** 2 - yPhysBndC * beam_width) / (2 * inert)).astype(data_type)

    YbndD0 = np.zeros_like(xPhysBndD).astype(data_type)
    YbndD1 = np.zeros_like(xPhysBndD).astype(data_type)
    Ybnd = np.concatenate((YbndB0, YbndB1, YbndC0, YbndC1, YbndD0, YbndD1), axis=0)

    # define the model
    tf.keras.backend.set_floatx(data_type)
    l1 = tf.keras.layers.Dense(20, "swish")
    l2 = tf.keras.layers.Dense(20, "swish")
    l3 = tf.keras.layers.Dense(20, "swish")
    l4 = tf.keras.layers.Dense(2, None)
    train_op = tf.keras.optimizers.Adam()
    train_op2 = "TFP-BFGS"
    num_epoch = 1000
    print_epoch = 100
    pred_model = Elast_TimoshenkoBeam([l1, l2, l3, l4], train_op, num_epoch,
                                      print_epoch, model_data, data_type)

    # convert the training data to tensors
    Xint_tf = tf.convert_to_tensor(Xint)
    Yint_tf = tf.convert_to_tensor(Yint)
    Xbnd_tf = tf.convert_to_tensor(Xbnd)
    Ybnd_tf = tf.convert_to_tensor(Ybnd)

    # training
    t0 = time.time()

    pred_model.network_learn(Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
    t1 = time.time()

    if train_op2 == "SciPy-LBFGS-B":
        loss_func = scipy_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
        init_params = np.float64(tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables).numpy())
        results = scipy.optimize.minimize(fun=loss_func, x0=init_params, jac=True, method='L-BFGS-B',
                                          options={'disp': None, 'maxls': 50, 'iprint': -1,
                                                   'gtol': 1e-6, 'eps': 1e-6, 'maxiter': 50000, 'ftol': 1e-6,
                                                   'maxcor': 50, 'maxfun': 50000})

        loss_func.assign_new_model_parameters(results.x)
    else:
        loss_func = tfp_function_factory(pred_model, Xint_tf, Yint_tf, Xbnd_tf, Ybnd_tf)
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(loss_func.idx, pred_model.trainable_variables)
        # train the model with L-BFGS solver
        results = tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=loss_func, initial_position=init_params,
            max_iterations=1000, tolerance=1e-6)
            #max_iterations = 1000, tolerance = 1e-6)
        loss_func.assign_new_model_parameters(results.position)
    t2 = time.time()

    # L2 error for train data
    numPtsUTest = 1 * numPtsU
    numPtsVTest = 1 * numPtsV
    xPhysTest, yPhysTest = geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1, 1, 1, 1])
    XTest = np.concatenate((xPhysTest, yPhysTest), axis=1).astype(data_type)
    XTest_tf = tf.convert_to_tensor(XTest)
    YTest = pred_model(XTest_tf).numpy()

    xPhysTest2D = np.resize(XTest[:, 0], [numPtsVTest, numPtsUTest])
    yPhysTest2D = np.resize(XTest[:, 1], [numPtsVTest, numPtsUTest])
    YTest2D_x = np.resize(YTest[:, 0], [numPtsVTest, numPtsUTest])
    YTest2D_y = np.resize(YTest[:, 1], [numPtsVTest, numPtsUTest])

    # comparison with exact solution
    ux_exact, uy_exact = exact_disp(xPhysTest, yPhysTest, model_data)
    ux_test = YTest[:, 0:1]
    uy_test = YTest[:, 1:2]
    err_norm = np.sqrt(np.sum((ux_exact - ux_test) ** 2 + (uy_exact - uy_test) ** 2))
    ex_norm = np.sqrt(np.sum(ux_exact ** 2 + uy_exact ** 2))
    rel_err_l2 = err_norm / ex_norm
    print("Relative L2 error in ", method, " sampling method: ", rel_err_l2)

    N = 10
    rel_errors = []
    for n in range(N):
        numPtsUTest = (n+2) * numPtsU
        numPtsVTest = (n+2) * numPtsV
        xPhysTest, yPhysTest = geomDomain.getUnifIntPts(numPtsUTest, numPtsVTest, [1,1,1,1])
        XTest = np.concatenate((xPhysTest, yPhysTest), axis=1).astype(data_type)
        XTest_tf = tf.convert_to_tensor(XTest)
        YTest = pred_model(XTest_tf).numpy()

        xPhysTest2D = np.resize(XTest[:, 0], [numPtsVTest, numPtsUTest])
        yPhysTest2D = np.resize(XTest[:, 1], [numPtsVTest, numPtsUTest])
        YTest2D_x = np.resize(YTest[:, 0], [numPtsVTest, numPtsUTest])
        YTest2D_y = np.resize(YTest[:, 1], [numPtsVTest, numPtsUTest])

        # comparison with exact solution
        ux_exact, uy_exact = exact_disp(xPhysTest, yPhysTest, model_data)
        ux_test = YTest[:, 0:1]
        uy_test = YTest[:, 1:2]
        err_norm = np.sqrt(np.sum((ux_exact - ux_test) ** 2 + (uy_exact - uy_test) ** 2))
        ex_norm = np.sqrt(np.sum(ux_exact ** 2 + uy_exact ** 2))
        rel_err_l2 = err_norm / ex_norm
        rel_errors.append(rel_err_l2)

    # plot the loss convergence
    return pred_model.adam_loss_hist, loss_func.history[5:], rel_errors


if __name__ == "__main__":
    methods = ["Grid", "Quadrature", "LHS", "Halton", "Hammersley", "Sobol"]
    adam_losses = []
    bfgs_losses = []
    fig = plt.figure('Loss')
    for method in methods:
        AdamLoss, BFGSLoss, Error = main(method=method)
        Error = tf.convert_to_tensor(Error, dtype=tf.float64)
        Error_avg = tf.reduce_mean(Error)
        Error_std = tf.math.reduce_std(Error)
        Error_min = tf.reduce_min(Error)
        Error_max = tf.reduce_max(Error)
        print("------- ", method, " -------")
        print("The average testing error is", Error_avg.numpy())
        print("Std. deviation of testing error is", Error_std.numpy())
        print("2 X Std. deviation of testing error is", Error_std.numpy())
        print("Min testing error is", Error_min.numpy())
        print("Max testing error is", Error_max.numpy())
        adam_losses.append(AdamLoss)
        bfgs_losses.append(BFGSLoss)
    adam_line_style = '--'
    bfgs_line_style = '-'

    adam_legend = plt.Line2D([0], [0], color='black', linestyle=adam_line_style, label='Adam')
    bfgs_legend = plt.Line2D([0], [0], color='black', linestyle=bfgs_line_style, label='BFGS')

    handles = []
    labels = []

    method_colors = ['b', 'g', 'r', 'c', 'm', 'y']

    fig = plt.figure('Loss')
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    for m in range(6):
        num_epoch = len(adam_losses[m])
        num_iter_bfgs = len(bfgs_losses[m])

        adam_handle, = plt.semilogy(range(num_epoch), adam_losses[m],
                                    linestyle=adam_line_style, color=method_colors[m], linewidth=1)
        #handles.append(adam_handle)
        #labels.append(methods[m])

        bfgs_handle, = plt.semilogy(range(num_epoch, num_epoch + num_iter_bfgs), bfgs_losses[m],
                                    linestyle=bfgs_line_style, color=method_colors[m], linewidth=1)
        handles.append(bfgs_handle)
        labels.append(methods[m])
    # Create legends
    legend1 = plt.legend(handles=[adam_legend, bfgs_legend], loc='upper center', frameon=False)
    legend2 = plt.legend(handles=handles, labels=labels, loc='upper right', frameon=False)

    # Add the legends to the plot
    plt.gca().add_artist(legend1)  # Add legend1 back to the plot
    plt.gca().add_artist(legend2)  # Add legend2 back to the plot

    plt.title('Loss convergence')
    plt.show()