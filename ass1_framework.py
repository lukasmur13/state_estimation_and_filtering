from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import scipy as sp

k1 = 0.1
k2 = 0.2
k3 = 0.01

k1f = 1
k2f = 2
k1r = 4
k2r = 3

def linear_system(t,x, K1=k1, K2=k2, K3=k3):
    x1, x2, x3 = x
    
    x1dot = -K1 * x1 + K3*x3
    x2dot = K1 * x1 - K2 * x2
    x3dot = K2 * x2 - K3 * x3

    return np.array([x1dot, x2dot, x3dot]).squeeze()

def nonlinear_system(t, x):
    # if you have issues with the dimensions, try A, B, C = x
    A, B, C = np.hsplit(x, 3)
    r1 = (A / k1f - B / k1r) / (1 + A / k1f + B / k1r)
    r2 = (B / k2f - C / k2r) / (1 + B / k2f + C / k2r)
    dAdt = -r1
    dBdt = r1 - r2
    dCdt = r2
    return np.array([dAdt, dBdt, dCdt]).squeeze()

A_lin = jacobian(linear_system)
A_nonlin = jacobian(nonlinear_system)

