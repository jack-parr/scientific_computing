# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
# %%
def solve_heat(D, x_max, t_max, nx, nt):
    # explicit euler
    return 1
# %%
# paramters
D = 1
x_max = 10
t_max = 1
nx = 100
nt = 1000

def l_bound(x, t): # at x=0
    return 0

def r_bound(x, t): # at x=x_max
    return 0

def initial(x, t, L):
    y = np.sin(math.pi * x / L)
    return y

# %%
def heat_exact(x, t, D, L):
    return np.exp((-D * t * math.pi**2) / (x_max**2)) * np.sin((math.pi * x) / (x_max))

x_arr = np.linspace(0, x_max, nx)
heat_true = heat_exact(x_arr, t_max, D, x_max)
plt.plot(x_arr, heat_true)