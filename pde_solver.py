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
x_min = 0
x_max = 5
t_min = 0
t_max = 0.5
nx = 100
nt = 1000

def l_bound(x, t): # at x=x_min
    return 0

def r_bound(x, t): # at x=x_max
    return 0

def initial(x, t, x_min, x_max):
    y = np.sin((math.pi * x) / (x_max - x_min))
    return y

dx = (x_max - x_min) / nx
dt = (t_max - t_min) / nt

# check stability condition
check = (dt * D) / (dx ** 2)
if check > 0.5:
    raise Exception('Stability condition not met.')

x_arr = np.linspace(x_min, x_max, nx+1)
t_arr = np.linspace(t_min, t_max, nt+1)

u_t = initial(x_arr[1:nx], 0, x_min, x_max)

# create A matrix
A = np.zeros((nx-1, nx-1))
A[0, 0] = -2
A[0, 1] = 1
A[nx-2, nx-3] = 1
A[nx-2, nx-2] = -2
for i in range(1, nx-2):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1

# create B matrix
def make_b(t):
    b = np.zeros(nx-1)
    b[0] = l_bound(x_min, t)
    b[-1] = r_bound(x_max, t)
    return b

# loop over time and solve at each time step.
# for j in range(0, nt):
#     b = make_b(t_arr[j])
#     # u_t = np.linalg.solve(A, -b-(dx**2))
#     u_t = np.linalg.solve(A_id, (A@u_t+b))

#u_t = np.concatenate((np.array([l_bound(x_min, 0)]), u_t, np.array([r_bound(x_max, 0)])))

# solvig using solve_ivp
def PDE(t, u, D, A, b):
    return D / dx**2 * (A @ u + b)

b = make_b(0)
sol = sp.integrate.solve_ivp(PDE, (t_min, t_max), u_t, args=(D, A, b))
t = sol.t
u = sol.y
u_t = u[:,-1]
u_t = np.concatenate((np.array([l_bound(x_min, 0)]), u_t, np.array([r_bound(x_max, 0)])))

#############

def heat_exact(x, t, D, x_min, x_max):
    L = x_max - x_min
    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

heat_true = heat_exact(x_arr, t_max, D, x_min, x_max)
plt.plot(x_arr, heat_true, x_arr, u_t)