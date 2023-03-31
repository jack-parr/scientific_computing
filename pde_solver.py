# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
# %%
def sparse_A(size, dm, ds):
    data = np.concatenate((dm*np.ones(size), ds*np.ones(2*(size-1))))
    row_idx = np.concatenate((np.arange(0, size), np.arange(0, size-1), np.arange(1, size)))
    col_idx = np.concatenate((np.arange(0, size), np.arange(1, size), np.arange(0, size-1)))
    return sp.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(size,size))


def solve_heat(method, boundary_type, D, x_max, t_max, nx, nt):

    dx = (x_max - x_min) / nx
    dt = (t_max - t_min) / nt
    C = (dt * D) / (dx ** 2)
    if method == 'explicit_euler':
        if C > 0.5:
            raise Exception('Stability condition not met.')
    
    x_arr = np.linspace(x_min, x_max, nx+1)
    t_arr = np.linspace(t_min, t_max, nt+1)
    
    if boundary_type == 'dirichlet':
        size = nx-1
        u_t = initial(x_arr[1:nx], 0, x_min, x_max)

    if boundary_type == 'neumann':
        size = nx+1
        u_t = initial(x_arr, 0, x_min, x_max)

    # CREATE MATRICES
    I_mat = sp.sparse.identity(size, format='csr')
    A_mat = sparse_A(size, -2, 1)
    
    if boundary_type == 'dirichlet':
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound(x_min, t)
            b[-1] = r_bound(x_max, t)
            return b
    
    if boundary_type == 'neumann':
        A_mat[size-1, size-2] *= 2
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound(x_min, t)
            b[-1] = r_bound(x_max, t) * 2 * dx
            return b
    
    # NEED TO REVIEW THIS
    if boundary_type == 'robin':
        A_mat[size-1, size-2] *= 2
        A_mat[size-1, size-1] *= 1+(r_bound(x_max,0)*dx)
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound(x_min, t)
            b[-1] = r_bound(x_max, t) * 2 * dx
            return b
    

    if method == 'explicit_euler':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t += sp.sparse.linalg.spsolve(I_mat, C*(A_mat@u_t + b))
    
    if method == 'implicit_euler':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t = sp.sparse.linalg.spsolve(I_mat - (C*A_mat), u_t + (C*b))
    
    if method == 'crank_nicolson':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t = sp.sparse.linalg.spsolve(I_mat - ((C/2)*A_mat), (I_mat + ((C/2)*A_mat))@u_t + (C*b))

    # MOD u_t BASED ON BOUNDARY COND
    if boundary_type == 'dirichlet':
        u_t = np.concatenate((np.array([l_bound(x_min, 0)]), u_t, np.array([r_bound(x_max, 0)])))

    return u_t, x_arr
# %%
# parameters
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

u_t, x_arr = solve_heat('crank_nicolson', 'dirichlet', D, x_max, t_max, nx, nt)

def heat_exact(x, t, D, x_min, x_max):
    L = x_max - x_min
    return np.exp(-D * t * (math.pi**2 / L**2)) * np.sin((math.pi * (x - x_min)) / (L))

heat_true = heat_exact(x_arr, t_max, D, x_min, x_max)
plt.plot(x_arr, heat_true, x_arr, u_t)
