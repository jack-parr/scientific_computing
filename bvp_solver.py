# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import ode_solver
import input_checks

def solve_bvp(method, boundary_type, l_bound_func, r_bound_func, init_func, D, x_min, x_max, nx, source_func=None, l_bound_args=None, r_bound_args=None, init_args=None, source_args=None):

    # ADJUST SOURCE TERM
    if source_func == None:
        def source_func(x, u, args):
            return np.zeros(np.size(x))
    

    # DEFINE FUNCTION FOR SOLVER
    def findiff_problem(u, t, args):
    
        F = np.zeros(size)

        # FIRST TERM
        F[0] = ((u[1] - 2*u[0] + l_bound_func(x_min, l_bound_args)) / dx**2)

        # INNER TERMS
        for i in range(1, size-1):
            F[i] = ((u[i+1] - 2*u[i] + u[i-1]) / dx**2)

        # FINAL TERM
        if boundary_type == 'dirichlet':
            F[-1] = ((r_bound_func(x_max, r_bound_args) - 2*u[-1] + u[-2]) / dx**2)
        elif boundary_type == 'neumann':
            F[-1] = ((-2*u[-1] + 2*u[-2]) / dx**2) + ((2*r_bound_func(x_max, r_bound_args))/dx)
        elif boundary_type == 'robin':
            F[-1] = (-2*(1 + r_bound_args[1]*dx)*u[-1] + 2*u[-2]) / (dx**2) + ((2*r_bound_args[0])/dx)

        return D*F + source_func(x_arr[1:size+1], t, u_t, source_args)


    dx = (x_max - x_min) / nx
    dt = 0.5*(dx**2)/D
    x_arr = np.linspace(x_min, x_max, nx+1)

    # INITIALISE
    if boundary_type == 'dirichlet':
        size = nx-1
        u_t = init_func(x_arr[1:nx], init_args)
    else:
        size = nx
        u_t = init_func(x_arr[1:], init_args)

    # SOLVE
    if method == 'rk4':
        u_t = ode_solver.solve_to(findiff_problem, 'rk4', u_t, 0, 1, dt)[:,-1][:-1]
    
    # ADD BOUNDARIES
    if boundary_type == 'dirichlet':
        u_t = np.concatenate((np.array([l_bound_func(x_min, l_bound_args)]), u_t, np.array([r_bound_func(x_max, r_bound_args)])))
    else:
        u_t = np.concatenate((np.array([l_bound_func(x_min, l_bound_args)]), u_t))

    return np.vstack([u_t, x_arr])


def l_bound(x, args):
    return 0


def r_bound_dirichlet(x, args):
    return 0

def r_bound_robin(x, args):
    delta, gamma = args
    return delta - (gamma*x)


def init(x, args):
    return 0.1*x


def source(x, u, args):
    return np.ones(np.size(x))


x_pred = solve_bvp(
    method='rk4', 
    boundary_type='dirichlet', 
    l_bound_func=l_bound, 
    r_bound_func=r_bound_dirichlet, 
    init_func=init, 
    D=1, 
    x_min=0, 
    x_max=2, 
    nx=100, 
    init_args=[0, 5])

print(np.shape(x_pred))
plt.plot(x_pred[-1], x_pred[0])