# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import ode_solver
import math
import input_checks


def findiff_problem(u, t, args):
    # finite difference for root solver.
    source, boundary, size, dx, l_bound, r_bound = args
    F = np.zeros(size)

    # FIRST TERM
    F[0] = ((u[1] - 2*u[0] + l_bound) / dx**2) + source[0]

    # INNER TERMS
    for i in range(1, size-1):
        F[i] = ((u[i+1] - 2*u[i] + u[i-1]) / dx**2) + source[i]

    # FINAL TERM
    if boundary == 'dirichlet':
        F[-1] = ((r_bound[0] - 2*u[-1] + u[-2]) / dx**2) + source[-1]
    elif boundary == 'neumann':
        F[-1] = ((-2*u[-1] + 2*u[-2]) / dx**2) + ((2*r_bound[0])/dx) + source[-1]
    elif boundary == 'robin':
        F[-1] = (-2*(1 + r_bound[1]*dx)*u[-1] + 2*u[-2]) / (dx**2) + ((2*r_bound[0])/dx) + source[-1]

    return F


def sparse_A(size, dm, do):
    """
    Creates a sparse matrix used by the scipy spsolve function.
    ----------
    Parameters
    size : int
        The square dimension value of the matrix. 
    dm : float OR int
        Value along the main diagonal.
    do : float OR int
        Value along both the offset diagonals.
    ----------
    Returns
        A scipy sparse csr_matrix.
    """

    data = np.concatenate((dm*np.ones(size), do*np.ones(2*(size-1))))
    row_idx = np.concatenate((np.arange(0, size), np.arange(0, size-1), np.arange(1, size)))
    col_idx = np.concatenate((np.arange(0, size), np.arange(1, size), np.arange(0, size-1)))

    return sp.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(size,size))


def solve_diffusion(method, boundary_type, l_bound_func, r_bound_func, r_bound_args, init_func, D, x_min, x_max, nx, t_min, t_max, nt, source_func=None, l_bound_args=None, init_args=None, source_args=None):
    """
    Solves the heat equation using finite difference methods, based on the input method and boundary conditions.
    ----------
    Parameters
    method : string
        Either 'root_rk4', 'explicit_euler', 'implicit_euler', or 'crank_nicolson'.
    boundary_type : string
        Either 'dirichlet', 'neumann', or 'robin'.
    l_bound_func : function
        Function that takes singular (x, t) as inputs and returns the left boundary value.
    r_bound_func : function
        Function that takes singular values (x, t) as inputs and returns the right boundary value.
    r_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'r_bound_func'. Must contain two values [delta, gamma], such that r_bound_func = delta - gamma*u(x).
    init_func : function
        Function that takes arrays (x, t) and singular values (x_min, x_max) as inputs and returns intitial solution array.
    D : float OR int
        Diffusion Coefficient.
    x_min : float OR int
        Minimum x value.
    x_max : float OR int
        Maximum x value.
    nx : int
        Number of x values in the grid, affects step size used for x.
    t_min : float OR int
        Minimum t value.
    t_max : float OR int
        Maximum t value.
    nt : int
        Number of t values in the grid, affects step sized used for t.
    source_func : function
        Function that takes singular values (x, t) and list (args) as inputs and returns source value.
    l_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'l_bound_func'.
    init_args : list OR numpy.ndarray
        Additional arguments needed by 'init_func'.
    source_args : list OR numpy.ndarray
        Additional arguments needed by 'source_func'.
    ----------
    Returns
        A numpy.array with a row of values for each solved parameter, and the final row being the x-values solved at.
    """

    # INPUT CHECKS
    input_checks.test_string(method, 'method')
    if method not in ['root_rk4', 'explicit_euler', 'implicit_euler', 'crank_nicolson']:
        raise Exception('Argument (method) must be either \'root_rk4\', \'explicit_euler\', \'implicit_euler\', or \'crank_nicolson\'.')
    input_checks.test_string(boundary_type, 'boundary_type')
    if boundary_type not in ['dirichlet', 'neumann', 'robin']:
        raise Exception('Argument (boundary_type) must be either \'dirichlet\', \'neumann\', or \'robin\'.')
    input_checks.test_function(l_bound_func, 'l_bound_func')
    input_checks.test_function(r_bound_func, 'r_bound_func')
    input_checks.test_list_nparray(r_bound_args, 'r_bound_args')
    if len(r_bound_args) != 2:
        raise Exception('Argument (r_bound_args) must contain two values.')
    input_checks.test_function(init_func, 'init_func')
    input_checks.test_float_int(D, 'D')
    input_checks.test_float_int(x_min, 'x_min')
    input_checks.test_float_int(x_max, 'x_max')
    input_checks.test_int(nx, 'nx')
    input_checks.test_float_int(t_min, 't_min')
    input_checks.test_float_int(t_max, 't_max')
    input_checks.test_int(nt, 'nt')
    if source_func != None:
        input_checks.test_function(source_func, 'source_func')
    if l_bound_args != None:
        input_checks.test_list_nparray(l_bound_args, 'l_bound_args')
    if init_args != None:
        input_checks.test_list_nparray(init_args, 'init_args')
    if source_args != None:
        input_checks.test_list_nparray(source_args, 'source_args')

    # MEETING STABILITY CONDITION
    dx = (x_max - x_min) / nx
    dt = 0.5*(dx**2)/D
    C = (dt * D) / (dx ** 2)
    
    # ADJUST SOURCE TERM
    if source_func == None:
        def source_func(x, t, u, args):
            return np.zeros(np.size(x))
    
    # CONSTRUCT GRIDS
    x_arr = np.linspace(x_min, x_max, nx+1)
    t_arr = np.linspace(t_min, t_max, nt+1)
    
    if boundary_type == 'dirichlet':
        size = nx-1
        u_t = init_func(x_arr[1:nx], 0, init_args)

    if boundary_type == 'neumann':
        size = nx
        u_t = init_func(x_arr[1:], 0, init_args)

    if boundary_type == 'robin':
        size = nx
        u_t = init_func(x_arr[1:], 0, init_args)

    # CREATE SPARSE MATRICES
    I_mat = sp.sparse.identity(size, format='csr')
    A_mat = sparse_A(size, -2, 1)
    
    # DEFINE RHS VECTOR AND MODIFY SPARSE MATRICIES ACCORDING TO BOUNDARY CONDITIONS
    if boundary_type == 'dirichlet':
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t, l_bound_args)
            b[-1] = r_bound_func(x_max, t, args=[r_bound_args[0], 0])
            return b + dt*source_func(x_arr[1:nx], t, u_t, source_args)
    
    if boundary_type == 'neumann':
        A_mat[size-1, size-2] *= 2
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t, l_bound_args)
            b[-1] = r_bound_func(x_max, t, args=[r_bound_args[0], 0]) * 2 * dx
            return b + dt*source_func(x_arr[1:], t, u_t, source_args)
    
    if boundary_type == 'robin':
        A_mat[size-1, size-2] *= 2
        A_mat[size-1, size-1] *= 1 + r_bound_args[1] * dx
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t, l_bound_args)
            b[-1] = r_bound_args[0] * 2 * dx
            return b + dt*source_func(x_arr[1:], t, u_t, source_args)
    
    # SOLVE
    if method == 'root_rk4':
        u_t = ode_solver.solve_to(findiff_problem, 'rk4', u_t, t_min, t_max, dt, args=[source_func(x_arr[1:size+1], 0, u_t, source_args), boundary_type, size, dx, l_bound_func(x_min, 0, l_bound_args), r_bound_args])[:,-1][:-1]

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

    # MODIFY u_t ACCORDING TO BOUNDARY CONDITIONS
    if boundary_type == 'dirichlet':
        u_t = np.concatenate((np.array([l_bound_func(x_min, 0, l_bound_args)]), u_t, np.array([r_bound_func(x_max, 0, args=[r_bound_args[0], 0])])))
    else:
        u_t = np.concatenate((np.array([l_bound_func(x_min, 0, l_bound_args)]), u_t))

    return np.vstack([u_t, x_arr])


def diff_l_bound(x, t, args):
    return 0


def diff_r_bound(x, t, args):
    delta, gamma = args
    return delta - (gamma*x)


def diff_init(x, t, args):
    return 0.1*x


def diff_source(x, t, u, args):
    return np.ones(np.size(x))


x_pred = solve_diffusion(
    method='root_rk4', 
    boundary_type='dirichlet', 
    l_bound_func=diff_l_bound, 
    r_bound_func=diff_r_bound, 
    r_bound_args=[1, 0],
    init_func=diff_init, 
    D=1, 
    x_min=0, 
    x_max=1, 
    nx=100, 
    t_min=0, 
    t_max=1, 
    nt=1000,
    init_args=[0, 5])

plt.plot(x_pred[-1], x_pred[0])