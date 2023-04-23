import scipy as sp
import numpy as np
import ode_solver
import input_checks


def solve_bvp(method, boundary_type, l_bound_func, r_bound_func, init_func, D, x_min, x_max, nx, source_func=None, l_bound_args=None, r_bound_args=None, init_args=None, source_args=None):
    """
    Solves a BVP problem using root solvers with finite difference methods, based on the input method and boundary conditions.
    ----------
    Parameters
    method : string
        Either 'scipy', 'euler', or 'rk4'.
    boundary_type : string
        Either 'dirichlet', 'neumann', or 'robin'.
    l_bound_func : function
        Function that takes singular value (x) and any arguments as inputs and returns the left boundary value.
    r_bound_func : function
        Function that takes singular value (x) and any arguments as inputs and returns the right boundary value.
    init_func : function
        Function that takes array (x) and arguments as inputs and returns intitial solution array.
    D : float OR int
        Coefficient of second order derivative in the problem.
    x_min : float OR int
        Minimum x value.
    x_max : float OR int
        Maximum x value.
    nx : int
        Number of x values in the grid, affects step size used for x.
    source_func : function
        Function that takes array (x) and list (args) as inputs and returns source array.
    l_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'l_bound_func'.
    r_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'r_bound_func'. If (boundary_type) is 'robin', must contain two values [delta, gamma], such that r_bound_func = delta - gamma*u(x).
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
    if method not in ['scipy', 'euler', 'rk4']:
        raise Exception('Argument (method) must be either \'scipy\', \'euler\', or \'rk4\'.')
    input_checks.test_string(boundary_type, 'boundary_type')
    if boundary_type not in ['dirichlet', 'neumann', 'robin']:
        raise Exception('Argument (boundary_type) must be either \'dirichlet\', \'neumann\', or \'robin\'.')
    input_checks.test_function(l_bound_func, 'l_bound_func')
    input_checks.test_function(r_bound_func, 'r_bound_func')
    input_checks.test_function(init_func, 'init_func')
    input_checks.test_float_int(D, 'D')
    input_checks.test_float_int(x_min, 'x_min')
    input_checks.test_float_int(x_max, 'x_max')
    input_checks.test_int(nx, 'nx')
    if source_func != None:
        input_checks.test_function(source_func, 'source_func')
    if l_bound_args != None:
        input_checks.test_list_nparray(l_bound_args, 'l_bound_args')
    if r_bound_args != None:
        input_checks.test_list_nparray(r_bound_args, 'r_bound_args')
    if boundary_type == 'robin':
        if len(r_bound_args) != 2:
            raise Exception('Argument (r_bound_args) must contain two values.')
    if init_args != None:
        input_checks.test_list_nparray(init_args, 'init_args')
    if source_args != None:
        input_checks.test_list_nparray(source_args, 'source_args')
    
    # ADJUST SOURCE TERM
    if source_func == None:
        def source_func(x, u, args):
            return np.zeros(np.size(x))
    
    # DEFINE FUNCTION FOR SOLVER
    def findiff_problem(u, t=None, args=None):
    
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

        return D*F + source_func(x_arr[1:size+1], u_t, source_args)


    # MEETING STABILITY CONDITION
    dx = (x_max - x_min) / nx
    dt = 0.25*(dx**2)/D
    C = (dt * D) / (dx ** 2)
    if C > 0.5:
        raise Exception('Error when adjusting (dt) to meet stability condition.')

    # INITIALISE
    x_arr = np.linspace(x_min, x_max, nx+1)
    if boundary_type == 'dirichlet':
        size = nx-1
        u_t = init_func(x_arr[1:nx], init_args)
    else:
        size = nx
        u_t = init_func(x_arr[1:], init_args)

    # SOLVE
    if method == 'scipy':
        u_t = sp.optimize.root(findiff_problem, u_t)
        print(u_t.message)
        u_t = u_t.x
    else:
        u_t = ode_solver.solve_to(findiff_problem, method, u_t, 0, 2, dt)[:,-1][:-1]
    
    # ADD BOUNDARIES
    if boundary_type == 'dirichlet':
        u_t = np.concatenate((np.array([l_bound_func(x_min, l_bound_args)]), u_t, np.array([r_bound_func(x_max, r_bound_args)])))
    else:
        u_t = np.concatenate((np.array([l_bound_func(x_min, l_bound_args)]), u_t))

    return np.vstack([u_t, x_arr])
