import scipy as sp
import numpy as np
import math
import input_checks
import ode_solver


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

    diagonals = [np.ones(size)*dm, np.ones(size-1)*do, np.ones(size-1)*do]
    return sp.sparse.diags(diagonals, [0, 1, -1], format='csr')


def full_A(size, dm, do):
    """
    Creates a tridiagonal matrix.
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
        A np.ndarray tridiagonal matrix.
    """
    
    return np.diag(np.ones(size)*dm, 0) + np.diag(np.ones(size-1)*do, 1) + np.diag(np.ones(size-1)*do, -1)


def solve_diffusion(method, l_bound_type, r_bound_type, l_bound_func, r_bound_func, init_func, D, x_min, x_max, nx, t_min, t_max, nt, source_func=None, l_bound_args=None, r_bound_args=None, init_args=None, source_args=None, use_sparse=True):
    """
    Solves the diffusion equation using finite difference methods, based on the specified method and boundary conditions.
    ----------
    Parameters
    method : string
        Either 'lines', 'explicit_euler', 'implicit_euler', 'crank_nicolson', or 'imex'.
    l_bound_type : string
        Either 'dirichlet', 'neumann', or 'robin'.
    r_bound_type : string
        Either 'dirichlet', 'neumann', or 'robin'.
    l_bound_func : function
        Function that takes values (x, u, t) and list (args) as inputs and returns the left boundary value.
    r_bound_func : function
        Function that takes singular values (x, u, t) and list (args) as inputs and returns the right boundary value.
    init_func : function
        Function that takes (x, u, t) and list (args) as inputs and returns intitial solution array.
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
        Function that takes singular values (x, t, u) and list (args) as inputs and returns source value.
    l_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'l_bound_func'. If (l_bound_type) is 'robin', must contain two values [delta, gamma], such that r_bound_func returns delta - gamma*u(x).
    r_bound_args : list OR numpy.ndarray
        Additional arguments needed by 'r_bound_func'. If (r_bound_type) is 'robin', must contain two values [delta, gamma], such that r_bound_func returns delta - gamma*u(x).
    init_args : list OR numpy.ndarray
        Additional arguments needed by 'init_func'.
    source_args : list OR numpy.ndarray
        Additional arguments needed by 'source_func'.
    use_sparse : bool
        True indicates that sparse matrices are used for calculations.
    ----------
    Returns
        A numpy.array with a row of values for each solved parameter, and the final row being the x-values solved at.
    """

    # INPUT CHECKS
    input_checks.test_string(method, 'method')
    if method not in ['lines', 'explicit_euler', 'implicit_euler', 'crank_nicolson']:
        raise Exception('Argument (method) must be either \'lines\', \'explicit_euler\', \'implicit_euler\', \'crank_nicolson\'.')
    input_checks.test_string(l_bound_type, 'l_bound_type')
    if l_bound_type not in ['dirichlet', 'neumann', 'robin']:
        raise Exception('Argument (boundary_type) must be either \'dirichlet\', \'neumann\', or \'robin\'.')
    input_checks.test_string(r_bound_type, 'r_bound_type')
    if r_bound_type not in ['dirichlet', 'neumann', 'robin']:
        raise Exception('Argument (boundary_type) must be either \'dirichlet\', \'neumann\', or \'robin\'.')
    input_checks.test_function(l_bound_func, 'l_bound_func')
    input_checks.test_function(r_bound_func, 'r_bound_func')
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
    if l_bound_type == 'robin':
        if len(l_bound_args) != 2:
            raise Exception('Argument (l_bound_args) must contain two values.')
    if r_bound_args != None:
        input_checks.test_list_nparray(r_bound_args, 'r_bound_args')
    if r_bound_type == 'robin':
        if len(r_bound_args) != 2:
            raise Exception('Argument (r_bound_args) must contain two values.')
    if init_args != None:
        input_checks.test_list_nparray(init_args, 'init_args')
    if source_args != None:
        input_checks.test_list_nparray(source_args, 'source_args')
    if use_sparse not in [True, False]:
        raise Exception('Argument (use_sparse) must be a boolean True or False.')

    # MEETING STABILITY CONDITION
    dx = (x_max - x_min) / nx
    dt = (t_max - t_min) / nt
    C = (dt * D) / (dx ** 2)
    if method == 'explicit_euler' or method == 'lines':
        dt = 0.49*(dx**2)/D
        nt = math.ceil((t_max - t_min) / dt)
        C = (dt * D) / (dx ** 2)
    
    # ADJUST SOURCE TERM
    if source_func == None:
        def source_func(x, t, u, args):
            return np.zeros(np.size(x))
    
    # CONSTRUCT GRIDS
    x_arr = np.linspace(x_min, x_max, nx+1)
    t_arr = np.linspace(t_min, t_max, nt+1)

    # DEFINE x_int AND INITIALISE u_t
    if l_bound_type == 'dirichlet':
        xs = 1
    else:
        xs = 0
    if r_bound_type == 'dirichlet':
        x_int = x_arr[xs:-1]
    else:
        x_int = x_arr[xs:]
    
    u_t = init_func(x_int, 0, 0, init_args)
    size = len(u_t)

    # CREATE MATRICES
    if use_sparse == True:
        I_mat = sp.sparse.identity(size, format='csr')
        A_mat = sparse_A(size, -2, 1)
    elif use_sparse == False:
        I_mat = np.identity(size)
        A_mat = full_A(size, -2, 1)
    
    # DEFINE RHS VECTOR AND MODIFY SPARSE MATRICIES ACCORDING TO BOUNDARY CONDITIONS
    if l_bound_type != 'dirichlet':
        A_mat[0, 1] *= 2
    if l_bound_type == 'robin':
        A_mat[0, 0] *= 1 + l_bound_args[1] * dx
    
    if r_bound_type != 'dirichlet':
        A_mat[-1, -2] *= 2
    if r_bound_type == 'robin':
        A_mat[-1, -1] *= 1 + r_bound_args[1] * dx
    
    def make_b(t):
        b = np.zeros(size)
        if l_bound_type == 'dirichlet':
            b[0] = l_bound_func(x_min, t, u_t, l_bound_args)
        elif l_bound_type == 'neumann':
            b[0] = l_bound_func(x_min, t, u_t, l_bound_args) * -2 * dx
        elif l_bound_type == 'robin':
            b[0] = l_bound_func(x_min, t, u_t[0], [l_bound_args[0], 0]) * -2 * dx
        
        if r_bound_type == 'dirichlet':
            b[-1] = r_bound_func(x_max, t, u_t, r_bound_args)
        elif r_bound_type == 'neumann':
            b[-1] = r_bound_func(x_max, t, u_t, r_bound_args) * 2 * dx
        elif r_bound_type == 'robin':
            b[-1] = r_bound_func(x_max, t, u_t[-1], [r_bound_args[0], 0]) * 2 * dx

        return b
    
    # SOLVE
    if use_sparse == True:
        solver = sp.sparse.linalg.spsolve
    elif use_sparse == False:
        solver = np.linalg.solve

    if method == 'lines':
        def PDE(u, t, args):
            A_mat, b = args
            return (C/dt) * (A_mat@u + b + (dx**2) * (source_func(x_int, t, u, source_args)))
        b = make_b(t_max)
        sol = ode_solver.solve_to(PDE, u_t, t_min, t_max, dt, args=[A_mat, b])
        u_t = sol[:,-1][:-1]
        
    if method == 'explicit_euler':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t = solver(I_mat, u_t + C*(A_mat@u_t + b + (dx**2) * (source_func(x_int, t_arr[j], u_t, source_args))))
    
    if method == 'implicit_euler':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t = solver(I_mat - (C*A_mat), u_t + C * (b + (dx**2) * (source_func(x_int, t_arr[j], u_t, source_args))))
    
    if method == 'crank_nicolson':
        for j in range(0, nt):
            b = make_b(t_arr[j])
            u_t = solver(I_mat - ((C/2)*A_mat), (I_mat + ((C/2)*A_mat))@u_t + C * (b + (dx**2) * (source_func(x_int, t_arr[j], u_t, source_args))))

    # MODIFY u_t ACCORDING TO BOUNDARY CONDITIONS
    if l_bound_type == 'dirichlet':
        u_t = np.concatenate((np.array([l_bound_func(x_min, 0, 0, l_bound_args)]), u_t))
    if r_bound_type == 'dirichlet':
        u_t = np.concatenate((u_t, np.array([r_bound_func(x_max, 0, 0, r_bound_args)])))

    return np.vstack([u_t, x_arr])
