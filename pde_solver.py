import scipy as sp
import numpy as np


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


def solve_diffusion(method, boundary_type, l_bound_func, r_bound_func, init_func, D, x_min, x_max, nx, t_min, t_max, nt):
    """
    Solves the heat equation using the input method and boundary conditions.
    ----------
    Parameters
    method : string
        Either 'explicit_euler', 'implicit_euler', or 'crank_nicolson'.
    boundary_type : string
        Either 'dirichlet', 'neumann', or 'robin'.
    l_bound_func : function
        Function that takes singular (x, t) as inputs and returns the left boundary value.
    r_bound_func : function
        Function that takes singular values (x, t) as inputs and returns the right boundary value.
    init_func : function
        Function that takes arrays (x, t) and singular values (x_min, x_max) as inputs and returns intitial solution array.
    D : float OR int
        Diffusion Coefficient.
    x_min : float OR int
        Minimum x value.
    x_max : float OR int
        Maximum x value.
    nx : int
        Number of x values to solve across, affects step size used for x.
    t_min : float OR int
        Minimum t value.
    t_max : float OR int
        Maximum t value.
    nt : int
        Number of t values to solve across, affects step sized used for t.
    ----------
    Returns
        A numpy.array with a column of values for each solved parameter, and the final column being the x-values solved at.
    """

    # CHECK STABILITY CONDITION
    dx = (x_max - x_min) / nx
    dt = (t_max - t_min) / nt
    C = (dt * D) / (dx ** 2)

    if method == 'explicit_euler':
        if C > 0.5:
            raise Exception('Stability condition not met.')
    
    # CONSTRUCT ARRAYS
    x_arr = np.linspace(x_min, x_max, nx+1)
    t_arr = np.linspace(t_min, t_max, nt+1)
    
    if boundary_type == 'dirichlet':
        size = nx-1
        u_t = init_func(x_arr[1:nx], 0, x_min, x_max)

    if boundary_type == 'neumann':
        size = nx+1
        u_t = init_func(x_arr, 0, x_min, x_max)

    # CREATE SPARSE MATRICES
    I_mat = sp.sparse.identity(size, format='csr')
    A_mat = sparse_A(size, -2, 1)
    
    # DEFINE RHS VECTOR AND MODIFY SPARSE MATRICIES ACCORDING TO BOUNDARY CONDITIONS
    if boundary_type == 'dirichlet':
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t)
            b[-1] = r_bound_func(x_max, t)
            return b
    
    if boundary_type == 'neumann':
        A_mat[size-1, size-2] *= 2
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t)
            b[-1] = r_bound_func(x_max, t) * 2 * dx
            return b
    
    # NEED TO REVIEW THIS
    if boundary_type == 'robin':
        A_mat[size-1, size-2] *= 2
        A_mat[size-1, size-1] *= 1+(r_bound_func(x_max,0)*dx)
        def make_b(t):
            b = np.zeros(size)
            b[0] = l_bound_func(x_min, t)
            b[-1] = r_bound_func(x_max, t) * 2 * dx
            return b
    
    # SOLVE
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
        u_t = np.concatenate((np.array([l_bound_func(x_min, 0)]), u_t, np.array([r_bound_func(x_max, 0)])))

    return np.vstack([u_t, x_arr]).T
