import numpy as np
import math
import input_checks


def euler_step(func, x0, t0, delta_t, args):
    """
    Performs one iteration of the Euler method, with a timestep of 'delta_t'.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array.
    x0 : list OR numpy.ndarray
        The initial values for this step.
    t0 : float OR int
        The initial time value for this step.
    delta_t : float OR int
        The timestep to be solved to.
    args : list OR numpy.ndarray
        Additional parameters needed by 'func'.
    ----------
    Returns
        (1) List values of the function solved after the timestep.
        (2) Float of the new time value after the timestep.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_float_int(t0, 't0')
    input_checks.test_float_int(delta_t, 'delta_t')
    if args != None:
        input_checks.test_list_nparray(args, 'args')

    grad = func(x0, t0, args)
    t1 = t0 + delta_t

    return x0 + grad*delta_t, t1


def rk4_step(func, x0, t0, delta_t, args):
    """
    Performs one iteration of the 4th-Order Runge-Kutta method, with a timestep of 'delta_t'.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array.
    x0 : list OR numpy.ndarray
        The initial values for this step.
    t0 : float OR int
        The initial time value for this step.
    delta_t : float OR int
        The timestep to be solved to.
    args : list OR numpy.ndarray
        Additional parameters needed by 'func'.
    ----------
    Returns
        (1) List values of the function solved after the timestep.
        (2) Float of the new time value after the timestep.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_float_int(t0, 't0')
    input_checks.test_float_int(delta_t, 'delta_t')
    if args != None:
        input_checks.test_list_nparray(args, 'args')

    k1 = grad = func(x0, t0, args)
    k2 = grad + (k1*delta_t)/2
    k3 = grad + (k2*delta_t)/2
    k4 = grad + (k3*delta_t)
    t1 = t0 + delta_t

    return x0 + (k1/6 + k2/3 + k3/3 + k4/6)*delta_t, t1


def solve_to(func, method, x0, t0, t1, deltat_max, args=None):
    """
    Solves the function from (x0, t0) to (x1, t1) taking steps no larger than 'deltat_max', using the method defined by 'method'.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array. 
    method : string
        Either 'euler' for Euler method, or 'rk4' for 4th-Order Runge-Kutta method.
    x0 : list OR numpy.ndarray
        Initial values at 't0'.
    t0 : float OR int
        Initial time value.
    t1 : float OR int
        Desired time value.
    deltat_max : float OR int
        Maximum allowed timestep.
    args : list OR numpy.ndarray
        Additional parameters needed by 'func'.
    ----------
    Returns
        A numpy.array with a row of values for each solved parameter, and the final row being the timesteps solved at.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_string(method, 'method')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_float_int(t0, 't0')
    input_checks.test_float_int(t1, 't1')
    input_checks.test_float_int(deltat_max, 'deltat_max')
    if args != None:
        input_checks.test_list_nparray(args, 'args')

    if method == 'euler':
        fstep = euler_step
    elif method == 'rk4':
        fstep = rk4_step
    else:
        raise Exception('Argument (method) must be either \'euler\' or \'rk4\'.')

    x_sol = np.empty(shape=(math.floor((t1-t0)/deltat_max)+2, len(x0)+1))*np.nan
    x_sol[0][:len(x0)] = x0
    x_sol[0][-1] = t0

    i = 1
    while not math.isclose(t0, t1):
        x0, t0 = fstep(func, x0, t0, min([t1-t0, deltat_max]), args)
        x_sol[i][:len(x0)] = x0
        x_sol[i][-1] = t0
        i += 1

    return x_sol[~np.isnan(x_sol).any(axis=1), :].T
