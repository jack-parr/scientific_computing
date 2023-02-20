import numpy as np
import math


def euler_step(func, x0, t0, delta_t, args):
    """
    Performs one iteration of the Euler method, with a timestep of 'delta_t'.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array.
    x0 : list
        The initial values for this step.
    t0 : float OR int
        The initial time value for this step.
    delta_t : float OR int
        The timestep to be solved to.
    args : list
        Additional parameters needed by func.
    ----------
    Returns
        (1) List values of the function solved after the timestep.
        (2) Float of the new time value after the timestep.
    """

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
    x0 : list
        The initial values for this step.
    t0 : float OR int
        The initial time value for this step.
    delta_t : float OR int
        The timestep to be solved to.
    args : list
        Additional parameters needed by func.
    ----------
    Returns
        (1) List values of the function solved after the timestep.
        (2) Float of the new time value after the timestep.
    """

    k1 = grad = func(x0, t0, args)
    k2 = grad + (k1*delta_t)/2
    k3 = grad + (k2*delta_t)/2
    k4 = grad + (k3*delta_t)
    t1 = t0 + delta_t

    return x0 + (k1/6 + k2/3 + k3/3 + k4/6)*delta_t, t1


def solve_to(func, method, x1, t1, t2, deltat_max, args):
    """
    Solves the function from (x1, t1) to (x2, t2) taking steps no larger than 'deltat_max', using the method defined by 'method'.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array. 
    method : string
        Either 'euler' for Euler method, or 'rk4' ofr 4th-Order Runge-Kutta method.
    x1 : list
        Initial values at 't1'.
    t1 : float OR int
        Initial time value.
    t2 : float OR int
        Desired time value.
    deltat_max : float OR int
        Maximum allowed timestep.
    args : list
        Additional parameters needed by func.
    ----------
    Returns
        A numpy.array with a column of values for each solved parameter, and the final column being the timesteps solved at.
    """

    if method == 'euler':
        fstep = euler_step
    elif method == 'rk4':
        fstep = rk4_step

    x_sol = np.empty(shape=(math.floor((t2-t1)/deltat_max)+2, len(x1)+1))*np.nan
    x_sol[0][:len(x1)] = x1
    x_sol[0][-1] = t1

    i = 1
    while not math.isclose(t1, t2):
        x1, t1 = fstep(func, x1, t1, min([t2-t1, deltat_max]), args)
        x_sol[i][:len(x1)] = x1
        x_sol[i][-1] = t1
        i += 1

    return x_sol[~np.isnan(x_sol).any(axis=1), :]

