import numpy as np
import scipy as sp
import ode_solver as solve_ode
import input_checks

def shooting_problem(func):
    """
    Constructs the function to be used for finding periodic orbits by numerical shooting.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array.
    ----------
    Returns
        A function, where the roots of its output are the initial values of a periodic orbit.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')

    def S(x0, phase_con, func_args=None, phase_args=None):
        """
        This is a function such that the root is the initial values is the periodic orbit of the ODE system 'func'.
        ----------
        Parameters
        x0 : list OR numpy.ndarray
            The guess of the coordinates and time period of a periodic orbit.
        phase_con : function
            Returns the phase condition of the shooting problem.
        func_args : list OR numpy.ndarray
            Additional parameters needed by 'func'.
        phase_args : list OR numpy.ndarray
            Additional parameters needed by 'phase_con'.
        ----------
        Returns
            A numpy.array containing the difference between the initial coordinates and evaluated coordinates after the time period, and the phase condition.
        """

        # INPUT CHECKS
        input_checks.test_list_nparray(x0, 'x0')
        if phase_con != None:
            input_checks.test_function(phase_con, 'phase_con')
        if func_args != None:
            input_checks.test_list_nparray(func_args, 'func_args')
        if phase_args != None:
            input_checks.test_list_nparray(phase_args, 'phase_args')

        T = x0[-1]
        x_in = x0[:-1]
        x_sol = solve_ode.solve_to(func, x_in, 0, T, 0.01, 'rk4', func_args)
        
        # EXTRACTING FINAL COORDINATE VALUES
        x_out = []
        for i in range(len(x_in)):
            x_out.append(x_sol[i][-1])

        if phase_con != None:
            return np.append(x_in - x_out, phase_con(x_in, phase_args))
        else:
            return x_in - x_out
    
    return S


def orbit_shoot(func, x0, solver=sp.optimize.fsolve, phase_con=None, func_args=None, phase_args=None):
    """
    Finds the initial values of a periodic orbit within func, based on 
    ----------
    Parameters
    func : function
        A function, where the roots of its output are the initial values of a periodic orbit.
    x0 : list OR numpy.ndarray
        An initial guess of values where an orbit could exist, with the final value being a guess of its time period.
    solver : function
        The function used for root-finding.
    phase_con : function
        Returns the phase condition of the shooting problem.
    func_args : list OR numpy.ndarray
        Additional parameters needed by 'func'.
    phase_args : list OR numpy.ndarray
        Additional parameters needed by 'phase_con'.
    ----------
    Returns
        A numpy.array with initial values where a periodic orbit exists, with the final value being the time period of this orbit. If the numerical root finder failed, the returned array is empty.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_function(solver, 'solver')
    if phase_con != None:
        input_checks.test_function(phase_con, 'phase_con')
    if func_args != None:
        input_checks.test_list_nparray(func_args, 'func_args')
    if phase_args != None:
        input_checks.test_list_nparray(phase_args, 'phase_args')

    S = shooting_problem(func)
    orbit = solver(S, x0, (phase_con, func_args, phase_args))

    return orbit
