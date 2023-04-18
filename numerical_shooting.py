import numpy as np
import ode_solver as solve_ode


def shooting_problem(func, phase_con, func_args, phase_args):
    """
    Constructs the function to be used for finding periodic orbits by numerical shooting.
    ----------
    Parameters
    func : function
        The ODE to solve. The ODE function should be in first-order form, take a single list input and return the right-hand side of the ODE as a numpy.array.
    phase_con : function
        Returns the phase condition of the shooting problem.
    func_args : list
        Additional parameters needed by 'func'.
    phase_args : list
        Additional parameters needed by 'phase_con'.
    ----------
    Returns
        A function, where the roots of its output are the initial values of a periodic orbit.
    """

    def G(x0, phase_con, func_args, phase_args):
        """
        This is a function such that the root is the initial values is the periodic orbit of the ODE system 'func'.
        ----------
        Parameters
        x0 : list
            The guess of the coordinates and time period of a periodic orbit.
        phase_con : function
            Returns the phase condition of the shooting problem.
        func_args : list
            Additional parameters needed by 'func'.
        phase_args : list
            Additional parameters needed by 'phase_con'.
        ----------
        Returns
            A numpy.array containing the difference between the initial coordinates and evaluated coordinates after the time period, and the phase condition.
        """

        T = x0[-1]
        x1 = x0[:-1]
        x_sol = solve_ode.solve_to(func, 'rk4', x1, 0, T, 0.01, func_args)

        if phase_con != None:
            return np.append(x1 - x_sol[-1][:-1], phase_con(x1, phase_args))
        else:
            return x1 - x_sol[-1][:-1]
    
    return G


def orbit_shoot(func, x0, solver, phase_con=None, func_args=None, phase_args=None):
    """
    Finds the initial values of a periodic orbit within func, based on 
    ----------
    Parameters
    func : function
        A function, where the roots of its output are the initial values of a periodic orbit.
    x0 : list
        An initial guess of values where an orbit could exist, with the final value being a guess of its time period.
    solver : function
        The function used for root-finding.
    phase_con : function
        Returns the phase condition of the shooting problem.
    func_args : list
        Additional parameters needed by 'func'.
    phase_args : list
        Additional parameters needed by 'phase_con'.
    ----------
    Returns
        A numpy.array with initial values where a periodic orbit exists, with the final value being the time period of this orbit. If the numerical root finder failed, the returned array is empty.
    """

    G = shooting_problem(func, phase_con, func_args, phase_args)
    orbit = solver(G, x0, (phase_con, func_args, phase_args))

    return orbit
