import numpy as np
import scipy as sp
import input_checks

def natural_continuation(func, x0, init_args, vary_par_idx, max_par, num_steps, discretisation=(lambda x: x), solver=sp.optimize.fsolve):
    """
    Performs natural parameter continuation. Solves the function while varying the indicated parameter.
    ----------
    Parameters
    func : function
        The function to be solved.
    x0 : list OR numpy.ndarray
        Initial coordinates and phase condition if relevant.
    init_args : list OR numpy.ndarray
        Initial args to be used by 'func'.
    vary_par_idx : int
        Index of parameter to be varied within 'init_args'.
    max_par : float OR int
        Maximum (or minimum) value of the parameter at 'vary_par_idx' to be solved at.
    num_steps : int
        Number of parameter values to be solved at. Paramters values are evenly incremented this number of times.
    discretisation : function
        Function to be used to discretise the problem. Defaults to no adaptation.
    solver : function
        Function to be used to solve the root problem. Defaults to scipy's 'fsolve'.
    ----------
    Returns
        A numpy.array with a row of values for each solved coordinate, and the final row being the varied parameter values solved at.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_list_nparray(init_args, 'init_args')
    input_checks.test_int(vary_par_idx, 'vary_par_idx')
    input_checks.test_float_int(max_par, 'max_par')
    input_checks.test_int(num_steps, 'num_steps')
    input_checks.test_function(discretisation, 'discretisation')
    input_checks.test_function(solver, 'solver')

    u_stor = []
    pars = np.linspace(init_args[vary_par_idx], max_par, num_steps)

    for par in pars:
        init_args[vary_par_idx] = par
        root = solver(discretisation(func), x0, args=init_args)
        u_stor.append(root)
        x0 = root

    return np.vstack([np.array(u_stor).T, pars])


def pseudo_arclength(func, x0, init_args, vary_par_idx, max_par, num_steps, discretisation=(lambda x: x), solver=sp.optimize.fsolve, phase_con=None):
    """
    Performs peudo-arclength continuation. Solves the augumented problem function, which is the normal root problem alongside the arclength equation, while varying the indicated parameter.
    ----------
    Parameters
    func : function
        The function to be solved.
    x0 : list OR numpy.ndarray
        Initial coordinates and phase condition if relevant.
    init_args : list OR numpy.ndarray
        Initial args to be used by 'func'.
    vary_par_idx : int
        Index of parameter to be varied within 'init_args'.
    max_par : float OR int
        Maximum (or minimum) value of the parameter at 'vary_par_idx' to be solved at.
    num_steps : int
        Number of parameter values to be solved at. Paramters values are evenly incremented this number of times.
    discretisation : function
        Function to be used to discretise the problem. Defaults to no adaptation.
    solver : function
        Function to be used to solve the root problem. Defaults to scipy's 'fsolve'.
    phase_con : function
        Function which returns the phase condition of the problem.
    ----------
    Returns
        A numpy.array with a row of values for each solved coordinate and phase condition if relevant, and the final row being the varied parameter values solved at.
    """

    # INPUT CHECKS
    input_checks.test_function(func, 'func')
    input_checks.test_list_nparray(x0, 'x0')
    input_checks.test_list_nparray(init_args, 'init_args')
    input_checks.test_int(vary_par_idx, 'vary_par_idx')
    input_checks.test_float_int(max_par, 'max_par')
    input_checks.test_int(num_steps, 'num_steps')
    input_checks.test_function(discretisation, 'discretisation')
    input_checks.test_function(solver, 'solver')
    if phase_con != None:
        input_checks.test_function(phase_con, 'phase_con')

    def make_args(phase_con, init_args, vary_par_idx, new_par):
        init_args[vary_par_idx] = new_par
        if phase_con != None:
            args = (phase_con, init_args, init_args)
        else:
            args = (init_args)
        return args
    
    pars = np.linspace(init_args[vary_par_idx], max_par, num_steps)

    # INITIALISE OUTPUTS
    x1 = solver(discretisation(func), x0, args=make_args(phase_con, init_args, vary_par_idx, pars[0]))
    x2 = solver(discretisation(func), x1, args=make_args(phase_con, init_args, vary_par_idx, pars[1]))
    u_stor = [x1, x2]
    par_stor = [pars[0], pars[1]]

    i = 2
    while i < num_steps:
        # RETRIEVE LAST TWO VALUES
        x_0, x_1 = u_stor[-2], u_stor[-1]
        p_0, p_1 = par_stor[-2], par_stor[-1]

        # SECANT
        dx = x_1 - x_0
        dp = p_1 - p_0

        # ARCLENGTH EQUATION
        x_pred = x_1 + dx
        p_pred = p_1 + dp
        preds = np.append(x_pred, p_pred)

        # DEFINE AUGMENTED PROBLEM
        def aug_prob(x0, func, vary_par_idx, dx, dp, discretisation, phase_con, init_args):
            x = x0[:-1]
            p = x0[-1]
            init_args = make_args(phase_con, init_args, vary_par_idx, p)
            d = discretisation(func)
            if phase_con != None:
                g = d(x, init_args[0], init_args[1], init_args[2])
            else:
                g = d(x, init_args)
            arc = np.dot(x - x_pred, dx) + np.dot(p - p_pred, dp)
            return np.append(g, arc)
        
        # SOLVE
        result = solver(aug_prob, preds, args=(func, vary_par_idx, dx, dp, discretisation, phase_con, init_args))
        u_stor.append(result[:-1])
        par_stor.append(result[-1])

        i += 1

    return np.vstack([np.array(u_stor).T, np.array(par_stor)])
