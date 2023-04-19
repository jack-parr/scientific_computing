# These function are used to check the formats of inputs as functions are called, and return relevant error messages.
import numpy as np


def test_float_int(arg, arg_name):
    if type(arg) is int or type(arg) is float or type(arg) is np.float64:
        pass
    else:
        raise Exception('Argument ('+arg_name+') must be either a float or integer value.')


def test_int(arg, arg_name):
    if type(arg) is int:
        pass
    else:
        raise Exception('Argument ('+arg_name+') must be an integer value.')


def test_list_nparray(arg, arg_name):
    if type(arg) is list or type(arg) is np.ndarray:
        pass
    else:
        raise Exception('Argument ('+arg_name+') must be either a list or a numpy.ndarray.')


def test_function(arg, arg_name):
    if callable(arg) and type(arg) is not type:
        pass
    else:
        raise Exception('Argument ('+arg_name+') must be a callable function.')


def test_string(arg, arg_name):
    if type(arg) is str:
        pass
    else:
        raise Exception('Argument ('+arg_name+') must be a string.')
