# %%
import unittest
import ode_solver as solve_ode

# %%
class TestMethods(unittest.TestCase):

    def testEulerStep(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()

# %%
'''
tests needed:
use the predator prey model to run tests for each function.
once this file is working completely then write another file which applies relevant tests
to the numerical shooting code (probably using the hopf bifurcation normal form given on the website).

euler_step:
check the format and size of the output.
check any intended error messages are successfully thrown.
check the maximum timestep taken does not exceed deltat_max.

rk4_step:
same tests as euler_step.

solve_to:
check the format and size of the output (make the output format more nicely into separate columns for each of the variables)
check any intended error messages are successfully thrown.
'''