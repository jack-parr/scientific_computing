# %%
import unittest
import numpy as np
import ode_solver as solve_ode

# %%
def pred_prey(x, t, args):
    a, b, d = args
    x1, x2 = x
    dxdt = np.array([(x1*(1-x1)) - ((a*x1*x2)/(d+x1)), b*x2*(1-(x2/x1))])
    return dxdt

# %%
class TestMethods(unittest.TestCase):

    def testEulerStep(self):
        x0 = [1.5, 1]
        t0 = 0
        delta_t = 1
        x_pred, t = solve_ode.euler_step(pred_prey, x0, t0, delta_t, args=[1, 0.2, 0.1])
        self.assertEqual(len(x_pred), 2)
        self.assertEqual(round(x_pred[0], 3), -0.188)
        self.assertEqual(round(x_pred[1], 3), 1.067)
        self.assertEqual(int(t), 1)
    

    def testRK4Step(self):
        x0 = [1.5, 1]
        t0 = 0
        delta_t = 1
        x_pred, t = solve_ode.rk4_step(pred_prey, x0, t0, delta_t, args=[1, 0.2, 0.1])
        self.assertEqual(len(x_pred), 2)
        self.assertEqual(round(x_pred[0], 3), -1.383)
        self.assertEqual(round(x_pred[1], 3), 1.114)
        self.assertEqual(int(t), 1)
    

    def testSolveTo(self):
        x1 = [1.5, 1]
        t1 = 0
        t2 = 10
        for method in ['euler', 'rk4']:
            x_pred = solve_ode.solve_to(pred_prey, method, x1, t1, t2, 0.127, args=[1, 0.2, 0.1])
            self.assertEqual(len(x_pred), 80)
            self.assertTrue(np.isclose(x_pred[-1][-1], 10))

            x_pred = solve_ode.solve_to(pred_prey, 'rk4', x1, t1, t2, 0.01, args=[1, 0.2, 0.1])
            self.assertEqual(len(x_pred), 1001)
            self.assertTrue(np.isclose(x_pred[-1][-1], 10))
            self.assertEqual(round(x_pred[-1][0], 3), 0.516)
            self.assertEqual(round(x_pred[-1][1], 3), 0.103)


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