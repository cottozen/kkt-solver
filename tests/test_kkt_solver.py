from math import exp
import sympy as sp
import unittest
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "src"))

from kkt_solver import KKTSolver, KKTSolution
from kkt_solver.kkt_solver import PointType


class KKTSolverTests(unittest.TestCase):
    """
    Tests KKT solver, expected solutions have been verified with WolframAlpha
    """

    def testSolve1(self):
        x, y = sp.symbols("x, y", real=True)
        # ------------------
        f = 2 * x**2 + y**2
        g_1 = 2 - x - y
        g_2 = y - 2
        g_3 = x - 3
        g_4 = 1 - y
        expected = [
            KKTSolution(
                vars={"x": 2 / 3, "y": 4 / 3},
                lambdas={
                    "lambda_1": 8 / 3,
                    "lambda_2": 0,
                    "lambda_3": 0,
                    "lambda_4": 0,
                },
                value=8 / 3,
            )
        ]
        solver = KKTSolver(f, [g_1, g_2, g_3, g_4])
        optimals = solver.solve()
        self.assertEqual(optimals, expected)
        for opt in optimals:
            self.assertTrue(solver.verify(opt.vars))
            self.assertEqual(solver.get_point_type(opt), PointType.GLOBAL_MINIMUM)

    def testSolve2(self):
        x, y = sp.symbols("x, y", real=True)
        # ------------------
        f = x**2 + y**2
        g_1 = 2 - x - y
        g_2 = y - 2
        g_3 = x - 3
        g_4 = 1 - y

        expected = [
            KKTSolution(
                vars={"x": 1, "y": 1},
                lambdas={
                    "lambda_1": 2,
                    "lambda_2": 0,
                    "lambda_3": 0,
                    "lambda_4": 0,
                },
                value=2,
            )
        ]

        solver = KKTSolver(f, [g_1, g_2, g_3, g_4])
        optimals = solver.solve()
        assert optimals == expected
        self.assertEqual(optimals, expected)
        for opt in optimals:
            self.assertTrue(solver.verify(opt.vars))
            self.assertEqual(solver.get_point_type(opt), PointType.GLOBAL_MINIMUM)

    def testSolve3(self):
        x, y = sp.symbols("x, y", real=True)
        # ------------------
        f = x + 3 * y**2
        g_1 = x**2 + 2 * y**2 - 1
        g_2 = x + y - 1
        g_3 = y - x

        expected = [
            KKTSolution(
                vars={"x": -1 / 6, "y": -1 / 6},
                lambdas={
                    "lambda_1": 0,
                    "lambda_2": 0,
                    "lambda_3": 1,
                },
                value=-1 / 12,
            )
        ]
        solver = KKTSolver(f, [g_1, g_2, g_3])
        optimals = solver.solve()
        self.assertEqual(optimals, expected)
        for opt in optimals:
            self.assertTrue(solver.verify(opt.vars))
            self.assertEqual(solver.get_point_type(opt), PointType.GLOBAL_MINIMUM)

    def testSolveWithEqualityContraint(self):
        x, y, z = sp.symbols("x, y, z", real=True)

        f = 2 * x**2 + y**2 + 3 * z**2
        # inequalities contraints
        g_1 = 2 - x
        g_2 = y + z - x
        g_3 = -y
        g_4 = -z
        # equalities contraints
        h_1 = x + z + y - 10

        expected = [
            KKTSolution(
                vars={"x": 5, "y": 15 / 4, "z": 5 / 4},
                lambdas={
                    "lambda_1": 0,
                    "lambda_2": 25 / 4,
                    "lambda_3": 0,
                    "lambda_4": 0,
                },
                multipliers={"mul_1": -55 / 4},
                value=275 / 4,
            )
        ]
        solver = KKTSolver(
            f,
            constraint_inequalities=[g_1, g_2, g_3, g_4],
            constraint_equalities=[h_1],
        )
        optimals = solver.solve()
        self.assertEqual(optimals, expected)
        for opt in optimals:
            self.assertTrue(solver.verify(opt.vars))
            self.assertEqual(solver.get_point_type(opt), PointType.GLOBAL_MINIMUM)

    def testSolveMaximize(self):
        x, y = sp.symbols("x, y")
        # ------------------
        f = x + 3 * y**2
        g_1 = x**2 + 2 * y**2 - 1
        g_2 = x + y - 1
        g_3 = y - x

        expected = [
            KKTSolution(
                vars={"x": 1 / 3, "y": -2 / 3},
                lambdas={"lambda_1": 3 / 2, "lambda_2": 0, "lambda_3": 0},
                value=5 / 3,
            )
        ]
        solver = KKTSolver(f, [g_1, g_2, g_3], minimize=False)
        optimals = solver.solve()
        self.assertEqual(optimals, expected)
        for opt in optimals:
            self.assertTrue(solver.verify(opt.vars))
            self.assertEqual(solver.get_point_type(opt), PointType.GLOBAL_MAXIMUM)

    def testNoneUniqueLambas(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        f = x**2 + y**2 + (z - 1) ** 2

        # Define the Constraints (g(v) <= 0 and h(v) = 0)
        g_1 = z - x
        g_2 = z + x
        g_3 = z - y
        g_4 = z + y
        g_5 = -1 - z

        expected = [
            KKTSolution(
                value=1,
                vars={"x": 0, "y": 0, "z": 0},
                lambdas={
                    "lambda_1": 0,
                    "lambda_2": 0,
                    "lambda_3": 1,
                    "lambda_4": 1,
                    "lambda_5": 0,
                },
                multipliers={},
            ),
            KKTSolution(
                value=1,
                vars={"x": 0, "y": 0, "z": 0},
                lambdas={
                    "lambda_1": 1,
                    "lambda_2": 1,
                    "lambda_3": 0,
                    "lambda_4": 0,
                    "lambda_5": 0,
                },
                multipliers={},
            ),
        ]

        solver = KKTSolver(
            f=f,
            constraint_inequalities=[g_1, g_2, g_3, g_4, g_5],
            constraint_equalities=[],
            verbose=True,
            minimize=True,
        )

        optimals = solver.solve()
        self.assertEqual(optimals, expected)

    def testSVM(self):
        a = sp.Symbol("a")
        b = sp.Symbol("b")
        c = sp.Symbol("c")

        f = a**2 + b**2

        # Define the Constraints (g(v) <= 0 and h(v) = 0)
        g_1 = 1 - a - b - c
        g_2 = 2 * a + 2 * b + c + 1

        expected = [
            KKTSolution(
                value=2,
                vars={"a": -1, "b": -1, "c": 3},
                lambdas={"lambda_1": 2, "lambda_2": 2},
                multipliers={},
            ),
        ]

        solver = KKTSolver(
            f=f,
            constraint_inequalities=[g_1, g_2],
            constraint_equalities=[],
            verbose=True,
            minimize=True,
        )
        optimals = solver.solve()
        self.assertEqual(optimals, expected)

    def test_global_minimum_convex(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        f = x**2 + y**2
        g1 = 1 - x

        solver = KKTSolver(f=f, constraint_inequalities=[g1], verbose=False)
        optimals = solver.solve()

        self.assertTrue(len(optimals) > 0, "Should find a solution")
        sol = optimals[0]

        self.assertAlmostEqual(float(sol.value), 1.0)

        point_type = solver.get_point_type(sol)
        self.assertEqual(point_type, PointType.GLOBAL_MINIMUM)

    def test_boundary_extremum_non_convex(self):
        x = sp.Symbol("x", real=True)
        f = -(x**2)
        g1 = x - 1
        g2 = -1 - x

        solver = KKTSolver(f=f, constraint_inequalities=[g1, g2], verbose=False)
        optimals = solver.solve()

        self.assertTrue(len(optimals) > 0)
        sol = optimals[0]
        point_type = solver.get_point_type(sol)
        self.assertEqual(point_type, PointType.LOCAL_MAXIMUM)

    def test_saddle_point(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        f = x**2 - y**2

        solver = KKTSolver(f=f, verbose=False)

        mock_sol = KKTSolution(
            vars={"x": 0, "y": 0}, lambdas={}, multipliers={}, value=0
        )

        point_type = solver.get_point_type(mock_sol)
        self.assertEqual(point_type, PointType.SADDLE_POINT)

    def test_exponential(self):
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)
        f = sp.exp(x) + sp.exp(y)

        g_1 = -x
        g_2 = -y
        g_3 = 1 - x - y

        solver = KKTSolver(
            f=f,
            constraint_inequalities=[g_1, g_2, g_3],
            verbose=False,
        )
        mock_sol = KKTSolution(
            vars={"x": 1 / 2, "y": 1 / 2},
            lambdas={"lambda_1": 0, "lambda_2": 0, "lambda_3": exp(1 / 2)},
            multipliers={},
            value=2 * sp.exp(1 / 2),
        )
        optimals = solver.solve()
        self.assertEqual(len(optimals), 1)
        self.assertEqual(optimals[0], mock_sol)


if __name__ == "__main__":
    unittest.main()
