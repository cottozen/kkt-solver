import sympy as sp
import unittest
import sys
import os

from kkt_solver.kkt_solver import PointType

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "src"))

from kkt_solver import KKTSolver, KKTSolution


class KKTSolverTests(unittest.TestCase):
    """
    Tests KKT solver, expected solutions have been verified with WolframAlpha
    """

    def testSolve1(self):
        x, y = sp.symbols("x, y")
        # ------------------
        f = 2 * x**2 + y**2
        g_1 = 2 - x - y
        g_2 = y - 2
        g_3 = x - 3
        g_4 = 1 - y
        expected = [
            KKTSolution(
                vars={"x": 2 / 3, "y": 4 / 3},
                lambdas={"lam_1": 8 / 3, "lam_2": 0, "lam_3": 0, "lam_4": 0},
                value=8 / 3,
            )
        ]
        solver = KKTSolver(f, [x, y], [g_1, g_2, g_3, g_4])
        optimals = solver.solve()
        assert optimals == expected
        for opt in optimals:
            assert solver.verify(opt.vars)
            # convex optimzation problem
            assert PointType.GLOBAL_MINIMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )

    def testSolve2(self):
        x, y = sp.symbols("x, y")
        # ------------------
        f = x**2 + y**2
        g_1 = 2 - x - y
        g_2 = y - 2
        g_3 = x - 3
        g_4 = 1 - y

        expected = [
            KKTSolution(
                vars={"x": 1, "y": 1},
                lambdas={"lam_1": 2, "lam_2": 0, "lam_3": 0, "lam_4": 0},
                value=2,
            )
        ]

        solver = KKTSolver(f, [x, y], [g_1, g_2, g_3, g_4])
        optimals = solver.solve()
        assert optimals == expected
        for opt in optimals:
            assert solver.verify(opt.vars)
            # convex optimzation problem
            assert PointType.GLOBAL_MINIMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )

    def testSolve3(self):
        x, y = sp.symbols("x, y")
        # ------------------
        f = x + 3 * y**2
        g_1 = x**2 + 2 * y**2 - 1
        g_2 = x + y - 1
        g_3 = y - x

        expected = [
            KKTSolution(
                vars={"x": -1 / 6, "y": -1 / 6},
                lambdas={
                    "lam_1": sp.Integer(0),
                    "lam_2": sp.Integer(0),
                    "lam_3": sp.Integer(1),
                },
                value=-1 / 12,
            )
        ]
        solver = KKTSolver(f, [x, y], [g_1, g_2, g_3])
        optimals = solver.solve()
        assert optimals == expected
        for opt in optimals:
            assert solver.verify(opt.vars)
            # convex optimzation problem
            assert PointType.GLOBAL_MINIMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )

    def testSolveWithEqualityContraint(self):
        x, y, z = sp.symbols("x, y, z")
        f = 2 * x**2 + y**2 + 3 * z**2
        # inequalities contraints
        g_1 = 2 - x
        g_2 = y + z - x
        g_3 = -y
        g_4 = -z
        # equalities contraints
        g_5 = x + z + y - 10

        expected = [
            KKTSolution(
                vars={"x": 5, "y": 15 / 4, "z": 5 / 4},
                lambdas={"lam_1": 0, "lam_2": 25 / 4, "lam_3": 0, "lam_4": 0},
                value=275 / 4,
            )
        ]
        solver = KKTSolver(
            f,
            [x, y, z],
            constraint_inequalities=[g_1, g_2, g_3, g_4],
            constraint_equalities=[g_5],
        )
        optimals = solver.solve()
        assert optimals == expected
        for opt in optimals:
            assert solver.verify(opt.vars)
            assert PointType.GLOBAL_MINIMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )

    def testSolveNumeric(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        f = x**2 + y**2
        f_symbols = [x, y]
        inequalities = [x + y - 1]
        equalities = [x - sp.cos(y)]

        expected = [
            KKTSolution(
                vars={"x": 1, "y": 0},
                lambdas={"lam_1": 0},
                value=1,
            )
        ]
        solver = KKTSolver(
            f=f,
            f_symbols=f_symbols,
            constraint_inequalities=inequalities,
            constraint_equalities=equalities,
        )
        optimals = solver.solve()
        assert optimals == expected
        for opt in optimals:
            assert solver.verify(opt.vars)
            # convex optimzation
            assert PointType.BOUNDARY_EXTREMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )

    def testSolveNoNumeric(self):
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        f = x**2 + y**2
        f_symbols = [x, y]
        inequalities = [x + y - 1]
        equalities = [x - sp.cos(y)]

        solver = KKTSolver(
            f=f,
            f_symbols=f_symbols,
            constraint_inequalities=inequalities,
            constraint_equalities=equalities,
            allow_numeric=False,
        )
        optimals = solver.solve()
        assert len(optimals) == 0

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
                lambdas={"lam_1": 3 / 2, "lam_2": 0, "lam_3": 0},
                value=5 / 3,
            )
        ]
        solver = KKTSolver(
            f, [x, y], [g_1, g_2, g_3], minimize=False, allow_numeric=False
        )
        optimals = solver.solve()
        assert optimals == expected, f"got: {optimals}"
        for opt in optimals:
            assert solver.verify(opt.vars)
            # convex optimzation problem
            assert PointType.GLOBAL_MAXIMUM == solver.get_point_type(opt), (
                f"got: {solver.get_point_type(opt)}"
            )


if __name__ == "__main__":
    unittest.main()
