from decimal import InvalidOperation
from operator import mul
from typing import Any
import sympy as sp
import numpy as np
import colorama
from kkt_solver.kkt_solution import KKTSolution
from kkt_solver.utils import compute_grad
import enum


class PointType(enum.Enum):
    BOUNDARY_EXTREMUM = enum.auto()
    GLOBAL_MINIMUM = enum.auto()
    GLOBAL_MAXIMUM = enum.auto()
    LOCAL_MINIMUM = enum.auto()
    LOCAL_MAXIMUM = enum.auto()
    SADDLE_POINT = enum.auto()
    CRITICAL_POINT = enum.auto()


class KKTSolver:
    def __init__(
        self,
        f: sp.Expr,
        f_symbols: list[sp.Symbol],
        constraint_inequalities: list[sp.Expr] | None = None,
        constraint_equalities: list[sp.Expr] | None = None,
        minimize: bool = True,
        allow_numeric: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the KKT Solver for constrained optimization.

        This solver sets up the KKT necessary conditions using symbolic differentiation
        (via SymPy) and prepares the system for solving.

        Parameters
        ----------
        f : sympy.Expr
            The objective function to be minimized, f(v).
        f_symbols : list of sympy.Symbol
            The list of optimization variables [v1, v2, ...]. This defines the
            vector of variables v.
        constraint_inequalities : list of sympy.Expr, optional
            A list of inequality constraint expressions, g_i(v), where the problem
            is subject to g_i(v) <= 0. Default is None (no inequality constraints).
        constraint_equalities : list of sympy.Expr, optional
            A list of equality constraint expressions, h_j(v), where the problem
            is subject to h_j(v) = 0. Default is None (no equality constraints).
        minimize: bool, optional
            Determines the goal of the optimization.
            - If True (default): optimizes min f(v)
            - If False: optimizes min -f(v)
        allow_numeric : bool, optional
            If True (default), the solver will attempt to use a numerical
            root finder if the primary analytical solver fails. If False, only analytical
            solutions are returned.
        verbose : bool, optional

        """
        self.minimize = minimize
        self.allow_numeric = allow_numeric
        self.verbose = verbose
        self.f = f
        self.f_symbols = f_symbols
        self.f_hessian: sp.Matrix = sp.hessian(self.f, self.f_symbols)
        if constraint_inequalities is None:
            constraint_inequalities = []
        if constraint_equalities is None:
            constraint_equalities = []
        self.constraint_inequalities = constraint_inequalities
        self.constraint_equalities = constraint_equalities

        # define lambdas for inequalities
        self.lambdas: list[sp.Symbol] = [
            sp.Symbol(f"lam_{i + 1}")
            for i, g_i in enumerate(self.constraint_inequalities)
        ]
        # define lagrangian multipliers for equalities
        self.multipliers: list[sp.Symbol] = [
            sp.Symbol(f"mul_{i + 1}")
            for i, g_i in enumerate(self.constraint_equalities)
        ]

        self.all_symbols = [*self.f_symbols, *self.lambdas, *self.multipliers]

    def _extract_symbol_values(self, potential_sol: dict[sp.Symbol, Any]):
        """
        extract function variables, lambdas and multipliers
        from a potential solution returned by solver.
        Handles missing keys (free variables) by setting them to 0.
        """

        substitution_map: dict = {
            sym: 0 for sym in self.all_symbols if sym not in potential_sol
        }

        def resolve_expr_or_value(sym):
            if sym in potential_sol:
                val = potential_sol[sym]
                # If the value is an expression substitute the free vars
                return val.subs(substitution_map) if isinstance(val, sp.Expr) else val
            else:
                return substitution_map[sym]

        sol_vars = {
            f_symbol.name: resolve_expr_or_value(f_symbol)
            for f_symbol in self.f_symbols
        }

        sol_lams = {lam_i.name: resolve_expr_or_value(lam_i) for lam_i in self.lambdas}

        sol_muls = {
            mul_i.name: resolve_expr_or_value(mul_i) for mul_i in self.multipliers
        }

        return sol_vars, sol_lams, sol_muls

    def _solve_equations(self, M: sp.Matrix, symbols: list[sp.Symbol]):
        try:
            # Attempt the analytical solve
            results = sp.solve(M, symbols, dict=True)
            return results
        except NotImplementedError as e:
            # Handle the case where SymPy fails analytically
            if self.verbose:
                print(colorama.Fore.RED + f"ANALYTICAL SOLVER FAILED: {e}")
            if not self.allow_numeric:
                print("Returning no solutions!")
                return []

            print(colorama.Fore.WHITE + "Trying numeric root finding (Newton method)!")
            v0 = np.zeros(len(symbols))
            # try numeric solve
            results = sp.nsolve(M.tolist(), symbols, v0, dict=True)
            return results

    def _define_equations(self):
        f_grad = compute_grad(self.f, self.f_symbols)

        # contruct L(v) = 0
        L = f_grad
        # if objective is minimization we optimize f(v) otherwise we optimize -f(v)
        if not self.minimize:
            L = -L

        equations = []

        if len(self.constraint_equalities) or len(self.constraint_inequalities):
            for lam_i, g_i in zip(self.lambdas, self.constraint_inequalities):
                L += lam_i * compute_grad(g_i, self.f_symbols)
            for mul_i, g_i in zip(self.multipliers, self.constraint_equalities):
                L += mul_i * compute_grad(g_i, self.f_symbols)
            equations.append(L)

        # add complementary slackness equations for inequalities
        for lam_i, g_i in zip(self.lambdas, self.constraint_inequalities):
            equations.append(lam_i * g_i)

        # add equality contraint equations:
        for g_i in zip(self.constraint_equalities):
            equations.append(g_i)
        M = sp.Matrix(equations)
        return M

    def _filter_for_optimum(
        self, optimal_value: sp.Expr | float, solutions: list[KKTSolution]
    ):
        optimals: list[KKTSolution] = []
        # return optimals
        for sol in solutions:
            if sol.value == optimal_value:
                sol.display_optimal_solution()
                optimals.append(sol)
        return optimals

    def verify_constraints(self, sol: KKTSolution):
        bound = 0
        # verify g_i constraints
        for g_i in self.constraint_inequalities:
            g_v = g_i.subs(sol.vars)  # pyright: ignore
            if g_v > bound:
                return False, f"failed inequality  constraint: {g_i} with value: {g_v}"
        for g_i in self.constraint_equalities:
            g_v = g_i.subs(sol.vars)  # pyright: ignore
            if g_v != bound:
                return False, f"failed equalitiy constraint: {g_i} with value: {g_v}"
        # verify lambda >= 0
        for l_i, v in sol.lambdas.items():
            # lambda for equalities dont have to be greater than 0
            if v < 0:
                return False, f"failed lambda constraint: {l_i}"
        return True, "VERIFIED"

    def has_active_constraints(self, sol: KKTSolution):
        for g_i in self.constraint_inequalities:
            g_v = g_i.subs(sol.vars)  # pyright: ignore
            if g_v == 0:
                return True
        if len(self.constraint_equalities) > 0:
            return True
        return False

    def is_convex_function(self) -> bool:
        return self.f_hessian.is_positive_semidefinite is True

    def is_convex_subset(self) -> bool:
        n = len(self.f_symbols)
        zero_matrix = sp.zeros(n, n)

        # for the equalitiy constraints h(v) = 0 we need to check that they are affine
        # meaning that h can be written as: h(v) = A.T * v + b
        # We check that h is affine by checkking if the hessian of h is the zero matrix.
        eq_is_convex_subset = all(
            [
                # compare to  zero matrix
                sp.hessian(g_i, self.f_symbols).equals(zero_matrix)
                for g_i in self.constraint_equalities
            ]
        )

        # for the inequality we use the fact that g(v) <= a is a convex subset if g is convex,
        # so we check that all inequality constraints are convex and then the union will be convex
        inq_is_convex_subset = all(
            [
                sp.hessian(g_i, self.f_symbols).is_positive_semidefinite
                for g_i in self.constraint_inequalities
            ]
        )
        # if both subsets are convex then the unions is convex
        return eq_is_convex_subset and inq_is_convex_subset

    def is_convex_problem(self):
        """
        Checks if the function is convex and that the subset we are optimizing over is convex
        """
        return self.is_convex_subset() and self.is_convex_function()

    def get_point_type(self, sol: KKTSolution):
        """
        Determines the critical point type for the provided solution
        """

        v: sp.Matrix = self.f_hessian.subs(sol.vars)

        if self.is_convex_problem():
            if self.minimize:
                return PointType.GLOBAL_MINIMUM
            return PointType.GLOBAL_MAXIMUM
        if self.has_active_constraints(sol):
            return PointType.BOUNDARY_EXTREMUM
        if v.is_positive_definite:
            return PointType.LOCAL_MINIMUM
        if v.is_negative_definite:
            return PointType.LOCAL_MAXIMUM
        if v.is_indefinite:
            return PointType.SADDLE_POINT
        return PointType.CRITICAL_POINT

    def verify(self, values: dict[str, sp.Expr | float]):
        """
        Verifies if values is a valid optomal for the optimization problem

        Parameters
        ----------
        values:  dict[str, sp.Expr | float]
            values of the proposed solution
        Returns
        True if values is a valid solution to the optimization problem based on the KKT conditions
        """
        M = self._define_equations()
        results = self._solve_equations(M.subs(values), self.multipliers + self.lambdas)
        for potential_sol in results:
            sol = KKTSolution(
                vars=values,
                lambdas={lam_i.name: potential_sol[lam_i] for lam_i in self.lambdas},  # pyright: ignore
                multipliers={
                    mul_i.name: potential_sol[mul_i]  # pyright: ignore
                    for mul_i in self.multipliers
                },
                value=self.f.subs(values),  # pyright: ignore
            )
            is_valid, error = self.verify_constraints(sol)
            if not is_valid:
                sol.display_invalid_solution(error)
                return False
        return True

    def solve(self):
        """
        Solves the constrained optimization problem by finding all points that
        satisfy the KKT conditions.

        ----------
        Returns
        -------
        list of KKTSolution
            A list containing all unique, feasible KKT solutions found.
            Each KKTSolution object represents a critical point (candidate for
            a local minimum, local maximum, or saddle point)
        """
        if self.verbose:
            print(f"SOLVING: {self.f}")
            print(f"VARIABLES: {self.f_symbols}")
            print(f"INEQUALITY CONSTRAINTS: {self.constraint_inequalities}")
            print(f"EQUALITY CONSTRAINTS: {self.constraint_equalities}")

        # define matrix of equations to solve
        M = self._define_equations()

        min_v = float("inf")
        max_v = float("-inf")
        solutions: list[KKTSolution] = []

        results = self._solve_equations(M, self.all_symbols)
        # find valid solution with KKT conditions
        for potential_sol in results:
            sol_vars, sol_lams, sol_muls = self._extract_symbol_values(potential_sol)  # pyright: ignore

            assert len(sol_lams) == len(self.constraint_inequalities)
            assert len(sol_muls) == len(self.constraint_equalities)
            assert len(sol_vars) == len(self.f_symbols)

            v = self.f.subs(sol_vars)

            sol = KKTSolution(
                vars=sol_vars, lambdas=sol_lams, multipliers=sol_muls, value=v
            )
            is_valid, error = self.verify_constraints(sol)
            if not is_valid:
                if self.verbose:
                    sol.display_invalid_solution(error)
                continue
            if self.verbose:
                sol.display_solution()
            solutions.append(sol)
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v

        # select min or max depending on optimization objective
        optimal_value = min_v if self.minimize else max_v
        optimals: list[KKTSolution] = self._filter_for_optimum(optimal_value, solutions)
        return optimals
