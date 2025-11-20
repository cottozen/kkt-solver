from typing import Any
import sympy as sp
import numpy as np
import colorama
from kkt_solver.kkt_solution import KKTSolution
from kkt_solver.utils import compute_grad


class KKTSolver:
    def __init__(
        self,
        f: sp.Expr,
        f_symbols: list[sp.Symbol],
        constraint_inequalities: list[sp.Expr] | None = None,
        constraint_equalities: list[sp.Expr] | None = None,
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
        allow_numeric : bool, optional
            If True (default), the solver will attempt to use a numerical
            root finder if the primary analytical solver fails. If False, only analytical
            solutions are returned.
        verbose : bool, optional

        """
        self.allow_numeric = allow_numeric
        self.verbose = verbose
        self.f = f
        self.f_symbols = f_symbols
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
        from a potential solution returned by solver
        """
        sol_vars: dict = {
            f_symbol.name: potential_sol[f_symbol] for f_symbol in self.f_symbols
        }
        sol_lams: dict = {lam_i.name: potential_sol[lam_i] for lam_i in self.lambdas}
        sol_muls: dict = {
            mul_i.name: potential_sol[mul_i] for mul_i in self.multipliers
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
            results = sp.nsolve(M.tolist(), symbols, v0, dict=True)
            return results

    def _define_equations(self, minimize: bool):
        f_grad = compute_grad(self.f, self.f_symbols)
        F = f_grad
        if not minimize:
            F = -F

        # dont know what to call this so its 'F'
        equations = []

        # contruct L(v) = 0
        if len(self.constraint_equalities) or len(self.constraint_inequalities):
            # if objective is minimization we optimize f(v) otherwise we optimize -f(v)
            for lam_i, g_i in zip(self.lambdas, self.constraint_inequalities):
                F += lam_i * compute_grad(g_i, self.f_symbols)
            for mul_i, g_i in zip(self.multipliers, self.constraint_equalities):
                F += mul_i * compute_grad(g_i, self.f_symbols)
            equations.append(F)

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

    def verify(self, values: dict[str, sp.Expr | float], minimize: bool = True):
        """
        Verifies if values is a valid optomal for the optimization problem

        Parameters
        ----------
        values:  dict[str, sp.Expr | float]
            values of the proposed solution
        minimize: bool, optional
            Determines the goal of the optimization.
            - If True (default): optimizes min f(v)
            - If False(default): optimizes min -f(v)
        Returns
        True if values is a valid solution to the optimization problem based on the KKT conditions
        """
        M = self._define_equations(minimize)
        results = self._solve_equations(M.subs(values), self.multipliers + self.lambdas)
        for potential_sol in results:
            sol = KKTSolution(
                vars=values,
                lambdas={lam_i.name: potential_sol[lam_i] for lam_i in self.lambdas},
                multipliers={
                    mul_i.name: potential_sol[mul_i] for mul_i in self.multipliers
                },
                value=self.f.subs(values),
            )
            is_valid, error = sol.verify_constraints(
                self.constraint_inequalities, self.constraint_equalities
            )
            if not is_valid:
                return False
        return True

    def solve(self, minimize: bool = True):
        """
        Solves the constrained optimization problem by finding all points that
        satisfy the KKT conditions.

        Parameters
        ----------
        minimize : bool, optional
            Determines the goal of the optimization.
            - If True (default): optimizes min f(v)
            - If False(default): optimizes min -f(v)
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
        M = self._define_equations(minimize)

        min_v = float("inf")
        max_v = float("-inf")
        solutions: list[KKTSolution] = []

        results = self._solve_equations(M, self.all_symbols)
        # find valid solution with KKT conditions
        for potential_sol in results:
            sol_vars, sol_lams, sol_muls = self._extract_symbol_values(potential_sol)

            assert len(sol_lams) == len(self.constraint_inequalities)
            assert len(sol_muls) == len(self.constraint_equalities)
            assert len(sol_vars) == len(self.f_symbols)

            v = self.f.subs(sol_vars)

            sol = KKTSolution(
                vars=sol_vars, lambdas=sol_lams, multipliers=sol_muls, value=v
            )
            is_valid, error = sol.verify_constraints(
                self.constraint_inequalities, self.constraint_equalities
            )
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
        optimal_value = min_v if minimize else max_v
        optimals: list[KKTSolution] = self._filter_for_optimum(optimal_value, solutions)
        return optimals
