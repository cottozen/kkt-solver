from typing import Any
import sympy as sp
from src.kkt_solution import KKTSolution


def compute_grad(f: sp.Expr, f_symbols: list[sp.Symbol]):
    partials = []
    for s in f_symbols:
        partials.append(sp.diff(f, s))
    grad = sp.Matrix(partials)
    return grad


class KKTSolver:
    def __init__(
        self,
        f: sp.Expr,
        f_symbols: list[sp.Symbol],
        constraint_inequalities: list[sp.Expr] | None = None,
        constraint_equalities: list[sp.Expr] | None = None,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.f = f
        self.f_symbols = f_symbols
        if constraint_inequalities is None:
            constraint_inequalities = []
        if constraint_equalities is None:
            constraint_equalities = []
        self.constraint_inequalities = constraint_inequalities
        self.constraint_equalities = constraint_equalities

        # define lagrangian multipliers for equalities
        self.multipliers: list[sp.Symbol] = []
        # define lambdas for inequalities
        self.lambdas: list[sp.Symbol] = []

        for i, g_i in enumerate(self.constraint_inequalities):
            self.lambdas.append(sp.Symbol(f"lam_{i + 1}"))

        for i, g_i in enumerate(self.constraint_equalities):
            self.multipliers.append(sp.Symbol(f"mul_{i + 1}"))

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

    def _solve_equations(self, M: sp.Matrix):
        all_symbols = [*self.f_symbols, *self.lambdas, *self.multipliers]
        results = sp.solve(M, all_symbols, dict=True)
        return results

    def _define_equations(self):
        f_grad = compute_grad(self.f, self.f_symbols)

        # dont know what to call this so its 'F'
        equations = []

        if len(self.constraint_equalities) or len(self.constraint_inequalities):
            # contruct L(v) = 0
            F = f_grad
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

    def _filter_for_optimum(self, min_v: sp.Expr | float, solutions: list[KKTSolution]):
        optimals: list[KKTSolution] = []
        # return optimals
        for sol in solutions:
            if sol.value == min_v:
                sol.display_optimal_solution()
                optimals.append(sol)
        return optimals

    def solve(self):
        if self.verbose:
            print("SOLVING: ", self.f)
            print("VARIABLES: ", self.f_symbols)
            print("INEQUALITY CONSTRAINTS: ", self.constraint_inequalities)
            print("EQUALITY CONSTRAINTS: ", self.constraint_equalities)

        # define matrix of equations to solve
        M = self._define_equations()

        min_v = float("inf")
        solutions: list[KKTSolution] = []

        results = self._solve_equations(M)
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

        optimals: list[KKTSolution] = self._filter_for_optimum(min_v, solutions)
        return optimals
