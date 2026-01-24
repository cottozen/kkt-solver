import itertools
from typing import Any
import sympy as sp
import enum
from kkt_solver.kkt_display import KKTDisplay
from kkt_solver.kkt_solution import KKTSolution, VariableMapping


def _compute_grad(f: sp.Expr, f_symbols: list[sp.Symbol]):
    grad = sp.Matrix([sp.diff(f, var) for var in f_symbols])
    return grad


class PointType(enum.Enum):
    GLOBAL_MINIMUM = enum.auto()
    GLOBAL_MAXIMUM = enum.auto()
    LOCAL_MINIMUM = enum.auto()
    LOCAL_MAXIMUM = enum.auto()
    SADDLE_POINT = enum.auto()
    BOUNDARY_EXTREMUM = enum.auto()
    CRITICAL_POINT = enum.auto()


class KKTSolverException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class KKTSolver:
    def __init__(
        self,
        f: sp.Expr,
        constraint_inequalities: list[sp.Expr] | None = None,
        constraint_equalities: list[sp.Expr] | None = None,
        minimize: bool = True,
        verbose: bool = True,
    ) -> None:
        self.minimize = minimize
        self.verbose = verbose
        self.f = f
        self.f_symbols = sorted(
            [s for s in f.free_symbols if isinstance(s, sp.Symbol)],
            key=lambda s: s.name,
        )

        self.constraint_inequalities = constraint_inequalities or []
        self.constraint_equalities = constraint_equalities or []
        constraint_symbols = set()
        for c in self.constraint_inequalities:
            constraint_symbols.update(
                [s for s in c.free_symbols if s not in self.f_symbols]
            )
        for c in self.constraint_equalities:
            constraint_symbols.update(
                [s for s in c.free_symbols if s not in self.f_symbols]
            )
        self.constraint_symbols = sorted(constraint_symbols, key=lambda s: s.name)

        self.f_hessian = sp.hessian(self.f, self.f_symbols)

        self.lambdas = [
            sp.Symbol(f"lambda_{i + 1}", real=True)
            for i in range(len(constraint_inequalities or []))
        ]
        self.multipliers = [
            sp.Symbol(f"mul_{i + 1}", real=True)
            for i in range(len(constraint_equalities or []))
        ]

        self.all_symbols = [
            *self.f_symbols,
            *self.constraint_symbols,
            *self.lambdas,
            *self.multipliers,
        ]

        self.symbol_map = {s.name: s for s in self.all_symbols}

        self.display = KKTDisplay(verbose)

    def _extract_symbol_values(self, potential_sol: dict[sp.Symbol, Any]):
        substitution_map: dict = {
            sym: 0 for sym in self.all_symbols if sym not in potential_sol
        }

        def resolve_expr_or_value(sym):
            if sym in potential_sol:
                val = potential_sol[sym]
                return val.subs(substitution_map) if hasattr(val, "subs") else val
            else:
                return substitution_map[sym]

        sol_vars: VariableMapping = {
            s.name: resolve_expr_or_value(s)
            for s in self.f_symbols + self.constraint_symbols
        }
        sol_lams: VariableMapping = {
            lam_i.name: resolve_expr_or_value(lam_i) for lam_i in self.lambdas
        }
        sol_muls: VariableMapping = {
            mul_i.name: resolve_expr_or_value(mul_i) for mul_i in self.multipliers
        }

        return sol_vars, sol_lams, sol_muls

    def _define_lagrangian(self):
        L = self.f if self.minimize else -self.f
        if len(self.constraint_equalities) or len(self.constraint_inequalities):
            for lam_i, g_i in zip(self.lambdas, self.constraint_inequalities):
                L += lam_i * g_i
            for mul_i, h_i in zip(self.multipliers, self.constraint_equalities):
                L += mul_i * h_i
        return L

    def _define_lagrangian_grad(self):
        L = self._define_lagrangian()
        L_grad = _compute_grad(L, self.f_symbols + self.constraint_symbols)
        return L_grad

    def construct_equations(self) -> tuple[sp.Matrix, list[str]]:
        """
        Construct a Matrix of equations AND return a list of string descriptions.
        """
        equations = []
        info_strings = []

        L_grad = self._define_lagrangian_grad()
        equations.append(L_grad)

        for i, sym in enumerate(self.f_symbols + self.constraint_symbols):
            if i < len(L_grad):
                info_strings.append(
                    f"Stationarity (d L / d {sym.name}):  {L_grad[i]} = 0"
                )

        # Complementary Slackness
        for lam_i, g_i in zip(self.lambdas, self.constraint_inequalities):
            eq = lam_i * g_i
            equations.append(eq)
            info_strings.append(f"Complementary Slackness: {eq} = 0")

        for h_i in self.constraint_equalities:
            equations.append(h_i)
            info_strings.append(f"Equality Constraint: {h_i} = 0")

        M = sp.Matrix(equations)
        return M, info_strings

    def get_kkt_conditions(self):
        """Returns the matrix of equations"""
        M, _ = self.construct_equations()
        return M

    def _solve_matrix(self, M: sp.Matrix):
        symbols_to_solve = (
            self.f_symbols + self.lambdas + self.multipliers + self.constraint_symbols
        )
        symbols_to_solve = [s for s in symbols_to_solve if s in M.free_symbols]
        try:
            results = sp.solve(M, symbols_to_solve, dict=True)
            return results
        except NotImplementedError as e:
            self.display.print_solver_error(e)
            raise KKTSolverException(
                "Failed to analytically solve optimization problem"
            )

    def _solve_equations_iter(self):
        """
        Solves equations iteratively by checking all combinations of lambdas
        """
        L_grad = self._define_lagrangian_grad()
        combinations = itertools.product(*[(0, 1) for _ in self.lambdas])

        all_results: list[tuple[list[str], list[Any]]] = []

        for case_idx, c in enumerate(combinations):
            equations = []
            inactive_lambas = {}
            desc_parts = []

            # Always add Equality constraints
            equations += self.constraint_equalities

            for i, active in enumerate(c):
                if active:
                    # Active: g(x) = 0
                    equations.append(self.constraint_inequalities[i])
                    desc_parts.append(f"g{i + 1} Active")
                else:
                    # Inactive: lambda_i = 0
                    inactive_lambas[self.lambdas[i]] = 0
                    desc_parts.append(f"g{i + 1} Slack")

            # Substitute inactive lambdas into gradient equations
            equations.append(L_grad.subs(inactive_lambas))

            self.display.print_case(case_idx, desc_parts)

            M = sp.Matrix(equations)
            results = (desc_parts, self._solve_matrix(M))
            all_results.append(results)

        return all_results

    def _sub_solutions_variables(
        self, expression: sp.Expr, *var_dicts: VariableMapping
    ):
        subs_map: dict = {
            (self.symbol_map.get(k, k) if not isinstance(k, sp.Symbol) else k): v
            for d in var_dicts
            for k, v in d.items()
        }
        return expression.subs(subs_map)

    def _filter_for_optimum(
        self, optimal_value: sp.Expr | float, solutions: list[KKTSolution]
    ):
        optimals: list[KKTSolution] = []
        for sol in solutions:
            if abs(float(sol.value) - float(optimal_value)) < 1e-6:
                optimals.append(sol)
        return optimals

    def verify_constraints(self, sol: KKTSolution):
        # verify g_i constraints
        for g_i in self.constraint_inequalities:
            g_v = self._sub_solutions_variables(g_i, sol.vars, sol.lambdas).evalf()
            if g_v > 1e-6:
                return False, f"failed inequality constraint: {g_i} with value: {g_v}"
        for h_i in self.constraint_equalities:
            h_v = self._sub_solutions_variables(h_i, sol.vars, sol.multipliers).evalf()
            if abs(h_v) > 1e-6:
                return False, f"failed equality constraint: {h_i} with value: {h_v}"
        # verify lambda >= 0
        for l_i, v in sol.lambdas.items():
            if v < -1e-6:
                return False, f"failed lambda constraint: {l_i} (val={v:.3f}) < 0"
        return True, "VERIFIED"

    def has_active_constraints(self, sol: KKTSolution):
        if len(self.constraint_equalities) > 0:
            return True
        for g_i in self.constraint_inequalities:
            # We must use proper substitution map here
            g_v = self._sub_solutions_variables(g_i, sol.vars, sol.lambdas).evalf()
            # If g(x) is close to 0, it is active
            if abs(g_v) < 1e-6:
                return True
        return False

    def is_convex_function(self) -> bool:
        return self.f_hessian.is_positive_semidefinite is True

    def is_convex_subset(self) -> bool:
        n = len(self.f_symbols)
        zero_matrix = sp.zeros(n, n)

        # simple check for convex subset where the constraints are linear and hessian matrix is 0
        eq_is_convex_subset = all(
            [
                sp.hessian(h_i, self.f_symbols).equals(zero_matrix)
                for h_i in self.constraint_equalities
            ]
        )

        # Inequality constraints must be convex
        inq_is_convex_subset = all(
            [
                sp.hessian(g_i, self.f_symbols).is_positive_semidefinite
                for g_i in self.constraint_inequalities
            ]
        )
        return eq_is_convex_subset and inq_is_convex_subset

    def is_convex_problem(self):
        """Checks if both function and set are convex."""
        return self.is_convex_subset() and self.is_convex_function()

    def get_point_type(self, sol: KKTSolution):
        """
        Determines the classification (Min, Max, Saddle) for the solution.
        """
        if self.is_convex_problem():
            if self.minimize:
                return PointType.GLOBAL_MINIMUM
            return PointType.GLOBAL_MAXIMUM

        L = self._define_lagrangian()
        L_hessian = sp.hessian(L, self.f_symbols)

        full_map = {**sol.vars, **sol.lambdas, **sol.multipliers}
        subs_dict = {}
        for name, val in full_map.items():
            if name in self.symbol_map:
                subs_dict[self.symbol_map[name]] = val

        L_hessian_v = L_hessian.subs(subs_dict).evalf()
        if L_hessian_v.is_positive_definite:
            return PointType.LOCAL_MINIMUM
        if L_hessian_v.is_negative_definite:
            return PointType.LOCAL_MAXIMUM

        if self.has_active_constraints(sol):
            return PointType.BOUNDARY_EXTREMUM

        if L_hessian_v.is_indefinite:
            return PointType.SADDLE_POINT

        return PointType.CRITICAL_POINT

    def verify(self, values: VariableMapping):
        M, _ = self.construct_equations()

        results = self._solve_matrix(self._sub_solutions_variables(M, values))
        for potential_sol in results:
            sol = KKTSolution(
                vars=values,
                lambdas={lam_i.name: potential_sol[lam_i] for lam_i in self.lambdas},
                multipliers={
                    mul_i.name: potential_sol[mul_i] for mul_i in self.multipliers
                },
                value=self._sub_solutions_variables(self.f, values).evalf(),
            )
            is_valid, error = self.verify_constraints(sol)
            if not is_valid:
                self.display.display_invalid_solution(sol, error)
                return False
        return True

    def solve(self):
        min_v = float("inf")
        max_v = float("-inf")
        solutions: list[KKTSolution] = []

        results = self._solve_equations_iter()
        seen = set()

        for case_idx, (description, case_solutions) in enumerate(results):
            self.display.print_case(case_idx, description)
            for potential_sol in case_solutions:
                sol_vars, sol_lams, sol_muls = self._extract_symbol_values(
                    potential_sol
                )  # pyright: ignore

                v = self._sub_solutions_variables(self.f, sol_vars).evalf()

                sol = KKTSolution(
                    vars={k: float(v) for k, v in sol_vars.items()},
                    lambdas={k: float(v) for k, v in sol_lams.items()},
                    multipliers={k: float(v) for k, v in sol_muls.items()},
                    value=float(v),
                )

                is_valid, error = self.verify_constraints(sol)

                if not is_valid:
                    if self.verbose:
                        self.display.display_invalid_solution(sol, error)
                    continue

                # Check duplicates
                key = (
                    tuple(sorted(sol.vars.items())),
                    tuple(sorted(sol.lambdas.items())),
                    tuple(sorted(sol.multipliers.items())),
                )
                if key in seen:
                    continue

                seen.add(key)
                solutions.append(sol)
                self.display.display_solution(sol)

                if v < min_v:
                    min_v = v
                if v > max_v:
                    max_v = v

        optimal_value = min_v if self.minimize else max_v
        optimals = self._filter_for_optimum(optimal_value, solutions)
        return optimals
