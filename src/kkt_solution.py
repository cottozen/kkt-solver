import sympy as sp
from dataclasses import dataclass, field
import colorama
import textwrap
from src import utils


@dataclass
class KKTSolution:
    value: sp.Expr | float
    vars: dict[str, sp.Expr | float] = field(default_factory=dict)
    lambdas: dict[str, sp.Expr | float] = field(default_factory=dict)
    multipliers: dict[str, sp.Expr | float] = field(default_factory=dict)

    # equality override for checking solutions
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, KKTSolution):
            return False

        return (
            utils.compare_float(self.value, value.value)
            and utils.compare_dict_var(self.vars, value.vars)
            and utils.compare_dict_var(self.lambdas, value.lambdas)
        )

    def verify_constraints(
        self,
        constraint_inequalities: list[sp.Expr],
        constraint_equalities: list[sp.Expr],
        bound: float = 0,
    ):
        # verify g_i constraints
        for g_i in constraint_inequalities:
            g_v = g_i.subs(self.vars)  # pyright: ignore
            if g_v > bound:
                return False, f"failed inequality  constraint: {g_i} with value: {g_v}"
        for g_i in constraint_equalities:
            g_v = g_i.subs(self.vars)  # pyright: ignore
            if g_v != bound:
                return False, f"failed equalitiy constraint: {g_i} with value: {g_v}"
        # verify lambda >= 0
        for l_i, v in self.lambdas.items():
            # lambda for equalities dont have to be greater than 0
            if v < 0:
                return False, f"failed lambda constraint: {l_i}"
        return True, "VERIFIED"

    def display_optimal_solution(self):
        print(
            colorama.Fore.GREEN
            + textwrap.dedent(f"""
            OPTIMAL SOLUTION FOUND:
            variables: {self.vars},
            lambdas: {self.lambdas} 
            """)
        )

    def display_solution(self):
        print(
            colorama.Fore.GREEN
            + textwrap.dedent(f"""
            SOLUTION FOUND:
            variables: {self.vars},
            lambdas: {self.lambdas} 
            """)
        )

    def display_invalid_solution(self, error: str):
        print(
            colorama.Fore.RED
            + textwrap.dedent(f"""
            FAILED INEQUALITY VERIFICATION:
            contraint: {error}
            variables: {self.vars},
            lambdas: {self.lambdas} 
            """)
        )
