import sympy as sp
from dataclasses import dataclass, field
import colorama
import textwrap
from kkt_solver import utils


@dataclass(frozen=True)
class KKTSolution:
    value: sp.Expr | float
    vars: dict[str, sp.Expr | float] = field(default_factory=dict)
    lambdas: dict[str, sp.Expr | float] = field(default_factory=dict)
    multipliers: dict[str, sp.Expr | float] = field(default_factory=dict)

    def __hash__(self):
        items = [
            *sorted(self.vars.items()),
            *sorted(self.lambdas.items()),
            *sorted(self.multipliers.items()),
        ]
        return hash(tuple(items))

    # equality override for checking solutions
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, KKTSolution):
            return False

        return (
            utils.compare_float(self.value, value.value)
            and utils.compare_dict_var(self.vars, value.vars)
            and utils.compare_dict_var(self.lambdas, value.lambdas)
        )

    def display_optimal_solution(self):
        print(
            colorama.Fore.GREEN
            + textwrap.dedent(f"""
            OPTIMAL SOLUTION FOUND:
            value: {self.value},
            variables: {self.vars},
            lambdas: {self.lambdas} 
            """)
        )

    def display_solution(self):
        print(
            colorama.Fore.GREEN
            + textwrap.dedent(f"""
            SOLUTION FOUND:
            value: {self.value},
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
            value: {self.value},
            variables: {self.vars},
            lambdas: {self.lambdas} 
            """)
        )
