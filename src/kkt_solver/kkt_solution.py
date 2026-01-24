import sympy as sp
from dataclasses import dataclass, field
import colorama
import textwrap


type VariableMapping = dict[str, float]


@dataclass(frozen=True)
class KKTSolution:
    value: sp.Expr | float
    vars: VariableMapping = field(default_factory=dict)
    lambdas: VariableMapping = field(default_factory=dict)
    multipliers: VariableMapping = field(default_factory=dict)

    # equality override for checking solutions
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KKTSolution):
            return False

        TOLELRANCE = 1e-8

        def float_compare(a, b):
            return abs(float(a) - float(b)) < TOLELRANCE

        if not float_compare(self.get_value(), other.get_value()):
            return False

        for k in self.vars:
            if not float_compare(self.vars.get(k, 0), other.vars.get(k, 0)):
                return False

        for k in self.lambdas:
            if not float_compare(self.lambdas.get(k, 0), other.lambdas.get(k, 0)):
                return False

        for k in self.multipliers:
            if not float_compare(
                self.multipliers.get(k, 0), other.multipliers.get(k, 0)
            ):
                return False

        return True

    def get_value(self):
        # Always return numeric float
        return (
            float(self.value.evalf()) if isinstance(self.value, sp.Expr) else self.value
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
