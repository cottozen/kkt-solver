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

    def __repr__(self):
        return f"Solution(val={self.value:.4f}, vars={self.vars}, lambdas={self.lambdas}, multipliers={self.multipliers})"
