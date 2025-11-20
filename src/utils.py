import sympy as sp


def compare_float(v1: sp.Expr | float, v2: sp.Expr | float):
    def get_float(v: sp.Expr | float) -> float:
        return float(v.evalf()) if isinstance(v, sp.Expr) else float(v)

    return get_float(v1) == get_float(v2)


def compare_dict_var(d1: dict[str, sp.Expr | float], d2: dict[str, sp.Expr | float]):
    for sym_name, v in d1.items():
        v2 = d2.get(sym_name)
        if v2 is None or not compare_float(v, v2):
            print(f"found none maching var: {sym_name} -> {v} != {v2} ")
            return False
    return True


def compute_grad(f: sp.Expr, f_symbols: list[sp.Symbol]):
    partials = []
    for s in f_symbols:
        partials.append(sp.diff(f, s))
    grad = sp.Matrix(partials)
    return grad
