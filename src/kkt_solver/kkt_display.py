import sympy as sp
import colorama
from colorama import Fore, Style
import textwrap
from kkt_solver.kkt_solution import KKTSolution

colorama.init()


class KKTDisplay:
    verbose: bool

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def print_header(self, text: str):
        if self.verbose:
            print(f"\n{Style.BRIGHT}{Fore.CYAN}{'=' * len(text)}")
            print(f"{text}")
            print(f"{'=' * len(text)}{Style.RESET_ALL}")

    def print_math(self, label: str, expr):
        if self.verbose:
            print(f"{Fore.YELLOW}{label}{Style.RESET_ALL}")
            sp.pprint(expr)
            print("")

    def print_case(self, case_idx: int, desc_parts: list[str]):
        case_desc = ", ".join(desc_parts) if desc_parts else "Unconstrained Case"
        if self.verbose:
            print(f"{Fore.CYAN}Case {case_idx + 1}:{Style.RESET_ALL} {case_desc}")

    def print_solver_error(self, error: Exception):
        if self.verbose:
            print(Fore.RED + f"ANALYTICAL SOLVER FAILED: {error}")

    def print_optimals(self, optimals: list[KKTSolution]):
        if optimals:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Optimal Value: {optimals[0].get_value():.4f}{Style.RESET_ALL}"
            )
        else:
            print(f"{Fore.RED}No valid solutions found.{Style.RESET_ALL}")

    def display_optimal_solution(self, sol: KKTSolution):
        print(
            Fore.GREEN
            + textwrap.dedent(f"""
            OPTIMAL SOLUTION FOUND:
            {sol}
            """)
        )

    def display_solution(self, sol: KKTSolution):
        print(
            Fore.GREEN
            + textwrap.dedent(f"""
            KKT Conditions Satisfied:
            {sol}
            {Style.RESET_ALL}
            """)
        )

    def display_invalid_solution(self, sol: KKTSolution, error: str):
        print(
            Fore.RED
            + textwrap.dedent(f"""
            FAILED INEQUALITY VERIFICATION:
            contraint: {error}
            {sol}
            """)
        )
