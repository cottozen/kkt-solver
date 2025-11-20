## 🚀 KKT Solver for Constrained Optimization

This repository contains a Karush–Kuhn–Tucker (KKT) conditions-based solver designed for **analytical** non-linear optimization. It leverages the **SymPy** library to symbolically compute the gradients and solve the resulting system of KKT equations

Solves optimization problems of the form:

$$\Large \min_{v \in \mathbf{R}^n} f(v) $$
subject to:
$$\Large g_i(v) \le 0, \quad \text{for } i = 1, \dots, m$$
$$\Large h_j(v) = 0, \quad \text{for } j = 1, \dots, p$$

## ✨ Features

* **Hybrid Solving Capability** ⚙️: Performs an **Analytical/Symbolic** solve first. If that fails (e.g., due to transcendental equations), it automatically falls back to **Numerical Root Finding** if enabled via the `allow_numeric` flag.
* **Minimization & Maximization**: The solver handles both minimization and maximization goals.
  * **Minimization:** Solves the standard $\min f(\mathbf{v})$.
  * **Maximization:** Solves $\max f(\mathbf{v})$ by applying KKT conditions to the equivalent minimization problem: $\min -f(\mathbf{v})$.
* **Solution Verification** ✅: Includes a separate `verify` function to confirm if any arbitrary point is a valid KKT critical point by checking for the existence of feasible Lagrange multipliers.

## 🛠️ Installation

This project uses **uv** for fast dependency management. Follow these three steps to get started.

### 1\. Install uv

Refer to the **[uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)**

* **Recommended Install (macOS/Linux):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2\. Clone and Sync

Navigate to your cloned repository and run `uv sync`. This command **automatically creates the virtual environment (`.venv`)

```bash
uv sync
```

## 💻 Usage

The solver is implemented in the class `KKTSolver`. You must define your objective and constraint functions using **SymPy Expressions**.

### Running the Solver

The primary method for solving the problem is `KKTSolver().solve()`.

```python
from sympy import Symbol, cos
from src.kkt_solver import KKTSolver

# 1. Define the Symbols (Variables)
x = Symbol('x')
y = Symbol('y')
f_symbols = [x, y]

# 2. Define the Objective Function (f(v))
# Example: Minimize f(x) = x^2 + y^2
f = x**2 + y**2

# 3. Define the Constraints (g(v) <= 0 and h(v) = 0)
g_1 = x1 + x2 - 1

# cos will require allow_numeric
h_1 = x - cos(y)

# 4. Initialize and Solve
solver = KKTSolver(
    f=f,
    f_symbols=f_symbols,
    constraint_inequalities=[g_1],
    constraint_equalities=[h_1],
    allow_numeric=True,
    verbose=True
)

optimal_solutions = solver.solve(minimize=True)
# ... Output processing ...
```

### Output

The `solve()` method returns a list of `KKTSolution` objects, which are the points satisfying all KKT conditions and resulting in the minimum objective value found. Each `KKTSolution` object contains:

* **`vars`**: Dictionary of the optimal variable values ($\mathbf{x}^*$).
* **`lambdas`**: Dictionary of Lagrange Multipliers for inequalities ($\boldsymbol{\lambda}^*$).
* **`multipliers`**: Dictionary of Lagrange Multipliers for equalities ($\boldsymbol{\mu}^*$).
* **`value`**: The optimal objective function value, $f(\mathbf{x}^*)$.

-----

## 🧠 Solver Logic & KKT Conditions

The solver works by transforming the constrained optimization problem into a system of non-linear algebraic equations based on the KKT necessary conditions.

### 1. KKT Equation Formulation

The solver automatically constructs a system of equations $\large M = \mathbf{0}$ using the following necessary conditions:

| Condition | Mathematical Formulation |
| :--- | :--- |
| **Stationarity** | $$\large \nabla f(\mathbf{x}) + \sum_{i} \lambda_i \nabla g_i(\mathbf{x}) + \sum_{j} \mu_j \nabla h_j(\mathbf{x}) = \mathbf{0}$$ |
| **Complementary Slackness** | $$\large \lambda_i \cdot g_i(\mathbf{x}) = 0 \quad \text{for all inequalities}$$ |
| **Equality Feasibility** | $$\large h_j(\mathbf{x}) = 0 \quad \text{for all equalities}$$ |

-----

## 📜 License

This project is licensed under the **MIT License**. See the **`LICENSE`** file in the root directory for the full terms and conditions.

-----
