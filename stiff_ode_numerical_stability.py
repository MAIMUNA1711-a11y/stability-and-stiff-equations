"""
Stiff ODEs & Numerical Stability
=================================
Python implementation of numerical methods for solving stiff ODEs.
Covers: Forward Euler, Backward Euler, RK2, RK4, and scipy's solve_ivp.

Problem: dy/dt = λy,  y(0) = 1,  exact solution: y(t) = e^(λt)
For stiff problems, λ is a large negative number (e.g., λ = -1000).

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────
# 1. ODE Definition (Test Equation)
# ─────────────────────────────────────────────
def f(t, y, lam):
    """Right-hand side of dy/dt = λy"""
    return lam * y


def exact_solution(t, lam, y0=1.0):
    """Exact solution: y(t) = y0 * e^(λt)"""
    return y0 * np.exp(lam * t)


# ─────────────────────────────────────────────
# 2. Forward Euler Method (Explicit)
# ─────────────────────────────────────────────
def forward_euler(lam, y0, t0, t_end, h):
    """
    Forward (Explicit) Euler Method.
    Formula: y_{n+1} = y_n + h * f(t_n, y_n)
    Stability condition: |1 + hλ| < 1  →  requires h < 2/|λ|
    """
    t_vals = [t0]
    y_vals = [y0]

    t, y = t0, y0
    while t < t_end - 1e-12:
        h_actual = min(h, t_end - t)
        y = y + h_actual * f(t, y, lam)
        t = t + h_actual
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


# ─────────────────────────────────────────────
# 3. Backward Euler Method (Implicit)
# ─────────────────────────────────────────────
def backward_euler(lam, y0, t0, t_end, h):
    """
    Backward (Implicit) Euler Method.
    Formula: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    For linear ODE: y_{n+1} = y_n / (1 - hλ)
    Unconditionally stable for λ < 0 (A-stable).
    """
    t_vals = [t0]
    y_vals = [y0]

    t, y = t0, y0
    while t < t_end - 1e-12:
        h_actual = min(h, t_end - t)
        # Analytical solution for linear case
        y = y / (1 - h_actual * lam)
        t = t + h_actual
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


def backward_euler_nonlinear(f_func, y0, t0, t_end, h):
    """
    Backward Euler for nonlinear ODE (uses Newton/fsolve).
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    """
    t_vals = [t0]
    y_vals = [y0]

    t, y = t0, y0
    while t < t_end - 1e-12:
        h_actual = min(h, t_end - t)
        t_next = t + h_actual
        # Solve: g(y_next) = y_next - y - h*f(t_next, y_next) = 0
        g = lambda y_next: y_next - y - h_actual * f_func(t_next, y_next)
        y_next = fsolve(g, y)[0]
        y = y_next
        t = t_next
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


# ─────────────────────────────────────────────
# 4. Runge-Kutta 2nd Order (RK2 / Midpoint)
# ─────────────────────────────────────────────
def rk2(lam, y0, t0, t_end, h):
    """
    2nd-order Runge-Kutta (Midpoint method).
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    y_{n+1} = y_n + h * k2
    Stability region: larger than FE but not unconditionally stable.
    """
    t_vals = [t0]
    y_vals = [y0]

    t, y = t0, y0
    while t < t_end - 1e-12:
        h_actual = min(h, t_end - t)
        k1 = f(t, y, lam)
        k2 = f(t + 0.5 * h_actual, y + 0.5 * h_actual * k1, lam)
        y = y + h_actual * k2
        t = t + h_actual
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


# ─────────────────────────────────────────────
# 5. Runge-Kutta 4th Order (RK4)
# ─────────────────────────────────────────────
def rk4(lam, y0, t0, t_end, h):
    """
    Classic 4th-order Runge-Kutta Method.
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    k3 = f(t_n + h/2, y_n + h/2 * k2)
    k4 = f(t_n + h,   y_n + h * k3)
    y_{n+1} = y_n + h/6 * (k1 + 2k2 + 2k3 + k4)
    Stability condition: |hλ| < 2.785 (for real λ)
    """
    t_vals = [t0]
    y_vals = [y0]

    t, y = t0, y0
    while t < t_end - 1e-12:
        h_actual = min(h, t_end - t)
        k1 = f(t,              y,                      lam)
        k2 = f(t + 0.5*h_actual, y + 0.5*h_actual*k1, lam)
        k3 = f(t + 0.5*h_actual, y + 0.5*h_actual*k2, lam)
        k4 = f(t + h_actual,     y + h_actual*k3,      lam)
        y = y + (h_actual / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h_actual
        t_vals.append(t)
        y_vals.append(y)

    return np.array(t_vals), np.array(y_vals)


# ─────────────────────────────────────────────
# 6. Amplification Factor Analysis
# ─────────────────────────────────────────────
def amplification_factor(method, h_lam_values):
    """
    Compute |R(hλ)| for each method — determines stability.
    |R(hλ)| < 1  →  stable
    |R(hλ)| > 1  →  unstable (solution grows unboundedly)
    """
    factors = {}
    for z in h_lam_values:   # z = hλ (complex in general, real here)
        factors['Forward Euler'] = abs(1 + z)
        factors['Backward Euler'] = abs(1 / (1 - z))
        factors['RK2'] = abs(1 + z + 0.5 * z**2)
        factors['RK4'] = abs(1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4)
    return factors


def stability_boundary():
    """
    Compute stability regions on the complex h*λ plane.
    Returns h*λ values where |R(hλ)| = 1 for each method.
    """
    theta = np.linspace(0, 2 * np.pi, 500)
    results = {}

    # For each method, find boundary where |R(z)| = 1
    # (approximated by evaluating on a grid)
    re_vals = np.linspace(-6, 2, 400)
    im_vals = np.linspace(-4, 4, 400)
    RE, IM = np.meshgrid(re_vals, im_vals)
    Z = RE + 1j * IM

    methods = {
        'Forward Euler':  lambda z: np.abs(1 + z),
        'Backward Euler': lambda z: np.abs(1 / (1 - z)),
        'RK2':            lambda z: np.abs(1 + z + 0.5 * z**2),
        'RK4':            lambda z: np.abs(1 + z + z**2/2 + z**3/6 + z**4/24),
    }

    for name, R in methods.items():
        results[name] = R(Z)

    return RE, IM, results


# ─────────────────────────────────────────────
# 7. Robertson's Problem (Classic Stiff System)
# ─────────────────────────────────────────────
def robertson_rhs(t, y):
    """
    Robertson's chemical kinetics problem — a famous stiff ODE system.
    dy1/dt = -0.04*y1 + 1e4*y2*y3
    dy2/dt =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    dy3/dt =  3e7*y2^2
    Initial conditions: y1(0)=1, y2(0)=0, y3(0)=0
    Stiffness ratio: eigenvalues span ~1 to ~1e11
    """
    y1, y2, y3 = y
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 =  0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    dy3 =  3e7 * y2**2
    return [dy1, dy2, dy3]


def solve_robertson():
    """Solve Robertson's problem using scipy's stiff solver (Radau)."""
    y0 = [1.0, 0.0, 0.0]
    t_span = (0, 1e11)
    t_eval = np.logspace(-5, 11, 500)

    sol = solve_ivp(
        robertson_rhs,
        t_span,
        y0,
        method='Radau',   # Implicit Runge-Kutta (good for stiff problems)
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        dense_output=True
    )
    return sol


# ─────────────────────────────────────────────
# 8. Main Demonstration & Comparison
# ─────────────────────────────────────────────
def run_stiff_demo():
    """
    Demonstrate stiffness: compare methods on dy/dt = λy with λ = -1000.
    Shows that explicit methods (FE, RK4) require tiny step sizes,
    while Backward Euler is unconditionally stable with any step size.
    """
    lam   = -1000.0     # Large negative → stiff problem
    y0    = 1.0
    t0    = 0.0
    t_end = 0.05

    # Critical step size for stability:
    # FE stable when h < 2/|λ| = 0.002
    h_stable   = 0.001   # Just within FE stability limit
    h_unstable = 0.003   # Outside FE stability limit (will blow up)
    h_be       = 0.01    # Much larger — BE stays stable regardless

    print("=" * 60)
    print("STIFF ODE DEMO: dy/dt = λy,  λ = -1000,  y(0) = 1")
    print("=" * 60)
    print(f"Exact critical step for FE stability: h < {2/abs(lam):.4f}")
    print()

    # Forward Euler — stable case
    t_fe_s, y_fe_s = forward_euler(lam, y0, t0, t_end, h_stable)
    print(f"Forward Euler  (h={h_stable}): Final y = {y_fe_s[-1]:.6e}")

    # Forward Euler — unstable case
    t_fe_u, y_fe_u = forward_euler(lam, y0, t0, t_end, h_unstable)
    print(f"Forward Euler  (h={h_unstable}): Final y = {y_fe_u[-1]:.6e}  ← UNSTABLE")

    # Backward Euler — large step, stays stable
    t_be, y_be = backward_euler(lam, y0, t0, t_end, h_be)
    print(f"Backward Euler (h={h_be}):  Final y = {y_be[-1]:.6e}  ← A-STABLE")

    # RK4 — also needs small steps for stiff problems
    t_rk4, y_rk4 = rk4(lam, y0, t0, t_end, h_stable)
    print(f"RK4            (h={h_stable}): Final y = {y_rk4[-1]:.6e}")

    # Exact solution at final time
    y_exact = exact_solution(t_end, lam, y0)
    print(f"Exact solution              : Final y = {y_exact:.6e}")

    return {
        'fe_stable':   (t_fe_s, y_fe_s),
        'fe_unstable': (t_fe_u, y_fe_u),
        'be':          (t_be, y_be),
        'rk4':         (t_rk4, y_rk4),
        'lam':          lam,
        't_end':        t_end,
    }


def run_method_comparison():
    """
    Compare all four methods on a mildly stiff problem (λ = -10).
    Shows accuracy differences between methods.
    """
    lam   = -10.0
    y0    = 1.0
    t0    = 0.0
    t_end = 1.0
    h     = 0.1

    t_fe,  y_fe  = forward_euler(lam, y0, t0, t_end, h)
    t_be,  y_be  = backward_euler(lam, y0, t0, t_end, h)
    t_rk2, y_rk2 = rk2(lam, y0, t0, t_end, h)
    t_rk4, y_rk4 = rk4(lam, y0, t0, t_end, h)

    # Error at final time
    y_exact_end = exact_solution(t_end, lam, y0)
    print("\n" + "=" * 60)
    print(f"METHOD COMPARISON: λ={lam}, h={h}, t∈[0,{t_end}]")
    print("=" * 60)
    for name, y_vals in [('Forward Euler', y_fe), ('Backward Euler', y_be),
                          ('RK2', y_rk2), ('RK4', y_rk4)]:
        err = abs(y_vals[-1] - y_exact_end)
        print(f"{name:<20} Final y = {y_vals[-1]:.8f}  Error = {err:.2e}")
    print(f"{'Exact':<20} Final y = {y_exact_end:.8f}")

    return {
        'fe': (t_fe, y_fe), 'be': (t_be, y_be),
        'rk2': (t_rk2, y_rk2), 'rk4': (t_rk4, y_rk4),
        'lam': lam, 't_end': t_end, 'h': h
    }


# ─────────────────────────────────────────────
# 9. Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Demo 1: Stiffness (large |λ|) ──
    stiff_results = run_stiff_demo()

    # ── Demo 2: Method accuracy comparison ──
    cmp_results = run_method_comparison()

    # ── Demo 3: Robertson's stiff system ──
    print("\n" + "=" * 60)
    print("ROBERTSON'S PROBLEM (scipy Radau solver)")
    print("=" * 60)
    sol = solve_robertson()
    print(f"Solved from t=1e-5 to t=1e11 in {sol.t.shape[0]} steps")
    print(f"y1(final) = {sol.y[0,-1]:.4f}, y2(final) = {sol.y[1,-1]:.4e}, y3(final) = {sol.y[2,-1]:.4f}")
    print(f"Conservation check |y1+y2+y3 - 1| = {abs(sol.y[:,0].sum() - 1):.2e}  (should be ~0)")

    # ── Demo 4: Amplification factor check ──
    print("\n" + "=" * 60)
    print("AMPLIFICATION FACTORS at hλ = -1.5")
    print("=" * 60)
    z = -1.5
    print(f"Forward Euler  |R| = {abs(1 + z):.4f}  {'STABLE' if abs(1+z)<1 else 'UNSTABLE'}")
    print(f"Backward Euler |R| = {abs(1/(1-z)):.4f}  {'STABLE' if abs(1/(1-z))<1 else 'UNSTABLE'}")
    print(f"RK4            |R| = {abs(1+z+z**2/2+z**3/6+z**4/24):.4f}  "
          f"{'STABLE' if abs(1+z+z**2/2+z**3/6+z**4/24)<1 else 'UNSTABLE'}")
