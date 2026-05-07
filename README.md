# 🧮 Stiff ODEs & Numerical Stability

> An interactive visualization and Python implementation of numerical methods for solving stiff ordinary differential equations.

---

## 📌 Overview

This project explores **why some differential equations break numerical solvers** — and how stability theory governs the choice between explicit and implicit methods.

All stability analysis is built on the **test equation**:

```
dy/dt = λy,    y(0) = 1,    exact solution: y(t) = eˡᵗ
```

When `Re(λ) << 0`, the equation becomes **stiff** — forcing tiny step sizes on explicit methods even when the solution changes slowly.

---

## 📂 Project Structure

```
📦 stiff-ode-numerical-stability
 ┣ 📄 index.html                        # Interactive visualization (open in browser)
 ┣ 📄 stiff_ode_numerical_stability.py  # Python implementation
 ┗ 📄 README.md
```

---

## 🌐 Interactive Visualization

Open `index.html` in any browser — no installation needed.

| Section | Topic |
|---------|-------|
| ① | What is Stiffness? |
| ② | Amplification Factor |
| ③ | Stability Regions |
| ④ | Forward Euler — Instability |
| ⑤ | Backward Euler — A-Stability |
| ⑥ | RK2 vs RK4 |
| ⑦ | All Methods Comparison |
| ⑧ | Practice Example |
| ⑨ | Limitations & Fix |

---

## 🐍 Python Implementation

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run

```bash
python stiff_ode_numerical_stability.py
```

### What's Implemented

| Function | Description |
|----------|-------------|
| `forward_euler()` | Explicit Euler — stable only when `h < 2/\|λ\|` |
| `backward_euler()` | Implicit Euler — unconditionally A-stable |
| `rk2()` | 2nd-order Runge-Kutta (Midpoint method) |
| `rk4()` | Classic 4th-order Runge-Kutta |
| `robertson_rhs()` | Robertson's stiff chemical kinetics system |
| `solve_robertson()` | Solved via SciPy's Radau implicit solver |
| `run_stiff_demo()` | Demonstrates instability at λ = −1000 |
| `run_method_comparison()` | Accuracy & error comparison across all methods |

---

## 📊 Key Concepts

### Stiffness
Occurs when eigenvalues of a system differ by **orders of magnitude**.
The fast-decaying component forces tiny step sizes — not for accuracy, but purely for stability.

```
Stiffness Ratio  S = |λ_max / λ_min|
```

### Amplification Factor
Each method has a stability function `R(hλ)`. The method is stable only when:
```
|R(hλ)| < 1
```

| Method | R(hλ) | Stability Condition |
|--------|--------|---------------------|
| Forward Euler | `1 + hλ` | `\|hλ\| < 2` |
| Backward Euler | `1 / (1 − hλ)` | Always stable ✅ |
| RK4 | `1 + z + z²/2 + z³/6 + z⁴/24` | `\|hλ\| < 2.79` |

### A-Stability
A method is **A-stable** if its stability region contains the entire left half of the complex plane.
→ Backward Euler is A-stable. Forward Euler and RK4 are **not**.

---

## 🧪 Sample Output

```
============================================================
STIFF ODE DEMO: dy/dt = λy,  λ = -1000,  y(0) = 1
============================================================
Exact critical step for FE stability: h < 0.0020

Forward Euler  (h=0.001): Final y = 0.000000e+00
Forward Euler  (h=0.003): Final y = -6.553600e+04  ← UNSTABLE 💥
Backward Euler (h=0.01):  Final y = 6.209213e-06   ← A-STABLE ✅
RK4            (h=0.001): Final y = 5.029947e-22
Exact solution           : Final y = 1.928750e-22

============================================================
AMPLIFICATION FACTORS at hλ = -1.5
============================================================
Forward Euler  |R| = 0.5000  STABLE
Backward Euler |R| = 0.4000  STABLE
RK4            |R| = 0.2734  STABLE
```

---

## 📚 Topic

**Course:** Numerical Methods
**Topic:** Stiff ODEs & Numerical Stability
**Lecture:** 39

---

## 👩‍💻 Author

**Maimuna Akter**
Student ID: 22CSE021
