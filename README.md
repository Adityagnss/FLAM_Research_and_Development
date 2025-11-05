# FLAM Research and Development - Parametric Curve Fitting

## Submission Format

\left(t*\cos(0.490746)-e^{0.021385\left|t\right|}\cdot\sin(0.3t)\sin(0.490746)+54.899216,42+t*\sin(0.490746)+e^{0.021385\left|t\right|}\cdot\sin(0.3t)\cos(0.490746)\right)

## Parameters Found

- θ = 28.117696° (0.490746 rad)
- M = 0.021385
- X = 54.899216

## Mathematical Approach

Used Differential Evolution global optimization followed by L-BFGS-B local refinement to minimize L1 distance between predicted and actual points.

**Objective Function:** L1 = Σ|x_pred(t_i) - x_given(t_i)| + |y_pred(t_i) - y_given(t_i)|

**Constraints satisfied:**
- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100
