import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

class ParametricCurveFitter:
    def __init__(self):
        self.theta_bounds = (1e-6, np.deg2rad(50) - 1e-6)
        self.M_bounds = (-0.05 + 1e-6, 0.05 - 1e-6)
        self.X_bounds = (1e-6, 100 - 1e-6)
    
    def parametric_curve(self, t, theta, M, X):
        x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
        y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
        return x, y
    
    def compute_l1_distance(self, params, t_values, true_x, true_y):
        theta, M, X = params
        x_pred, y_pred = self.parametric_curve(t_values, theta, M, X)
        return np.sum(np.abs(x_pred - true_x) + np.abs(y_pred - true_y))
    
    def fit_curve(self, points_data):
        t_values = points_data[:, 0]
        true_x = points_data[:, 1]
        true_y = points_data[:, 2]
        
        bounds = [self.theta_bounds, self.M_bounds, self.X_bounds]
        
        result = differential_evolution(
            self.compute_l1_distance,
            bounds,
            args=(t_values, true_x, true_y),
            maxiter=1000,
            popsize=15,
            seed=42,
            polish=True
        )
        
        refined = minimize(
            self.compute_l1_distance,
            result.x,
            args=(t_values, true_x, true_y),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if refined.success and refined.fun < result.fun:
            best_params = refined.x
            best_distance = refined.fun
        else:
            best_params = result.x
            best_distance = result.fun
        
        theta_opt, M_opt, X_opt = best_params
        
        return {
            'theta_rad': theta_opt,
            'theta_deg': np.rad2deg(theta_opt),
            'M': M_opt,
            'X': X_opt,
            'l1_distance': best_distance
        }
    
    def generate_desmos_format(self, results):
        theta = results['theta_rad']
        M = results['M']
        X = results['X']
        
        return f"\\left(t*\\cos({theta:.6f})-e^{{{M:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta:.6f})+{X:.6f},42+t*\\sin({theta:.6f})+e^{{{M:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta:.6f})\\right)"

def main():
    df = pd.read_csv('xy_data.csv')
    t_values = np.linspace(6, 60, len(df))
    points_data = np.column_stack([t_values, df['x'].values, df['y'].values])
    
    fitter = ParametricCurveFitter()
    results = fitter.fit_curve(points_data)
    
    desmos_format = fitter.generate_desmos_format(results)
    print(desmos_format)
    
    with open('README.md', 'w') as f:
        f.write("# FLAM Research and Development - Parametric Curve Fitting\n\n")
        f.write("## Submission Format\n\n")
        f.write(f"{desmos_format}\n\n")
        f.write("## Parameters Found\n\n")
        f.write(f"- θ = {results['theta_deg']:.6f}° ({results['theta_rad']:.6f} rad)\n")
        f.write(f"- M = {results['M']:.6f}\n")
        f.write(f"- X = {results['X']:.6f}\n\n")
        f.write("## Mathematical Approach\n\n")
        f.write("Used Differential Evolution global optimization followed by L-BFGS-B local refinement to minimize L1 distance between predicted and actual points.\n\n")
        f.write("**Objective Function:** L1 = Σ|x_pred(t_i) - x_given(t_i)| + |y_pred(t_i) - y_given(t_i)|\n\n")
        f.write("**Constraints satisfied:**\n")
        f.write("- 0° < θ < 50°\n")
        f.write("- -0.05 < M < 0.05\n")
        f.write("- 0 < X < 100\n")

if __name__ == "__main__":
    main()
