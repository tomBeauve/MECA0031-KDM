import os
import pandas as pd
from LHS import generate_optimal_LHS
import numpy as np
import matplotlib.pyplot as plt
from mechanismInit import Suspension
from initialConditions import IC
from mechanismInit import g


#### INTEGRATION PARAMETERS  #####

dt = 0.001
T = 10
tol_res = 1e-10
tol_g = 1e-10
gamma = 1/2 + 0.1
beta = 1/4 * (gamma + 1/2)**2 + 0.1


#### DoE PARAMETERS #####
bounds_k_log = [2.0, 7.0]
bounds_c_log = [2.0, 7.0]
n_samples = 500
n_iter_LHS = 100000

# Create directory if it doesn't exist
output_dir = "doe_results2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_and_save_doe(k_val, c_val, sim_id):
    # Update parameters
    Suspension.p["k"] = k_val
    Suspension.p["c"] = c_val
    filename = f"sim_k{int(k_val)}_c{int(c_val)}.csv"
    filepath = os.path.join(output_dir, filename)

    q, dq, ddq, lambdas = Suspension.Newmark(
        dt, T, IC, tol_res, tol_g, gamma, beta)

    if q is None:
        df_failed = pd.DataFrame({"status": ["FAILED"]})
        df_failed.to_csv(filepath, index=False)
        print(f"Sample k={k_val}, c={c_val}: FAILED")
        return

    time_steps = q.shape[1]
    time_array = np.linspace(0, T, time_steps)
    data = {"time": time_array, "status": "PASSED"}

    labels = ["xA", "yA", "xC", "yC", "xE", "yE", "xG", "yG", "xH",
              "yH", "phiH", "phiJ", "xL", "yL", "phiL", "phiN", "phiO"]

    for i, label in enumerate(labels):
        data[f"q_{label}"] = q[i, :]
        data[f"dq_{label}"] = dq[i, :]
        data[f"ddq_{label}"] = ddq[i, :]

    # Add Lagrange Multipliers and Constraint Violations
    for i in range(lambdas.shape[0]):
        data[f"lambda_{i}"] = lambdas[i, :]
        data[f"g_{i}"] = g(q, Suspension.p)[i]

    df_sim = pd.DataFrame(data)

    # 3. Export as CSV
    df_sim.to_csv(filepath, index=False)
    return


lhs_samples = generate_optimal_LHS(n_sample=n_samples, n_dim=2, bounds=[
                                   bounds_k_log, bounds_c_log], iterations=n_iter_LHS)


for i, (k, c) in enumerate(lhs_samples):
    run_and_save_doe(10**k, 10**c, i)
