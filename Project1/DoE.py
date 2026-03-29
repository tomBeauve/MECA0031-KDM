import os
import pandas as pd
from LHS import generate_optimal_LHS
import numpy as np
import matplotlib.pyplot as plt
from mechanismInit import Suspension
from initialConditions import IC
from mechanismInit import g
import time

"""
Performs a Design of Experiments (DoE) on a design space (k,c)
Samples are selected using a Latin Hypercube Sampling logic
Time integration is performed on each individual sample
Quantities of interest are stored inside csv files 
"""


#### INTEGRATION PARAMETERS  #####
dt = 0.001
T = 5
tol_res = 1e-10
tol_g = 1e-10
gamma = 1/2 + 0.1
beta = 1/4 * (gamma + 1/2)**2 + 0.1

# Special parameters for finding the steady state of the system
# Steady state is passed as initial condition
# This avoids unphysical motion at the beginning
dt_IC = 0.05
T_IC = 20
gamma_IC = 1/2 + 0.3
beta_IC = 1/4 * (gamma + 1/2)**2 + 0.1


#### DoE PARAMETERS #####
bounds_k_log = [3.9, 5.3]
bounds_c_log = [4.00, 4.28]
n_samples = 20
n_iter_LHS = 100000
output_dir = "doe_resultsZoom"


def run_and_save_doe(k_val, c_val):

    # Update parameters for the given sample
    Suspension.p["k"] = k_val
    Suspension.p["c"] = c_val
    filename = f"sim_k{int(k_val)}_c{int(c_val)}.csv"
    filepath = os.path.join(output_dir, filename)

    # Performs light time integration to find steady state of the given mechanism
    q_steady, _, _, _ = Suspension.Newmark(
        dt_IC, T_IC, IC, tol_res, tol_g, gamma_IC, beta_IC)
    dq_steady = np.zeros(len(IC))
    IC_specific = np.vstack([q_steady[:, -1], dq_steady]).T

    # Time integration
    q, dq, ddq, lambdas = Suspension.Newmark(
        dt, T, IC_specific, tol_res, tol_g, gamma, beta)

    # Special handling if the time integration failed
    if q is None:
        df_failed = pd.DataFrame({"status": ["FAILED"]})
        df_failed.to_csv(filepath, index=False)
        print(f"Sample k={k_val}, c={c_val}: FAILED")
        return

    # Storing results in a csv file
    time_steps = q.shape[1]
    time_array = np.linspace(0, T, time_steps)
    data = {"time": time_array, "status": "PASSED"}

    """
    # Saves all quantities of the system, heavier but useful if KPIs are not defined
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
    """

    # Lighter alternative : Saves only quantities of interest when KPIs known
    data = {
        "time": time_array,
        "status": "PASSED",
        "q_yH": q[9, :],
        "dq_yH": dq[9, :],
        "ddq_yH": ddq[9, :],
        "q_phiO": q[16, :],
        "dq_phiO": dq[16, :],
        "ddq_phiO": ddq[16, :]
    }

    df_sim = pd.DataFrame(data)

    # 3. Export as CSV
    df_sim.to_csv(filepath, index=False)
    return


# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Computation of a time-constrained optimal latin hypercube sampling
startLHS = time.time()
lhs_samples = generate_optimal_LHS(n_sample=n_samples, n_dim=2, bounds=[
                                   bounds_k_log, bounds_c_log], iterations=n_iter_LHS)
print(f"Optimal LHS generated in {time.time() - startLHS:.2f} seconds.")

startpar = time.time()
# Performs DoE using the samples
for _, (k, c) in enumerate(lhs_samples):
    run_and_save_doe(10**k, 10**c)
print(
    f"DoE finished in {time.time() - startpar:.2f} seconds for {n_samples} samples.")
