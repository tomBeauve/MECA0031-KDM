from pathlib import Path
import re
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def T_steadyState(dq, timeArray, threshold=0.02):
    """
    Computes settling time based on the velocity array.
    """
    max_vel = np.max(np.abs(dq))
    if max_vel == 0:
        return 0.0

    limit = threshold * max_vel

    outside_idx = np.where(np.abs(dq) > limit)[0]

    if len(outside_idx) == 0:
        return 0.0

    last_idx = outside_idx[-1]

    if last_idx >= len(timeArray) - 1:
        print("Warning: System did not settle within simulation time.")
        return timeArray[-1]

    return timeArray[last_idx]


#################################################
#                                               #
#       POST PROCESSING FROM DATA FROM DoE      #
#                                               #
#################################################

# Path to your folder
folder_path = "doe_results2"
# Collect all paths matching the pattern
# This creates a list of strings: ['doe_results/sim_k100_c50.csv', ...]
file_paths = glob.glob(os.path.join(folder_path, "sim_k*_c*.csv"))


summary_list = []


for path in file_paths:
    p = Path(path)
    # p.stem is the filename without extension: 'sim_k1234_c567'

    # Split into a list using '_' as the anchor
    parts = p.stem.split('_')

    # Assign based on index (assuming [0]=sim, [1]=k..., [2]=c...)
    k = float(parts[1][1:])  # Take part 1, skip first character 'k'
    c = float(parts[2][1:])  # Take part 2, skip first character 'c'

    df = pd.read_csv(path)

    if df["status"].to_numpy()[0] == "FAILED":
        continue  # This now correctly skips to the NEXT file

    accel_A = df["ddq_yA"].to_numpy()
    displ_A = df["q_yA"].to_numpy()
    time = df["time"].to_numpy()
    phiO = df["q_phiO"].to_numpy()

    peak_accel = np.max(np.abs(accel_A))
    settling_time = T_steadyState(accel_A, time)
    # maximum displacements compared to initial condition assumed to be steady state config.
    peak_displ = np.max(np.abs(displ_A - displ_A[0]))
    peak_angle = np.max(np.abs(phiO - phiO[0]))

    # Store metadata and results in a dictionary
    summary_list.append({
        "filename": os.path.basename(path),
        "k": k,
        "c": c,
        "max_accel": peak_accel,
        "settling_time": settling_time,
        "max_displ": peak_displ,
        "max_angle": peak_angle
    })

# Create the Master Summary
df_master = pd.DataFrame(summary_list)
df_master.to_csv(os.path.join(
    folder_path, "doe_summary_results.csv"), index=False)


plt.scatter(df_master["k"], df_master["c"])
plt.xlabel("Spring stiffness k")
plt.ylabel("Damping viscosity c")
plt.xscale("log")
plt.yscale("log")
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sc1 = ax1.scatter(df_master["max_accel"], df_master["settling_time"],
                  c=np.log10(df_master["k"]), cmap='plasma', alpha=0.7)
fig.colorbar(sc1, ax=ax1, label="Stiffness log10(k)")
ax1.set_title("Sensitivity to Stiffness (k)")


sc2 = ax2.scatter(df_master["max_accel"], df_master["settling_time"],
                  c=np.log10(df_master["c"]), cmap='plasma', alpha=0.7)
fig.colorbar(sc2, ax=ax2, label="Damping log10(c)")
ax2.set_title("Sensitivity to Damping (c)")

# Apply formatting to both axes
for ax in [ax1, ax2]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Max Acceleration (Chassis Comfort)")
    ax.set_ylabel("Settling Time (Stability)")
    ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(
    folder_path, "paretoFront.png"), dpi=300)
plt.show()


##################################################################
# Load the summary data
df_summ = pd.read_csv(os.path.join(
    folder_path, "doe_summary_results.csv"))

idx_min_accel = df_summ["max_accel"].idxmin()

best_comfort_row = df_summ.loc[idx_min_accel]

min_accel_val = best_comfort_row["max_accel"]
best_k = best_comfort_row["k"]
best_c = best_comfort_row["c"]
best_settling = best_comfort_row["settling_time"]

print(f"--- Absolute Minimum Acceleration (Max Comfort) ---")
print(f"Min Max Accel: {min_accel_val:.4e} m/s^2")
print(f"Optimal Stiffness (k): {best_k:.2f} N/m")
print(f"Optimal Damping (c): {best_c:.2f} Ns/m")
print(f"Resulting Settling Time: {best_settling:.2f} s")


##################################################################
idx_min_t = df_summ["settling_time"].idxmin()

best_sport_row = df_summ.loc[idx_min_t]

min_accel_val = best_sport_row["max_accel"]
best_k = best_sport_row["k"]
best_c = best_sport_row["c"]
best_settling = best_sport_row["settling_time"]

print(f"--- Absolute Minimum Settling time  ---")
print(f"Optimal Stiffness (k): {best_k:.2f} N/m")
print(f"Optimal Damping (c): {best_c:.2f} Ns/m")
print(f"Min Settling Time: {best_settling:.2f} s")
print(f" Max Accel: {min_accel_val:.4e} m/s^2")
