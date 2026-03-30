from pathlib import Path
import re
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def T_steadyState(dq, timeArray, threshold=0.02, skip_steps=0):
    """
    Computes settling time based on the velocity array.
    Proceeds by taking the last index where velocity > treshold * max velocity
    Skips "skip_steps" initial steps to avoid undesired results
    """
    dq_active = dq[skip_steps:]
    time_active = timeArray[skip_steps:]

    dq_abs = np.abs(dq_active)
    max_vel = np.max(dq_abs)

    if max_vel < 1e-12:
        return 0.0

    limit = threshold * max_vel

    outside_idx = np.where(dq_abs > limit)[0]

    if len(outside_idx) == 0:
        return timeArray[skip_steps]

    last_idx = outside_idx[-1]

    if last_idx >= len(time_active) - 1:
        return timeArray[-1]

    return time_active[last_idx]


#################################################
#                                               #
#           DoE results unpacking               #
#                                               #
#################################################

# Path to folders results
folder_path = "doe_results"
T_sslim = 2.5

file_paths = glob.glob(os.path.join(folder_path, "sim_k*_c*.csv"))

folder_path2 = "doe_resultsStrict"
file_paths2 = glob.glob(os.path.join(folder_path2, "sim_k*_c*.csv"))
file_paths.extend(file_paths2)
summary_list = []

if __name__ == "__main__":

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
            continue

        accel_H = df["ddq_yH"].to_numpy()
        vel_H = df["dq_yH"].to_numpy()
        displ_H = df["q_yH"].to_numpy()
        time = df["time"].to_numpy()
        sec5 = np.where(time >= 5)[0][0]
        accel_H = accel_H[:sec5]
        vel_H = vel_H[:sec5]
        displ_H = displ_H[:sec5]
        time = time[:sec5]

        # Extract 2 KPIs :
        # min accel of yh => min grip due to the road irregularity, to max = min -(min_grip)
        # var of yH => ride height variation, to min for better aero stability
        min_grip = -np.min(accel_H)
        rideHeightVar = np.std(displ_H - np.mean(displ_H))
        # peak_RH_change = np.max(np.abs(displ_H - displ_H[0]))

        summary_list.append({
            "filename": os.path.basename(path),
            "k": k,
            "c": c,
            "min_grip": min_grip,
            "rideHeightVar": rideHeightVar,
            "settling_time": T_steadyState(vel_H, time)
        })

    df_master = pd.DataFrame(summary_list)
    # Filtering of DoE to remove results out of design space
    m_sprung = 200
    damping_ratio = df_master["c"] / (2 * np.sqrt(df_master["k"] * m_sprung))
    df_master = df_master[damping_ratio.between(0.1, 1.5)].copy()

    df_master.to_csv(os.path.join(
        folder_path, "doe_summary_results.csv"), index=False)


#################################################
#                                               #
#       Print the DoE within design space       #
#                                               #
#################################################

    # Plots the DoE samples in the design space
    plt.scatter(df_master["k"], df_master["c"])
    plt.xlabel("Spring stiffness k")
    plt.ylabel("Damping viscosity c")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

#################################################
#                                               #
#    Identification of best design and print    #
#                                               #
#################################################

###########################
# Best min grip, filtered by settling time required <= T_ssLim
###########################

    df_summ = pd.read_csv(os.path.join(
        folder_path, "doe_summary_results.csv"))
    df_summ = df_summ[df_summ["settling_time"] < T_sslim].copy()

    idx_best_min_grip = df_summ["min_grip"].idxmin()

    best_min_grip = df_summ.loc[idx_best_min_grip]

    min_grip_val = best_min_grip["min_grip"]
    k_val = best_min_grip["k"]
    c_val = best_min_grip["c"]
    RHVar_val = best_min_grip["rideHeightVar"]
    settling_val = best_min_grip["settling_time"]

    print("#########################################################################")
    print(f"--- Best (max) Min grip ---")
    print(f"Min grip: {min_grip_val:.4e} m/s^2")
    print(f"Optimal Stiffness (k): {k_val:.2f} N/m")
    print(f"Optimal Damping (c): {c_val:.2f} Ns/m")
    print(f"Resulting Ride Height Var: {RHVar_val:.7f} m^2")
    print(f"Resulting Settling Time: {settling_val:.2f} s")

###########################
# Best ride height var, filtered by settling time required <= T_ssLim
###########################
    idx_best_RHVar = df_summ["rideHeightVar"].idxmin()

    best_RHVar = df_summ.loc[idx_best_RHVar]

    min_grip_val = best_RHVar["min_grip"]
    k_val = best_RHVar["k"]
    c_val = best_RHVar["c"]
    RHVar_val = best_RHVar["rideHeightVar"]
    settling_val = best_RHVar["settling_time"]

    print("#########################################################################")
    print(f"--- Best (min) Ride Height Variance ---")
    print(f"Ride Height Var: {RHVar_val:.7f} m^2")
    print(f"Optimal Stiffness (k): {k_val:.2f} N/m")
    print(f"Optimal Damping (c): {c_val:.2f} Ns/m")
    print(f"Resulting Min grip: {min_grip_val:.4e} m/s^2")
    print(f"Resulting Settling Time: {settling_val:.2f} s")

###########################
# Normalizes the KPIs between 0 and 1, 0 is best, 1 is worst
# Then computes performance score as weighed sum of the normalized KPI
# Best overall score still filtered by settling time  required <= T_ssLim
############################
    df_summ = pd.read_csv(os.path.join(
        folder_path, "doe_summary_results.csv"))
    df_summ = df_summ[df_summ["settling_time"] < T_sslim].copy()

    best_RHVar = df_summ.loc[idx_best_RHVar]
    worst_RHVar = df_summ.loc[df_summ["rideHeightVar"].idxmax()]
    best_min_grip = df_summ.loc[idx_best_min_grip]
    worst_min_grip = df_summ.loc[df_summ["min_grip"].idxmax()]

    rhVarNormalized = (np.log10(df_summ["rideHeightVar"]) - np.log10(best_RHVar["rideHeightVar"])) / \
        (np.log10(worst_RHVar["rideHeightVar"]) -
         np.log10(best_RHVar["rideHeightVar"]))

    minGripNormalized = (df_summ["min_grip"] - best_min_grip["min_grip"]) / \
        (worst_min_grip["min_grip"] - best_min_grip["min_grip"])

    df_summ["performance_score"] = 0.5 * \
        rhVarNormalized + 50 * minGripNormalized
    idx_best_score = df_summ["performance_score"].idxmin()
    best_score = df_summ.loc[idx_best_score]
    print("#########################################################################")
    print(f"--- Best overall score ---")
    print(f"Performance score: {best_score['performance_score']:.4f}")
    print(f"Optimal Stiffness (k): {best_score['k']:.2f} N/m")
    print(f"Optimal Damping (c): {best_score['c']:.2f} Ns/m")
    print(f"Resulting Min grip: {best_score['min_grip']:.4e} m/s^2")
    print(f"Resulting Ride Height Var: {best_score['rideHeightVar']:.7f} m^2")
    print(f"Resulting Settling Time: {best_score['settling_time']:.2f} s")

#################################################
#                                               #
#           Pareto Plot of the 2 KPIs           #
#                                               #
#################################################

    df_master = df_master[df_master["settling_time"] < T_sslim].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    sc1 = ax1.scatter(df_master["min_grip"], df_master["rideHeightVar"],
                      c=np.log10(df_master["k"]), cmap='plasma', alpha=0.7)
    fig.colorbar(sc1, ax=ax1, label="Stiffness log10(k)")
    ax1.set_title("Sensitivity to Stiffness (k)")

    sc2 = ax2.scatter(df_master["min_grip"], df_master["rideHeightVar"],
                      c=np.log10(df_master["c"]), cmap='plasma', alpha=0.7)
    fig.colorbar(sc2, ax=ax2, label="Damping log10(c)")
    ax2.set_title("Sensitivity to Damping (c)")

    sc3 = ax3.scatter(df_master["min_grip"], df_master["rideHeightVar"],
                      c=df_master["settling_time"], cmap='plasma', alpha=0.7)
    fig.colorbar(sc3, ax=ax3, label="Settling time (s)")
    ax3.set_title("Impact of Settling Time")

    for ax in [ax1, ax2, ax3]:
        ax.scatter(best_score["min_grip"], best_score["rideHeightVar"],
                   color='red', marker='*', s=200, label='Best Overall')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Min Grip (Tire Contact)")
        if ax == ax1:
            ax.set_ylabel("Ride Height Var (Aero Stability)")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    # Save the unified Pareto figure
    plt.savefig(os.path.join(
        folder_path, "pareto_comparison_3way.png"), dpi=300)
    plt.show()


#################################################
#                                               #
#        DoE plot colored by min grip           #
#                                               #
#################################################

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plots the DoE samples in the design space
    sc = ax.scatter(df_master["k"], df_master["c"],
                    c=np.log10(df_master["min_grip"]), cmap='plasma', alpha=0.7)
    fig.colorbar(sc, ax=ax, label="Min grip log10()")

    ax.scatter(best_score["k"], best_score["c"],
               color='red', marker='*', s=100, label='Best Overall')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("c")
    plt.show()
