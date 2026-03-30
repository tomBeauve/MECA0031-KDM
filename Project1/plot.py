import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cycler

###########################################
#                                         #
#     File to make report-ready plots     #
#                                         #
###########################################
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,

    "axes.grid": True,
    "grid.color": "0.7",
    "grid.linewidth": 0.8,
    "grid.linestyle": ":",
    "grid.alpha": 0.7,

    "lines.linewidth": 2.5,
    "lines.markersize": 5.0,
    "lines.markeredgewidth": 1.0,
    "image.cmap": "viridis",

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 7.0,
    "ytick.major.size": 7.0,
    "xtick.direction": "in",
    "ytick.direction": "in",

    "axes.prop_cycle": plt.cycler('color', plt.cm.Set1.colors),
    "figure.autolayout": True,
    "figure.figsize": (8, 6),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


SINGLE_PLOT = False
k = 50000.0
c = 0
filepath = f"main_results/k{k}_c{c}_force.csv"

COMPARATIVE_PLOT = True
CONFIG = [[0.0, 1000], [50000.0, 1000]]


if SINGLE_PLOT:
    ################## Single Plot ######################

    df = pd.read_csv(filepath)

    acc_yH = df["ddq_yH"].to_numpy()
    vel_yH = df["dq_yH"].to_numpy()
    displ_yH = df["q_yH"].to_numpy()
    time = df["time"].to_numpy()

    accel_phiO = df["ddq_phiO"].to_numpy()
    vel_phiO = df["dq_phiO"].to_numpy()
    displ_phiO = df["q_phiO"].to_numpy()

    f_ext = 100 * np.exp(-(time-0.5)**2)

    #############################################
    # Displacement of wheel bottom
    plt.plot(time, displ_yH)
    plt.xlabel("Time (s)")
    plt.ylabel("Vertical Position (m)")
    plt.tight_layout()
    plt.savefig(f"main_plots/wheelDispl_k{k}_c{c}_force.pdf", dpi=300)
    plt.show()
    print(f"k = {k}, c = {c}")

    #############################################
    # Rocker angle displacement
    plt.plot(time, np.rad2deg(displ_phiO))
    plt.xlabel("Time (s)")
    plt.ylabel("Rocker Angle (°)")
    plt.tight_layout()
    plt.savefig(f"main_plots/AngleDispl_k{k}_c{c}_force.pdf", dpi=300)
    plt.show()

    # Rocker angle angular velocity
    plt.plot(time, np.rad2deg(vel_phiO))
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (°/s)")
    plt.tight_layout()
    plt.savefig(f"main_plots/AngleVel_k{k}_c{c}_force.pdf", dpi=300)
    plt.show()

    # Rocker angle angular acceleration
    plt.plot(time, np.rad2deg(accel_phiO))
    plt.xlabel("Time (s)")
    plt.ylabel(r"Angular Acceleration (°/$s^2$)")
    plt.tight_layout()
    plt.savefig(f"main_plots/AngleAccel_k{k}_c{c}_force.pdf", dpi=300)
    plt.show()


########### COMPARATIVE PLOTS ##################

if COMPARATIVE_PLOT:
    for k, c in CONFIG:
        filepath = f"main_results/k{k}_c{c}.csv"

        df = pd.read_csv(filepath)

        acc_yH = df["ddq_yH"].to_numpy()
        vel_yH = df["dq_yH"].to_numpy()
        displ_yH = df["q_yH"].to_numpy()
        time = df["time"].to_numpy()

        accel_phiO = df["ddq_phiO"].to_numpy()
        vel_phiO = df["dq_phiO"].to_numpy()
        displ_phiO = df["q_phiO"].to_numpy()

        plt.plot(time, displ_yH, label=f"k = {k:.1e}")
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical Position (m)")

        print(
            f"k = {k} -- Wheel Position SteadyState : {displ_yH[-1]} m")

    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"main_plots/comp_WheelDispl.pdf", dpi=300)
    plt.show()

    for k, c in CONFIG:
        filepath = f"main_results/k{k}_c{c}.csv"

        df = pd.read_csv(filepath)

        acc_yH = df["ddq_yH"].to_numpy()
        vel_yH = df["dq_yH"].to_numpy()
        displ_yH = df["q_yH"].to_numpy()
        time = df["time"].to_numpy()

        accel_phiO = df["ddq_phiO"].to_numpy()
        vel_phiO = df["dq_phiO"].to_numpy()
        displ_phiO = df["q_phiO"].to_numpy()

        print(
            f"k = {k} -- Rocker Angle SteadyState : {np.rad2deg(displ_phiO)[-1]} °")

        plt.plot(time, np.rad2deg(displ_phiO), label=f"k = {k:.1e}")
        plt.xlabel("Time (s)")
        plt.ylabel("Rocker Angle (°)")

    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"main_plots/comp_RockerAngle.pdf", dpi=300)
    plt.show()
