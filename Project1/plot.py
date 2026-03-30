import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "text.usetex": True,
    "image.cmap": "cividis",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9
})


################## Single Plot ######################
k = 0.0
c = 1000


filepath = f"main_results/k{k}_c{c}.csv"

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
plt.savefig(f"main_plots/wheelDispl_k{k}_c{c}.png", dpi=300)
plt.show()
print(f"k = {k}, c = {c}")
print(f"Wheel Position SteadyState : {displ_yH[-1]} m")

#############################################
# Rocker angle displacement
plt.plot(time, np.rad2deg(displ_phiO))
plt.xlabel("Time (s)")
plt.ylabel("Rocker Angle (°)")
plt.tight_layout()
plt.savefig(f"main_plots/AngleDispl_k{k}_c{c}.png", dpi=300)
plt.show()

print(f"Rocker Angle SteadyState : {np.rad2deg(displ_phiO)[-1]} °")


########### COMPARATIVE PLOTS ##################

CONFIG = [[0.0, 1000], [50000.0, 1000]]

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

plt.legend()
plt.savefig(f"main_plots/comp_WheelDispl.png", dpi=300)
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

    plt.plot(time, np.rad2deg(displ_phiO), label=f"k = {k:.1e}")

plt.legend()
plt.savefig(f"main_plots/comp_RockerAngle.png", dpi=300)
plt.show()
