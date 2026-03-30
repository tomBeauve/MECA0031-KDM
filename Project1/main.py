import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from initialConditions import IC
from mechanismInit import Suspension, force_mag
from pareto import T_steadyState
import pandas as pd

##### GEOMETRIC PARAMETERS ###
n_const = Suspension.p["n_const"]

L1 = Suspension.p["L1"]
L2 = Suspension.p["L2"]
L3 = Suspension.p["L3"]

D = Suspension.p["D"]
x_r = Suspension.p["x_r"]
y_r = Suspension.p["y_r"]
d = Suspension.p["d"]


#### INTEGRATION PARAMETERS  #####

dt = 0.001
T = 2
tol_res = 1e-6
tol_g = 1e-6
gamma = 1/2 + 0.1
beta = 1/4 * (gamma + 1/2)**2 + 0.1

dt_IC = 0.05
T_IC = 10

###### Newmark time integration ########


# Performs light time integration to find steady state of the given mechanism

q_steady, _, _, _ = Suspension.Newmark(
    dt_IC, T_IC, IC, tol_res, tol_g, gamma, beta)
dq_steady = np.zeros(len(IC))

# Steady state is passed as initial condition
# This avoids unphysical motion at the beginning
IC_specific = np.vstack([q_steady[:, -1], dq_steady]).T

q, dq, ddq, lambdas = Suspension.Newmark(
    dt, T, IC_specific, tol_res=tol_res, tol_g=tol_g, gamma=gamma, beta=beta)

# Save in a file to plot in other file

labels = ["xA", "yA", "xC", "yC", "xE", "yE", "xG", "yG", "xH",
          "yH", "phiH", "phiJ", "xL", "yL", "phiL", "phiN", "phiO"]

time_steps = q.shape[1]
time_array = np.linspace(0, T, time_steps)
data = {"time": time_array}

for i, label in enumerate(labels):
    data[f"q_{label}"] = q[i, :]
    data[f"dq_{label}"] = dq[i, :]
    data[f"ddq_{label}"] = ddq[i, :]

# Add Lagrange Multipliers and Constraint Violations
for i in range(lambdas.shape[0]):
    data[f"lambda_{i}"] = lambdas[i, :]
    data[f"g_{i}"] = Suspension.g(q, Suspension.p)[i]

df_sim = pd.DataFrame(data)

filepath = f"main_results/k{Suspension.p["k"]}_c{Suspension.p["c"]}_force.csv"
# df_sim.to_csv(filepath, index=False)


################ POST PROCESSING ####################

######## Line Plots ############

# Constraints time evolution
time = np.linspace(0, T, int(T/dt))
for i in range(n_const):
    plt.plot(time, Suspension.g(q[:, :], Suspension.p)[i], label=f"g {i}")
plt.legend()
plt.title("constraints vs time")
plt.show()


#### Force Plotting ####

plt.plot(time, force_mag * np.exp(-(time-0.5)**2))
plt.title("Force amplitude vs time")
plt.xlabel("Time (s)")
plt.ylabel("Force amplitude (N)")
plt.show()

# Rocker angle, vel and accel vs time evolution
plt.plot(time, np.rad2deg(q[16, :]))
plt.title("Rocker Angle vs time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (°)")
plt.show()


plt.plot(time, dq[16, :])
plt.title("Rocker Angular Velocity vs time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (°/s)")
plt.show()

plt.plot(time, ddq[16, :])
plt.title("Rocker Angular Acceleration vs time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (°/s**2)")
plt.show()

# Bottom of the wheel vertical displacement (yh) plotting
plt.plot(time, q[9, :])
plt.title("Wheel vertical displacement vs time")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m))")
plt.show()

# Bottom of the wheel vertical displacement (yh) plotting
plt.plot(time, (q[9, :] - np.min(q[9, :])) /
         (np.max(q[9, :]) - np.min(q[9, :])), label="displ")
plt.plot(time, np.exp(-(time-0.5)**2), label="force")
plt.title("Wheel vertical displacement vs time")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m))")
plt.legend()
plt.show()

# Bottom of the wheel vertical displacement (yh) plotting
plt.plot(time, (dq[9, :]))
plt.title("Wheel vertical velpcity vs time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.show()

# Bottom of the wheel vertical displacement (yh) plotting
plt.plot(time, ddq[9, :])
plt.title("Wheel vertical acc vs time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m))")
plt.show()

#############################################################
print(f"Min grip: {-np.min(ddq[9, :]):.4e} m/s^2")
print(
    f"Resulting Ride Height Var: {np.std(q[9, :] - np.mean(q[9, :])):.7f} m^2")
print(f"Resulting Settling Time: {T_steadyState(dq[9, :], time):.2f} s")

#################### ANIMATION ###########################
# 1. Setup Figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
# Adjust limits to see the whole mechanism clearly
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.0, 1.5)
ax.grid(True, linestyle='--')

# 2. Static Elements (Visual Guides)
# Draw the circular path of the rotor
theta_circle = np.linspace(0, 2*np.pi, 100)
ax.plot(x_r + D/2 * np.cos(theta_circle),
        y_r + D/2 * np.sin(theta_circle), 'k:', alpha=0.3, label='Rotor Path')

# 3. Dynamic Elements
bar_vert, = ax.plot([], [], 'ro-', lw=3, markersize=6,
                    label='Bar L3 (A-C-E-G-H)')
link_top, = ax.plot([], [], 'b-o', lw=2, label='Top L1')
link_bot, = ax.plot([], [], 'b-o', lw=2, label='Bottom L1')
link_mid, = ax.plot([], [], 'g-o', lw=2, label='Middle L2')
rotor_arm, = ax.plot([], [], 'k-o', lw=2, label='Rotor')

# Fixed pivot points from your g(q) definitions
# Top L1 pivot is at (0, d)
# Bottom L1 pivot is at (0, 0)
# Rotor center is at (x_r, y_r)


def init():
    bar_vert.set_data([], [])
    link_top.set_data([], [])
    link_bot.set_data([], [])
    link_mid.set_data([], [])
    rotor_arm.set_data([], [])
    return bar_vert, link_top, link_bot, link_mid, rotor_arm


def update(frame):
    # Extract coordinates
    # Indices based on your q:
    # 0,1:A | 2,3:C | 4,5:E | 6,7:G | 8,9:H | 12,13:L
    c_q = q[:, frame]

    # Update Vertical Bar (Nodes: A, C, E, G, H)
    # Note: We plot them in order so the line connects them properly
    vx = [c_q[0], c_q[2], c_q[4], c_q[6], c_q[8]]
    vy = [c_q[1], c_q[3], c_q[5], c_q[7], c_q[9]]
    bar_vert.set_data(vx, vy)

    # Top Link: (0, d) -> Node C (index 2,3)
    link_top.set_data([0, c_q[2]], [d, c_q[3]])

    # Bottom Link: (0, 0) -> Node G (index 6,7)
    link_bot.set_data([0, c_q[6]], [0, c_q[7]])

    # Middle Link: Node E (index 4,5) -> Node L (index 12,13)
    link_mid.set_data([c_q[4], c_q[12]], [c_q[5], c_q[13]])

    # Rotor Arm: (x_r, y_r) -> Node L (index 12,13)
    rotor_arm.set_data([x_r, c_q[12]], [y_r, c_q[13]])

    return bar_vert, link_top, link_bot, link_mid, rotor_arm


# 4. Run Animation
# interval = dt * 1000 for real-time speed
ani = animation.FuncAnimation(
    fig, update, frames=len(time), init_func=init,
    interval=dt*1000, blit=True
)

plt.legend(loc='upper right', fontsize='small')
plt.show()
