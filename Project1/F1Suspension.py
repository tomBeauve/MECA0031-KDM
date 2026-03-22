import matplotlib.animation as animation
import numpy as np
from Newmark_KDM import newmark
import matplotlib.pyplot as plt
from math import asin
from initialConditions import IC

"""
############# SET OF ACTIVE COORDINATES #############
q = [x1, y1, theta1, x2, y2, theta2]
4 constraints => 2 DOF left
ndof = 6, nconst = 4
"""

n_dof = 17
n_const = 16
m1 = 2.5
m2 = 2.0
m3 = 15.0
m4 = 1.0
L1 = 0.5
L2 = 0.6
L3 = 0.72
mu1 = m1/L1
mu2 = m2/L2
mu3 = m3/L3
D = 0.15
x_r = 0.1306
y_r = 0.1934
phi_0_deg = 120  # degrees
phi_0 = np.deg2rad(phi_0_deg)  # radians
k = 5e3
d = 0.25
l_0 = 0.0
x_s0 = x_r
y_s0 = 0


def g(q):
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    return np.array([
        xA - xC - L3/3 * np.cos(phiH),
        yA - yC - L3/3 * np.sin(phiH),
        xC - xE - (2*L3/3 - 2*L3/5) * np.cos(phiH),
        yC - yE - (2*L3/3 - 2*L3/5) * np.sin(phiH),
        xE - xG - (2*L3/5 - L3/3) * np.cos(phiH),
        yE - yG - (2*L3/5 - L3/3) * np.sin(phiH),
        xG - xH - L3/3 * np.cos(phiH),
        yG - yH - L3/3 * np.sin(phiH),
        xC - L1 * np.cos(phiJ),
        yC - d - L1 * np.sin(phiJ),
        xE - xL - L2 * np.cos(phiL),
        yE - yL - L2 * np.sin(phiL),
        xG - L1*np.cos(phiN),
        yG - L1*np.sin(phiN),
        xL - x_r - D/2 * np.cos(phiO),
        yL - y_r - D/2 * np.sin(phiO)
    ])


def G(q):
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    G = np.zeros((n_const, n_dof))
    G[0, 0] = 1
    G[1, 1] = 1
    G[2, 2] = 1
    G[3, 3] = 1
    G[4, 4] = 1
    G[5, 5] = 1
    G[6, 6] = 1
    G[7, 7] = 1
    G[8, 2] = 1
    G[9, 3] = 1
    G[10, 4] = 1
    G[11, 5] = 1
    G[12, 6] = 1
    G[13, 7] = 1
    G[14, 12] = 1
    G[15, 13] = 1
    G[0, 2] = -1
    G[1, 3] = -1
    G[2, 4] = -1
    G[3, 5] = -1
    G[4, 6] = -1
    G[5, 7] = -1
    G[6, 8] = -1
    G[7, 9] = -1
    G[8, 11] = L1 * np.sin(phiJ)
    G[9, 11] = -L1 * np.cos(phiJ)
    G[10, 12] = -1
    G[11, 13] = -1
    G[12, 15] = L1*np.sin(phiN)
    G[13, 15] = -L1 * np.cos(phiN)
    G[14, 16] = D/2 * np.sin(phiO)
    G[15, 16] = -D/2 * np.cos(phiO)
    G[0, 10] = L3/3 * np.sin(phiH)
    G[1, 10] = -L3/3 * np.cos(phiH)
    G[2, 10] = (2*L3/3 - 2*L3/5) * np.sin(phiH)
    G[3, 10] = - (2*L3/3 - 2*L3/5) * np.cos(phiH)
    G[4, 10] = (2*L3/5 - L3/3) * np.sin(phiH)
    G[5, 10] = -(2*L3/5 - L3/3) * np.cos(phiH)
    G[6, 10] = L3/3 * np.sin(phiH)
    G[7, 10] = -L3/3 * np.cos(phiH)
    G[10, 14] = L2 * np.sin(phiL)
    G[11, 14] = -L2 * np.cos(phiL)

    return G


def K_t(q, lambdas):
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    K_t = np.zeros((n_dof, n_dof))
    K_t[10, 10] =\
        lambdas[0] * (L3/3 * np.cos(phiH)) + \
        lambdas[1] * (L3/3 * np.sin(phiH)) + \
        lambdas[2] * ((2*L3/3 - 2*L3/5) * np.cos(phiH)) + \
        lambdas[3] * ((2*L3/3 - 2*L3/5) * np.sin(phiH)) + \
        lambdas[4] * ((2*L3/5 - L3/3) * np.cos(phiH)) + \
        lambdas[5] * ((2*L3/5 - L3/3) * np.sin(phiH)) + \
        lambdas[6] * (L3/3 * np.cos(phiH)) + \
        lambdas[7] * (L3/3 * np.sin(phiH))
    K_t[11, 11] =\
        lambdas[8] * L1 * np.cos(phiJ) +\
        lambdas[9] * L1 * np.sin(phiJ)
    K_t[14, 14] = \
        lambdas[10] * (L2 * np.cos(phiL)) +\
        lambdas[11] * (L2 * np.sin(phiL))
    K_t[15, 15] = \
        lambdas[12] * (L1 * np.cos(phiN)) +\
        lambdas[13] * (L1 * np.sin(phiN))
    K_t[16, 16] = \
        lambdas[14] * (D/2 * np.cos(phiO)) +\
        lambdas[15] * (D/2 * np.sin(phiO))

    x_s = x_r + (D/2) * np.cos(phiO + phi_0)
    y_s = y_r + (D/2) * np.sin(phiO + phi_0)
    dxS_dphi = -D/2 * np.sin(phiO + phi_0)
    dyS_dphi = D/2 * np.cos(phiO + phi_0)

    K_t[16, 16] += k * D**2 / 4 - k * \
        (x_s - x_s0) * (x_s - x_r) - k * (y_s - y_s0) * (y_s - y_r)

    return K_t


def C_t(q, lambdas):
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    C_t = np.zeros((n_dof, n_dof))
    return C_t


def f_ext(t, q, dq):
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    f = np.zeros(n_dof)
    f[9] = 100 * np.exp(-(t-0.5)**2)

    # spring force
    x_sF = x_r + (D/2) * np.cos(phiO + phi_0)
    y_sF = y_r + (D/2) * np.sin(phiO + phi_0)

    lx_s = x_sF - x_s0
    ly_s = y_sF - y_s0

    f_x = -k * lx_s
    f_y = -k * ly_s

    tau = f_y * (x_sF - x_r) - f_x * (y_sF - y_r)

    f[16] = tau

    return f


M = np.diag([mu3/2 * L3/3,
             mu3/2 * L3/3,
             mu3/2 * (2*L3/3 - 2*L3/5) + mu3/2 * L3/3 + mu2/2 * L2,
             mu3/2 * (2*L3/3 - 2*L3/5) + mu3/2 * L3/3 + mu2/2 * L2,
             mu3/2 * (2*L3/5 - L3/3) + mu3/2 * (2*L3/3 - 2*L3/5) + mu1/2 * L1,
             mu3/2 * (2*L3/5 - L3/3) + mu3/2 * (2*L3/3 - 2*L3/5) + mu1/2 * L1,
             mu3/2 * L3/3 + mu3/2 * (2*L3/5 - L3/3) + mu1/2 * L1,
             mu3/2 * L3/3 + mu3/2 * (2*L3/5 - L3/3) + mu1/2 * L1,
             mu3/2 * L3/3,
             mu3/2 * L3/3,
             m3 * L3**2/3,
             m1 * L1**2/3,
             mu2/2 * L2,
             mu2/2 * L2,
             m2 * L2**2/3,
             m1 * L1**2/3,
             m4*D**2/8])


dt = 0.01
T = 10
"""
IC = np.array([[-L1,
                2/3*L3,
                -L1,
                L3/3,
                -L1,
                2/5*L3 - L3/3,
                -L1,
                0,
                -L1,
                -L3/3,
                np.pi/2,
                asin(float((d-L3)/L1)),
                x_r - D/2,
                y_r,
                asin(float((y_r - (2/5*L3 - L3/3)/L2))),
                np.pi,
                np.pi-0.1],
               np.zeros(n_dof)]).T

"""
time = np.linspace(0, T, int(T/dt))


q, dq, ddq, lambdas = newmark(M, G, g, f_ext, dt, T, IC, C_t=C_t, K_t=K_t,
                              tol_res=1e-12, tol_g=1e-12, gamma=1/2+0.05, beta=1/4+0.1)


for i in range(n_const):
    plt.plot(time, g(q[:, :])[i], label=f"g {i}")
plt.legend()
plt.title("constraints vs time")
plt.show()

############ ANIMATION ############
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
