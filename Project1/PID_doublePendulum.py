import matplotlib.animation as animation
import numpy as np
from Newmark_KDM import newmark
import matplotlib.pyplot as plt


"""
System quantities
############# SET OF ACTIVE COORDINATES #############
q = [x1, y1, theta1, x2, y2, theta2]
4 constraints => 2 DOF left
ndof = 6, nconst = 4
"""
l1 = 1
l2 = 1
m1 = 1
m2 = 1
n_dof = 6
n_const = 4


def g(q):
    x1, y1, theta1, x2, y2, theta2 = q
    return np.array([
        x1 - l1*np.sin(theta1),
        y1 + l1*np.cos(theta1),
        x2 - x1 - l2*np.sin(theta2),
        y2 - y1 + l2*np.cos(theta2)
    ])


def gravity(t):
    return np.array([0, -m1*9.81, 0, 0, -m2*9.81, 0])


I1 = (1/12)*m1*l1**2
I2 = (1/12)*m2*l2**2

M = np.diag([m1, m1, I1/1e1, m2, m2, I2/1e1])


def G(q):
    x1, y1, theta1, x2, y2, theta2 = q
    return np.array([
        [1, 0, -l1*np.cos(theta1), 0, 0, 0],
        [0, 1, -l1*np.sin(theta1), 0, 0, 0],
        [-1, 0, 0, 1, 0, -l2*np.cos(theta2)],
        [0, -1, 0, 0, 1, -l2*np.sin(theta2)]
    ])


IC = np.array([[1, 0, np.pi/2, 2, 0, np.pi/2], np.zeros(6)]).T


"""
PID control
Goal : pendulum at vertical top position
actuators : torque at both joints
"""

theta_ref = np.array([np.pi, np.pi/2])

Kp = [100, 100]
Kd = [20, 20]
Ki = [20, 20]

e1_prev = 0
e2_prev = 0
e_int = np.zeros(2)


def f(t, q):
    global e1_prev, e2_prev, e_int

    theta1 = q[2]
    theta2 = q[5]

    e1 = theta_ref[0] - theta1
    e2 = theta_ref[1] - theta2

    de1 = (e1 - e1_prev)/dt
    de2 = (e2 - e2_prev)/dt

    e_int[0] += e1 * dt
    e_int[1] += e2 * dt

    tau1 = (
        Kp[0]*e1 +
        Kd[0]*de1
    )
    tau2 = (
        Kp[1]*e2 +
        Kd[1]*de2
    )

    e1_prev = e1
    e2_prev = e2

    return np.array([0, 0, tau1, 0, 0, tau2])


# Simulation parameters
dt = 0.001
T = 5
q = np.zeros((n_dof, int(T/dt)+1))
dq = np.zeros_like(q)
ddq = np.zeros_like(q)
lambdas = np.zeros((n_const, int(T/dt)+1))
q[:, 0] = IC[:, 0]
dq[:, 0] = IC[:, 1]
f_used = np.zeros((n_dof, int(T/dt)+1))
# Time loop
for i in range(int(T/dt)):
    f_actuators = f(i*dt, q[:, i])
    f_used[:, i] = f_actuators

    def f_ext(t):
        return gravity(t) + f_actuators
    a, b, c, d = newmark(M, G, g, f_ext, dt, dt, np.array([q[:, i],
                                                           dq[:, i]]).T, C_t=None, K_t=None,
                         tol_res=1e-6, tol_g=1e-6, gamma=1/2+0.01, beta=1/4+0.015)
    q[:, i+1], dq[:, i+1], ddq[:, i+1], lambdas[:,
                                                i+1] = a[:, 1], b[:, 1], c[:, 1], d[:, 1]

time = np.linspace(0, T, int(T/dt))


############ ANIMATION ############
x1 = q[0, :]
y1 = q[1, :]
x2 = q[3, :]
y2 = q[4, :]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid()

# lines and masses
line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def update(frame):
    # first link: origin -> mass1
    line1.set_data([0, x1[frame]], [0, y1[frame]])

    # second link: mass1 -> mass2
    line2.set_data([x1[frame], x2[frame]],
                   [y1[frame], y2[frame]])
    return line1, line2


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time),
    init_func=init,
    interval=dt*1000,
    blit=True
)

plt.show()
