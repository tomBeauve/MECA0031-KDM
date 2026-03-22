import matplotlib.animation as animation
import numpy as np
from Newmark_KDM import newmark
import matplotlib.pyplot as plt

"""
############# SET OF ACTIVE COORDINATES #############
q = [x1, y1, theta1, x2, y2, theta2]
4 constraints => 2 DOF left
ndof = 6, nconst = 4
"""
l1 = 1
l2 = 1
m1 = 1
m2 = 1
n_dof = 3
n_const = 2


def g(q):
    x1, y1, theta1 = q
    return np.array([
        x1 - l1*np.sin(theta1),
        y1 + l1*np.cos(theta1),
    ])


def f_ext(t):
    return np.array([0, -m1*9.81, 0])


I1 = (1/12)*m1*l1**2
I2 = (1/12)*m2*l2**2

M = np.diag([m1, m1, 0])


def G(q):
    x1, y1, theta1 = q
    return np.array([
        [1, 0, -l1*np.cos(theta1)],
        [0, 1, -l1*np.sin(theta1)]
    ])


dt = 0.01
T = 5
IC = np.array([[1, 0, np.pi/2], np.zeros(3)]).T
time = np.linspace(0, T, int(T/dt))


q, dq, ddq, lambdas = newmark(M, G, g, f_ext, dt, T, IC, C_t=None, K_t=None,
                              tol_res=1e-6, tol_g=1e-6, gamma=1/2+0.001, beta=1/4+0.0015)


energy_potential = m1*9.81*q[1, :]
energy_kinetic = 0.5*m1*(dq[0, :]**2 + dq[1, :]**2)

energy = energy_potential + energy_kinetic

plt.plot(time, energy)
plt.title("Energy vs time")
plt.show()


plt.plot(q[0, :], q[1, :])
plt.title("x vs y second pendulum")
plt.show()
plt.plot(time, q[2, :])
plt.title("angle 1 vs time")
plt.show()

g0_array = np.zeros_like(time)
g1_array = np.zeros_like(time)

for i in range(len(time)):
    g0_array[i] = g(q[:, i])[0]
    g1_array[i] = g(q[:, i])[1]
plt.plot(time, g0_array, label="g0")
plt.plot(time, g1_array, label="g1")
plt.legend()
plt.title("constraints vs time")
plt.show()


############ ANIMATION ############
x1 = q[0, :]
y1 = q[1, :]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 1.5)
ax.grid()

# lines and masses
line1, = ax.plot([], [], 'o-', lw=2)


def init():
    line1.set_data([], [])
    return (line1,)


def update(frame):
    # first link: origin -> mass1
    line1.set_data([0, x1[frame]], [0, y1[frame]])
    return (line1,)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(time),
    init_func=init,
    interval=dt*1000,
    blit=True
)

plt.show()
