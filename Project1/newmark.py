from constants import gamma_newmark, beta_newmark
import numpy as np


def time_integ(K, M, C, f_ext, f_ext_t, dt, T, gamma=gamma_newmark, beta=beta_newmark):
    """
    K,M,C system matrices
    f_ext = shape(Ndof) : vector of initial forces applied
    f_ext_t = duration of the initial shock
    dt = integration time step
    T = total duration of the integration to perform
    """

    # compute some numbers
    N_steps = int(T / dt) + 1
    N_dof = M.shape[0]
    f_steps = int(np.ceil(f_ext_t / dt))
    # initialize arrays
    t = np.linspace(0, T, N_steps)
    q = np.zeros((N_dof, N_steps))
    q_dot = np.zeros((N_dof, N_steps))
    q_ddot = np.zeros((N_dof, N_steps))
    p = np.zeros((N_dof, N_steps))
    for i in range(f_steps):
        p[:, i] = f_ext

    # initial acceleration
    q_ddot[:, 0] = np.linalg.solve(M, p[:, 0] - C @ q_dot[:, 0] - K @ q[:, 0])

    # parameters:
    a0 = 1 / (beta * dt**2)
    a1 = 1 / (beta * dt)
    a2 = (1-2*beta) / (2 * beta)
    a3 = gamma*dt * a0
    a4 = 1 - gamma*dt*a1
    a5 = dt * (1 - gamma - gamma*a2)

    # effective stiffness matrix
    K_eff = K + a0 * M + a3 * C
    K_eff_inv = np.linalg.inv(K_eff)

    # time integration loop
    for n in range(N_steps - 1):
        if n % 100 == 0:
            print(f'Time step {n}/{N_steps}', end="\r")
        # effective force vector
        f_eff = p[:, n+1] + M@(a0 * q[:, n] + a1 * q_dot[:, n] + a2*q_ddot[:, n]) + \
            C @ (a3*q[:, n] - a4*q_dot[:, n] - a5*q_ddot[:, n])

        # solve
        # q[:, n+1] = np.linalg.solve(K_eff, f_eff)
        q[:, n+1] = K_eff_inv @ f_eff

        # compute next acceleration and velocity with newmark formulas
        q_ddot[:, n+1] = a0 * (q[:, n+1] - q[:, n]) - \
            a1 * q_dot[:, n] - a2 * q_ddot[:, n]
        q_dot[:, n+1] = a3 * (q[:, n+1] - q[:, n]) + a4 * \
            q_dot[:, n] + a5 * q_ddot[:, n]

    return q, q_dot, q_ddot, t


def time_integ_slides(K, M, C, f_ext, f_ext_t, dt, T, gamma=gamma_newmark, beta=beta_newmark):
    """
    K,M,C system matrices
    f_ext = shape(Ndof) : vector of initial forces applied
    f_ext_t = duration of the initial shock
    dt = integration time step
    T = total duration of the integration to perform
    """

    # compute some numbers
    N_steps = int(T / dt) + 1
    N_dof = M.shape[0]
    f_steps = int(np.ceil(f_ext_t / dt))
    # initialize arrays
    t = np.linspace(0, T, N_steps)
    q = np.zeros((N_dof, N_steps))
    q_dot = np.zeros((N_dof, N_steps))
    q_ddot = np.zeros((N_dof, N_steps))
    p = np.zeros((N_dof, N_steps))
    for i in range(f_steps):
        p[:, i] = f_ext

    # initial acceleration
    q_ddot[:, 0] = np.linalg.solve(M, p[:, 0] - C @ q_dot[:, 0] - K @ q[:, 0])

    # effective stiffness matrix
    S = M + dt*gamma * C + dt**2 * beta * K
    S_inv = np.linalg.inv(S)

    # time integration loop
    for n in range(N_steps - 1):

        if n % 100 == 0:
            print(f'Time step {n}/{N_steps}', end="\r")
        # prediction:
        q_pred = q[:, n] + dt * q_dot[:, n] + \
            dt**2 * (1/2 - beta) * q_ddot[:, n]
        q_dot_pred = q_dot[:, n] + dt * (1-gamma) * q_ddot[:, n]

        # computation of accelerations
        rhs = p[:, n+1] - C @ q_dot_pred - K @ q_pred
        # q_ddot[:, n+1] = np.linalg.solve(S, rhs)
        q_ddot[:, n+1] = S_inv @ rhs

        # correction:
        q[:, n+1] = q_pred + dt**2 * beta * q_ddot[:, n+1]
        q_dot[:, n+1] = q_dot_pred + dt * gamma * q_ddot[:, n+1]

    return q, q_dot, q_ddot, t
