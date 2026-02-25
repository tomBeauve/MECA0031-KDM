import numpy as np


def newmark(M, G, g, f_ext, dt, T, IC, gamma=1/2, beta=1/4):
    """
    Parameters
    M : Mass matrix
    G(q) :Function returning the Matrix of gradient of constraints
    g(q) :Function returning vector of constraints
    f_ext : external forces at each time step, shape :n_steps
    dt : time step
    T : total integration time
    IC : initial conditions, shape (ndof, 2) for initial q and dq
    gamma, beta : newmark parameters
    """

    n_steps = int(T/dt)
    n_dof = np.shape(M)[0]
    n_const = np.shape(g)[0]

    q = np.zeros((n_dof, n_steps))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    lambdas = np.zeros((n_const, n_steps))
    q_0, dq_0 = IC
    q[:, 0] = q_0
    dq[:, 0] = dq_0
    ddq[:, 0] = np.linalg.solve(M, f_ext[:, 0] - G(q_0) @ lambdas[:, 0])

    for i in range(n_steps - 1):
