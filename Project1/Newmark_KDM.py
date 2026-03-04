import numpy as np


def newmark(M, G, g, f_ext, dt, T, IC, C_t=None, K_t=None, tol_res=1e-6, tol_g=1e-6, gamma=1/2, beta=1/4):
    """
    Performs time integration of a constrained system using Newmark algorithm

    Args:
        M : Mass matrix, assumed constant: shape ndof
        G(q) :Function returning the Matrix of gradient of constraints, returns 2D array shape (n_const x ndof)
        g(q) :Function returning vector of constraints, returns 1D array shape (n_const)
        f_ext(t) : Function returning external forces on each dof, returns 1D array shape (ndof)
        dt : time step
        T : total integration time
        IC : initial conditions, shape (ndof, 2) for initial q and dq
        C_t, K_t : Functions returning tangential stiffness and damping matrices, return 2D arrays shape (ndofxndof)
        gamma, beta : newmark parameters

    Returns:
        tuple of arrays (q, dq, ddq, lambdas) : 
        time evolution of input variables, velocities and accelerations each of size (ndof, int(T/dt))
        and lagrange multipliers of size (n_const, int(T/dt))
    """

    # Extract dimensions of the problem ( weird way to avoid parameters in function signature)
    n_dof = IC.shape[0]
    n_const = g(IC[:, 0]).shape[0]
    n_steps = int(T/dt)

    # Initialize arrays to store results
    q = np.zeros((n_dof, n_steps))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    lambdas = np.zeros((n_const, n_steps))

    # Set initial conditions, solve for init accel
    q_0 = IC[:, 0]
    dq_0 = IC[:, 1]
    q[:, 0] = q_0
    dq[:, 0] = dq_0
    ddq[:, 0] = np.linalg.solve(M, f_ext(0) - G(q_0).T @ lambdas[:, 0])

    # Time integration loop
    for n in range(n_steps - 1):
        # Compute the predictors ( explicit terms of the Newmark formulas)
        q_pred = q[:, n] + dt * dq[:, n] + \
            dt**2 * (1/2 - beta) * ddq[:, n]
        dq_pred = dq[:, n] + dt * (1-gamma) * ddq[:, n]

        q[:, n+1] = q_pred
        dq[:, n+1] = dq_pred

        # Newton Raphson Loop to solve the nonLinear system at each time step
        max_iter = 100

        for i in range(max_iter):
            print(f'Time step {n+1}/{n_steps}, NR iter {i}', end="\r")

            # Compute the linearized system quantities at the current guess
            C_t = np.zeros((n_dof, n_dof))
            K_t = np.zeros_like(C_t)

            res_guess = M @ ddq[:, n+1] + C_t @ dq[:, n+1] + \
                K_t @ q[:, n+1] - f_ext((n+1)*dt)
            g_guess = g(q[:, n+1])
            G_guess = G(q[:, n+1])

            # Convergence check
            if np.linalg.norm(res_guess) < tol_res and np.linalg.norm(g_guess) < tol_g:
                break

            # Solve the linearized system for the correction delta_q & delta_lambda
            S_11 = 1/(beta * dt**2) * M + gamma/(beta*dt) * C_t + K_t
            S_12 = G_guess.T
            S_21 = G_guess
            S_22 = np.zeros((n_const, n_const))
            S_t = np.block([[S_11, S_12], [S_21, S_22]])
            r_guess = np.concatenate([res_guess, g_guess])
            delta = np.linalg.solve(-S_t, r_guess)

            # update the guess for q, dq, ddq and lambdas
            delta_q = delta[:n_dof]
            q[:, n+1] += delta_q
            lambdas[:, n+1] += delta[n_dof:]
            dq[:, n+1] += gamma/(beta*dt) * delta_q
            ddq[:, n+1] += 1/(beta*dt**2) * delta_q

    return q, dq, ddq, lambdas
