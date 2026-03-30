import numpy as np


class Mechanism:
    def __init__(self, M=None, G=None, g=None, f_ext=None, C_t=None, K_t=None, parameters=None):
        self.M = M
        self.G = G
        self.g = g
        self.f_ext = f_ext
        self.C_t = C_t
        self.K_t = K_t
        self.p = parameters

    def Newmark(self, dt, T, IC, tol_res=1e-6, tol_g=1e-6, gamma=1/2+1e-2, beta=1/4+1e-3):
        """
        Performs time integration of a constrained system using Newmark algorithm

        Args:
            self:
                M(params) : Mass matrix, assumed constant: shape ndof
                G(q, params) :Function returning the Matrix of gradient of constraints, returns 2D array shape (n_const x ndof)
                g(q, params) :Function returning vector of constraints, returns 1D array shape (n_const)
                f_ext(t, q, dq, params) : Function returning external forces on each dof, returns 1D array shape (ndof)
                C_t, K_t : Functions returning tangential stiffness and damping matrices, return 2D arrays shape (ndofxndof)
            dt : time step
            T : total integration time
            IC : initial conditions, shape (ndof, 2) for initial q and dq
            gamma, beta : newmark parameters
            tol_g, tol_res : tolerances on g ( constraints) and residuals for the non-linear iterative ( Newton-Raphson) solver

        Returns:
            tuple of arrays (q, dq, ddq, lambdas) : 
            time evolution of input variables, velocities and accelerations each of size (ndof, int(T/dt))
            and lagrange multipliers of size (n_const, int(T/dt))
        """
        M = self.M(self.p)
        G = self.G
        g = self.g
        f_ext = self.f_ext
        C_t = self.C_t
        K_t = self.K_t

        # Extract dimensions of the problem (to avoid passing additional argument in function signature)
        n_dof = IC.shape[0]
        n_const = g(IC[:, 0], self.p).shape[0]
        n_steps = int(T/dt)

        # Initialize arrays to store results
        q = np.zeros((n_dof, n_steps))
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
        lambdas = np.zeros((n_const, n_steps))

        # Set initial conditions, accel is left at 0
        q_0 = IC[:, 0]
        dq_0 = IC[:, 1]
        q[:, 0] = q_0
        dq[:, 0] = dq_0

        max_iter = 1000

        # Time integration loop
        for n in range(n_steps-1):
            # Compute the predictors (explicit terms of the Newmark formulas)
            q_pred = q[:, n] + dt * dq[:, n] + \
                dt**2 * (1/2 - beta) * ddq[:, n]
            dq_pred = dq[:, n] + dt * (1-gamma) * ddq[:, n]

            q[:, n+1] = q_pred
            dq[:, n+1] = dq_pred

            # Newton Raphson Loop to solve the non-linear system at each time step
            for i in range(max_iter):
                print(f'Time step {n+1}/{n_steps}, NR iter {i}', end="\r")

                # Compute the linearized system quantities at the current guess of state variables
                C_t_guess = C_t(q[:, n+1], dq[:, n+1], lambdas[:, n+1], self.p)
                K_t_guess = K_t(q[:, n+1], lambdas[:, n+1], self.p)
                g_guess = g(q[:, n+1], self.p)
                G_guess = G(q[:, n+1], self.p)

                res_guess = M @ ddq[:, n+1] + \
                    G_guess.T @ lambdas[:, n+1] - \
                    f_ext((n+1)*dt, q[:, n+1], dq[:, n+1], self.p)

                # Convergence check
                if np.linalg.norm(res_guess) < tol_res and np.linalg.norm(g_guess) < tol_g:
                    break

                # Solve the linearized system for the correction delta_q & delta_lambda
                S_11 = 1/(beta * dt**2) * M + gamma / \
                    (beta*dt) * C_t_guess + K_t_guess
                S_12 = G_guess.T
                S_21 = G_guess
                S_22 = np.zeros((n_const, n_const))

                delta_q, delta_lambdas = self.solveWithScaling(
                    S_11, S_12, S_21, S_22, res_guess, g_guess, n_dof, dt)

                # Update the guess with correctors
                q[:, n+1] += delta_q
                lambdas[:, n+1] += delta_lambdas
                dq[:, n+1] += gamma/(beta*dt) * delta_q
                ddq[:, n+1] += 1/(beta*dt**2) * delta_q

                # Special handling in case of no convergence, returns None for failure identification
                if (i == max_iter-1):
                    res_guess = M @ ddq[:, n+1] - \
                        f_ext((n+1)*dt, q[:, n+1], dq[:, n+1], self.p)
                    print(
                        f"\n[FAILURE] Step {n}: NR failed to converge after {max_iter} iterations.")
                    print(
                        f"Final Residual: {np.linalg.norm(res_guess):.2e}, Final g: {np.linalg.norm(g_guess):.2e}")
                    return None, None, None, None

        return q, dq, ddq, lambdas

    def SolveWitPreconditioning(self, S11, S12, S21, S22, r, g, ndof, dt):
        """
        Solves the linearized system using a preconditioner for better condition number of the matrix
        Preconditioning by normalizing each row by its maximum element
        """
        S = np.block([[S11, S12], [S21, S22]])

        row_scales = np.max(np.abs(S), axis=1)
        row_scales[row_scales == 0] = 1.0

        S_preconditioned = S / row_scales[:, np.newaxis]
        r_guess = np.concatenate([r, g])
        r_preconditiond = r_guess / row_scales

        # 3. Solve the well-conditioned system
        delta = np.linalg.solve(-S_preconditioned, r_preconditiond)

        # 4. Update (Note: delta_q and delta_lambda are now in original units)
        delta_q = delta[:ndof]
        delta_lambdas = delta[ndof:]

        return delta_q, delta_lambdas

    def solveWithScaling(self, S11, S12, S21, S22, r, g, ndof, dt):
        """
        Solves the linearized system using a scaling logic for better condition number of the matrix
        Scales off-diagonal blocks by a heuristic scaling factor for the smallest values of the matrix 1/dt**2 
        Approx 2 times faster than precondtitioning, use unless condition number is really bad
        """

        k = 1/(dt**2)
        S12 = k * S12
        S21 = k * S21
        S_t = np.block([[S11, S12], [S21, S22]])
        r_guess = np.concatenate([r, k*g])
        delta = np.linalg.solve(-S_t, r_guess)
        delta_q = delta[:ndof]
        delta_lambdas = k * delta[ndof:]

        return delta_q, delta_lambdas
