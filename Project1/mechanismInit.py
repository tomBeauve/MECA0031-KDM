from mechanism import Mechanism
import numpy as np

# Magnitude of the external force from the statement
force_mag = 100


def g(q, param):
    """
    g(q) = 0 is the vector of constraints, as computed in the report
    """
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    L1, L2, L3, D = param["L1"], param["L2"], param["L3"], param["D"]
    x_r, y_r, d = param["x_r"], param["y_r"], param["d"]

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


def G(q, param):
    """
    G(q) is the constraint gradient matrix, as defined in the report
    """
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    L1, L2, L3, D, n_const, n_dof = param["L1"], param["L2"], param[
        "L3"], param["D"], param["n_const"], param["n_dof"]
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


def K_t(q, lambdas, param):
    """
    K_t(q, lambda) is the tangent stiffness matrix, as defined in the report
    """
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    L1, L2, L3, D, x_r, x_s0, y_s0, phi_0, k, n_dof = param["L1"], param["L2"], param["L3"], param[
        "D"], param["x_r"], param["x_s0"], param["y_s0"], param["phi_0"], param["k"], param["n_dof"]
    x_r, y_r = param["x_r"], param["y_r"]

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

    # Spring contribution
    x_s = x_r + (D/2) * np.cos(phiO + phi_0)
    y_s = y_r + (D/2) * np.sin(phiO + phi_0)

    K_t[16, 16] += k * D**2 / 4 - k * \
        (x_s - x_s0) * (x_s - x_r) - k * (y_s - y_s0) * (y_s - y_r)

    return K_t


def C_t(q, dq, lambdas, param):
    """
    C_t(q, dq, lambda) is the tangent damping matrix, as defined in the report
    """
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    c, D, n_dof = param["c"], param["D"], param["n_dof"]
    C_t = np.zeros((n_dof, n_dof))

    # Damper contribution
    C_t[16, 16] += c * D**2/4
    return C_t


def f_ext(t, q, dq, param):
    """
    f_ext(q, dq, t) is the vector of external forces, as defined in the report
    Sums the contributions of external load, gravity, damper and spring
    """
    xA, yA, xC, yC, xE, yE, xG, yG, xH, yH, phiH, phiJ, xL, yL, phiL, phiN, phiO = q
    dphiO = dq[16]

    L1, L2, L3, D, x_r, y_r, x_s0, y_s0, phi_0, k, c, n_dof, mu1, mu2, mu3, d = \
        param["L1"], param["L2"], param["L3"], param["D"], param["x_r"], param["y_r"], param["x_s0"], param["y_s0"], \
        param["phi_0"], param["k"], param["c"], param["n_dof"], param["mu1"], param["mu2"], param["mu3"], param["d"]

    f = np.zeros(n_dof)

    ###### External force ######
    f[9] += force_mag * np.exp(-(t-0.5)**2)

    ##### Gravity #####
    g = 9.81
    f[1] -= (mu3/2 * L3/3) * g
    f[3] -= (mu3/2 * (2*L3/3 - 2*L3/5) + mu3/2 * L3/3 + mu2/2 * L2) * g
    f[5] -= (mu3/2 * (2*L3/5 - L3/3) + mu3/2 *
             (2*L3/3 - 2*L3/5) + mu1/2 * L1) * g
    f[7] -= (mu3/2 * L3/3 + mu3/2 * (2*L3/5 - L3/3) + mu1/2 * L1) * g
    f[9] -= (mu3/2 * L3/3) * g
    f[13] -= (mu2/2 * L2) * g

    ##### spring force #######
    x_sF = x_r + (D/2) * np.cos(phiO + phi_0)
    y_sF = y_r + (D/2) * np.sin(phiO + phi_0)

    lx = x_sF - x_s0
    ly = y_sF - y_s0

    f_xs = -k * lx
    f_ys = -k * ly

    tau_s = f_ys * (x_sF - x_r) - f_xs * (y_sF - y_r)

    f[16] += tau_s

    ####### Damper force ######
    tau_d = - c * D**2/4 * dphiO

    f[16] += tau_d

    return f


def M(param):
    """
    M is the mass matrix, a diagonal matrix with M_ii = sum (1/2 mass of elements connecting to the node i)
    """
    L1, L2, L3, D, mu1, mu2, mu3 = \
        param["L1"], param["L2"], param["L3"], param["D"], \
        param["mu1"], param["mu2"], param["mu3"]
    m1, m2, m3, m4 = param["m1"], param["m2"], param["m3"], param["m4"]

    M = np.diag([mu3/2 * L3/3,
                mu3/2 * L3/3,
                mu3/2 * (2*L3/3 - 2*L3/5) + mu3/2 * L3/3 + mu2/2 * L2,
                mu3/2 * (2*L3/3 - 2*L3/5) + mu3/2 * L3/3 + mu2/2 * L2,
                mu3/2 * (2*L3/5 - L3/3) + mu3/2 *
                 (2*L3/3 - 2*L3/5) + mu1/2 * L1,
                mu3/2 * (2*L3/5 - L3/3) + mu3/2 *
                 (2*L3/3 - 2*L3/5) + mu1/2 * L1,
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
    return M


# Parameters of the problem, as given in the statement
p = {
    "n_dof": 17,
    "n_const": 16,
    "m1": 2.5,
    "m2": 2.0,
    "m3": 15.0,
    "m4": 1.0,
    "L1": 0.5,
    "L2": 0.6,
    "L3": 0.72,
    "D": 0.15,
    "x_r": 0.1306,
    "y_r": 0.1934,
    "phi_0_deg": 120,
    "k": 5e4,
    "c": 6300,
    "d": 0.25,
    "l_0": 0.0
}

# derived quantities
p["phi_0"] = np.deg2rad(p["phi_0_deg"])
p["mu1"] = p["m1"] / p["L1"]
p["mu2"] = p["m2"] / p["L2"]
p["mu3"] = p["m3"] / p["L3"]
p["x_s0"] = p["x_r"]
p["y_s0"] = 0


# Initialization of the class to be used later
Suspension = Mechanism(M, G, g, f_ext=f_ext, C_t=C_t,
                       K_t=K_t, parameters=p)
