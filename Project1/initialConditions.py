import numpy as np

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
phi_0 = np.radians(phi_0_deg)  # radians
k = 5e4
d = 0.25
l_0 = 0.0


# 1. Choose an independent starting angle for the main vertical bar
phiH_0 = np.pi / 2  # Perfectly vertical

# 2. Position of Node G (Bottom pivot link L1)
# g[12]: xG - L1*cos(phiN) = 0
# g[13]: yG - L1*sin(phiN) = 0
phiN_0 = np.pi  # Pointing left
xG_0 = L1 * np.cos(phiN_0)
yG_0 = L1 * np.sin(phiN_0)

# 3. Propagate positions along the rigid bar L3 based on phiH
# We use G as the reference point (index 6,7)
# Segments: G-H is L3/3 | G-E is (2L3/5 - L3/3) | E-C is (2L3/3 - 2L3/5) | C-A is L3/3
xH_0 = xG_0 - (L3/3) * np.cos(phiH_0)
yH_0 = yG_0 - (L3/3) * np.sin(phiH_0)

xE_0 = xG_0 + (2*L3/5 - L3/3) * np.cos(phiH_0)
yE_0 = yG_0 + (2*L3/5 - L3/3) * np.sin(phiH_0)

xC_0 = xE_0 + (2*L3/3 - 2*L3/5) * np.cos(phiH_0)
yC_0 = yE_0 + (2*L3/3 - 2*L3/5) * np.sin(phiH_0)

xA_0 = xC_0 + (L3/3) * np.cos(phiH_0)
yA_0 = yC_0 + (L3/3) * np.sin(phiH_0)

# 4. Solve for Dependent Angles (phiJ, phiL, phiO)
# phiJ: connects (0, d) to C
phiJ_0 = np.arctan2(yC_0 - d, xC_0)

# phiO: The rotor angle (Assuming the rotor is at x_r, y_r pointing to L)
# For the very start, let's assume L is at its leftmost position on the rotor
phiO_0 = np.pi
xL_0 = x_r + (D/2) * np.cos(phiO_0)
yL_0 = y_r + (D/2) * np.sin(phiO_0)

# phiL: connects E to L
phiL_0 = np.arctan2(yE_0 - yL_0, xE_0 - xL_0)

# 5. Assemble the final IC vector
q0 = np.array([
    xA_0, yA_0,   # 0, 1
    xC_0, yC_0,   # 2, 3
    xE_0, yE_0,   # 4, 5
    xG_0, yG_0,   # 6, 7
    xH_0, yH_0,   # 8, 9
    phiH_0,       # 10
    phiJ_0,       # 11
    xL_0, yL_0,   # 12, 13
    phiL_0,       # 14
    phiN_0,       # 15
    phiO_0        # 16
])

dq0 = np.zeros(n_dof)
IC = np.vstack([q0, dq0]).T
