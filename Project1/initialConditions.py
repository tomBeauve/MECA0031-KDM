import numpy as np
from mechanismInit import Suspension

n_dof = Suspension.p["n_dof"]


### Values obtained from steady state withspring ###

q_steady = np.array([
    -0.49791649,  0.45224999,  # A
    -0.49857297,  0.21225089,  # C
    -0.49909815,  0.02025161,  # E
    -0.49922944, -0.02774821,  # G
    -0.49988592, -0.26774731,  # H
    1.56806101,  # phiH
    -3.06602253,            # phiJ
    0.06390028,  0.22769501,  # L
    -2.78856641,               # phiL
    3.1971176,  # phiN
    2.66667319  # phiO
])


### Steady state with very strong spring ####
q_steady = np.array([
    -0.49871489,  0.46147826,  # A
    -0.49918587,  0.22147872,  # C
    -0.49956266,  0.02947909,  # E
    -0.49965686, -0.01852081,  # G
    -0.50012785, -0.25852035,  # H
    1.56883388,              # phiH
    -3.08451912,              # phiJ
    0.06562973,  0.23086817,  # L
    -2.79929911,              # phiL
    3.17864276,              # phiN
    2.61848386               # phiO
])

### Values obtained from steady state without spring ###
q_steady = np.array([
    -0.47041178,  0.32989699,  # A
    -0.47368149,  0.08991926,  # C
    -0.47629727, -0.10206292,  # E
    -0.47695121, -0.15005847,  # G
    -0.48022093, -0.39003619,  # H
    1.55717208,             # phiH
    -2.81569272,             # phiJ
    0.06337998,  0.16013638,  # L
    -2.68933332,             # phiL
    3.44640789,             # phiN
    3.60110934              # phiO
])


dq_steady = np.zeros(n_dof)

# Assemble IC as (n_dof, 2) where column 0 is q and column 1 is dq
IC = np.vstack([q_steady, dq_steady]).T
