import numpy as np
import matplotlib.pyplot as plt
from initialConditions import IC
from mechanismInit import Suspension


def natFreq(q, time):

    dt = time[1] - time[0]
    peakIdx = np.zeros(len(time)-1)
    nbPeaks = 0
    for i in range(1, len(time) - 1):
        if q[i] > q[i-1] and q[i] > q[i+1]:
            peakIdx[nbPeaks] = i
            nbPeaks += 1

    peakIdx = peakIdx[:nbPeaks]
    Tn = dt * (peakIdx[1:] - peakIdx[:-1])
    fn = 1/Tn
    print(fn)
    return fn

#### INTEGRATION PARAMETERS  #####


dt = 0.001
T = 10
tol_res = 1e-6
tol_g = 1e-6
gamma = 1/2 + 0.05
beta = 1/4 * (gamma + 1/2)**2 + 0.1

###### NEWMARK ########

q, dq, ddq, lambdas = Suspension.Newmark(
    dt, T, IC, tol_res=tol_res, tol_g=tol_g, gamma=gamma, beta=beta)


######## POST PROCESSING #######


time = np.linspace(0, T, int(T/dt))


fn = natFreq(q[9, :], time)

plt.plot(fn, fn)
plt.show()

# Bottom of the wheel vertical displacement (yh) plotting
plt.plot(time, q[9, :])
plt.title("Wheel vertical displacement vs time")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m))")
plt.show()
