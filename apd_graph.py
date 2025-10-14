from numerical_integrator import run
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import statistics

version = 0

if version == 0:
    mu = 1
    alpha = 0.2
    beta = 1.09
    lower_beta = 1.04
    mid_beta = 1.0889
    max_beta = 1.119
    eps = 0.04
    v_param = [mu, alpha]
    w_param = [eps, beta]

if version == 4:
    alpha = 0.1
    gamma = 1
    beta = 0.3
    eps = 0.05
    delta = 0.07
    v_param = alpha
    w_param = [eps, beta, gamma, delta]

# Stimulation Parameters
I_stim = 0.5
T = 230
duration = 1
t_start = 0
n_T = 500

I_param = [T, duration, I_stim, t_start]

# Time Parameters
dt = 1
t_f = T*n_T
n = int(np.ceil(t_f / dt))

# Initial Conditions and Variables Storage Arrays
v_0 = 0
w_0 = 0.1
t_0 = 0
v_val = np.zeros(n + 1)
w_val = np.zeros(n + 1)
t_val = np.zeros(n + 1)
v_val[0] = v_0
w_val[0] = w_0
t_val[0] = t_0

percent_value = 7/8

def apd_calc(t_val, v_val, percent_value):
    threshold = 0.3*max(v_val)
    i2 = []
    for i in range(0, len(v_val)-1):
        if (v_val[i] - threshold)*(v_val[i+1] - threshold) < 0:
            i2.append([((v_val[i+1] - threshold)*t_val[i+1] + (threshold - v_val[i])*t_val[i])/(v_val[i+1]-v_val[i]),threshold])

    apd = []
    for i in range(0, len(i2)-1, 2):
        apd.append(i2[i+1][0]-i2[i][0])

    apd_calc_start = round(percent_value*len(apd))

    apd1 = np.mean(list(apd[i] for i in range(apd_calc_start, len(apd), 2)))
    apd2 = np.mean(list(apd[i] for i in range(apd_calc_start+1, len(apd), 2)))
    avg_apd = (apd1+apd2)/2
    return sorted([apd1, apd2]), avg_apd

apd_list = []
avs_list = []
T_list = []

for T in range(10, 300, 10):
    I_param = [T, duration, I_stim, t_start]
    v_val, w_val, t_val = run(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param)
    apds, av = apd_calc(t_val, v_val, percent_value) 
    apd_list.append(apds)
    avs_list.append(av)
    T_list.append(1/T)

fig = plt.figure(figsize=(9, 9))

plt.plot(T_list, apd_list, marker='', linestyle='-', color='r')
plt.plot(T_list, avs_list, marker='', linestyle='dashed', color='b')

plt.grid(True)
plt.show()