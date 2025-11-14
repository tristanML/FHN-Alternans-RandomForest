from numerical_integrator import *
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from time import time


compound_v_vals = []
compound_w_vals = []
compound_t_vals = []

version = 0
mu = 1
alpha = 0.2
beta = 1.09
lower_beta = 1.04
mid_beta = 1.0889
max_beta = 1.119
eps = 0.04

I_stim = 0.5
T = 230
duration = 1
t_start = 0
n_T = 5

dt = 1
t_f = T*n_T
n = int(np.ceil(t_f / dt))

# # Initial Conditions and Variables Storage Arrays
v_0 = 0
w_0 = 0.1
t_0 = 0

percent_value = 7/8

base_param = [version, v_0, w_0, t_0, dt, n, [mu, alpha], [eps, beta], [T, duration, I_stim, t_start]]
ti = time()
for en in np.linspace(0.004, 0.014, 10):
    for T in range(50, 150, 1):
        base_param[-1][0] = T
        base_param[-2][0] = en
        v_val, w_val, t_val = run2(base_param)
        compound_v_vals.append(v_val)
        compound_w_vals.append(w_val)
        compound_t_vals.append(t_val)
print(time()-ti)
fig = plt.figure(figsize=(9, 9))

for i in range(len(compound_v_vals)):
    plt.plot(compound_t_vals[i], compound_v_vals[i], marker='', linestyle='-', color="b")
    plt.plot(compound_t_vals[i], compound_w_vals[i], marker='', linestyle='-', color="r")

plt.grid(True)
plt.show()
