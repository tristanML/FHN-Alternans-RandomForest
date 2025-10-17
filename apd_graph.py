from numerical_integrator import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

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

# # Initial Conditions and Variables Storage Arrays
v_0 = 0
w_0 = 0.1
t_0 = 0

percent_value = 7/8
   
h = 0.012
k = -0.001
specific_T = 130

compound_v_vals = []
compound_w_vals = []
comound_t_vals = []
compound_T_vals = []
compound_apd_vals = []
alt_list = []
avs_list = []
avg_v_list = []
avg_w_list = []
apd_index_list = []

for i in range(8):
    eps = h + i*k
    w_param = [eps, beta]
    T_list, apd_list, avs_list, specifics = apd_grapher(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param, percent_value, specific_T)
    compound_T_vals.append(T_list)
    compound_apd_vals.append(apd_list)
    compound_v_vals.append(specifics[0])
    compound_w_vals.append(specifics[1])
    comound_t_vals.append(specifics[2])
    alt_list.append(specifics[3])
    avs_list.append(specifics[4])
    avg_v_list.append(specifics[5])
    avg_w_list.append(specifics[6])
    apd_index_list.append(specifics[8])

fig = plt.figure(figsize=(9, 9))
colors = ["r", "g", "b", "orange", "purple", "violet", "r", "g", "b", "orange", "purple", "violet",]

for i in range(len(compound_T_vals)):
    plt.plot(compound_T_vals[i], compound_apd_vals[i], marker='', linestyle='-', color=colors[2])
plt.plot((1/specific_T, 1/specific_T), (0,120), marker='', linestyle='-', color=colors[2])

plt.xlabel("Frequency f (Hz)")
plt.ylabel("APD (sec)")

plt.grid(True)
plt.show()

fig = plt.figure(figsize=(9, 9))

for i in range(len(compound_T_vals)):
    index_start = apd_index_list[i]
    plt.plot((apd_index_list[i]*dt, apd_index_list[i]*dt), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    plt.plot((apd_index_list[i]*dt+specific_T, apd_index_list[i]*dt+specific_T), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    plt.plot((apd_index_list[i]*dt+2*specific_T, apd_index_list[i]*dt+2*specific_T), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    if alt_list[i]:
        plt.plot((apd_index_list[i]*dt, t_f), (avg_v_list[i], avg_v_list[i]), marker='', linestyle='dashed', color=colors[3])
        plt.plot((apd_index_list[i]*dt, t_f), (avg_w_list[i], avg_w_list[i]), marker='', linestyle='dashed', color=colors[3])
        plt.plot(comound_t_vals[i][index_start:], compound_v_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
        plt.plot(comound_t_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
    else:
        plt.plot(comound_t_vals[i][index_start:], compound_v_vals[i][index_start:], marker='', linestyle='-', color=colors[2])
        plt.plot(comound_t_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[2])

plt.grid(True)
plt.show()

v_null = np.linspace(min(compound_v_vals[0]), max(compound_v_vals[0]), 500)
if version == 0:
    w_null_1 = list(v*(beta-v) for v in v_null)
    w_null_2 = list(mu*(1-v)*(v-alpha) for v in v_null)
if version == 4:
    w_null_1 = list((beta * v - delta)/gamma for v in v_null)
    w_null_2 = list(v * (1 - v) * (v - alpha) for v in v_null)

plt.figure(figsize=(8, 8))
for i in range(len(compound_T_vals)):
    index_start = apd_index_list[i]
    if alt_list[i]:
        plt.plot((0,avg_v_list[i]), (avg_w_list[i],avg_w_list[i]), marker='', linestyle='dashed', color=colors[0])
        # plt.plot(compound_v_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
    else:
        plt.plot(compound_v_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[2])

plt.plot(v_null, w_null_1, marker='', linestyle='-', color='orange')
plt.plot(v_null, w_null_2, marker='', linestyle='-', color='green')
# k_p = (beta-1)/4
# plt.plot((min(compound_v_vals[0]), max(compound_v_vals[0])), (k_p, k_p), marker='', linestyle='dashed', color='purple')

plt.xlabel("v")
plt.ylabel("w")

plt.grid(True)
plt.show()