from numerical_integrator import run
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import statistics

version = 0

if version == 0:
    mu = 1
    alpha = 0.1
    beta = 1.09
    eps = 0.005
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

I_param = [T, duration, I_stim, t_start]

# Time Parameters
dt = 0.1
t_f = 10000
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

threshold = 0.2

v_val, w_val, t_val, intersect = run(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param)

i2 = []
for i in range(0, len(v_val)-1):
    if (v_val[i] - threshold)*(v_val[i+1] - threshold) < 0:
        i2.append([((v_val[i+1] - threshold)*t_val[i+1] + (threshold - v_val[i])*t_val[i])/(v_val[i+1]-v_val[i]),threshold])

apd = []
for i in range(0, len(i2)-1, 2):
    apd.append(i2[i+1][0]-i2[i][0])

start = 0
apd1 = np.mean(list(apd[i] for i in range(start, len(apd), 2)))
apd2 = np.mean(list(apd[i] for i in range(start+1, len(apd), 2)))

print(apd1, apd2)

v_null = np.linspace(min(v_val), max(v_val), 500)
if version == 0:
    w_null_1 = list(v*(beta-v) for v in v_null)
    w_null_2 = list(mu*(1-v)*(v-alpha) for v in v_null)
if version == 4:
    w_null_1 = list((beta * v - delta)/gamma for v in v_null)
    w_null_2 = list(v * (1 - v) * (v - alpha) for v in v_null)

graph = True

fig = plt.figure(figsize=(10, 10))
plt.plot(t_val, v_val, marker='', linestyle='-', color='r')
plt.plot(t_val, w_val, marker='', linestyle='-', color='b')
# plt.plot(list(i[0] for i in intersect), list(i[1] for i in intersect), marker='', linestyle='-', color='g')
# plt.plot(list(i[0] for i in intersect), list(i[2] for i in intersect), marker='', linestyle='-', color='g')
plt.plot(list(i[0] for i in i2), list(i[1] for i in i2), marker='x', linestyle='-', color='g')
# alpha_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor="r")
# alpha_slider = Slider(alpha_slider_ax, 'mu', 0.0, 0.5, valinit=alpha)

# def sliders_on_changed(val):
#     run(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param)
#     print(v_0, w_0, t_0)
#     plt.plot(t_val, v_val, marker='', linestyle='-', color='r')
#     plt.plot(t_val, w_val, marker='', linestyle='-', color='b')

# alpha_slider.on_changed(sliders_on_changed)


# plt.plot([t_val[0],t_val[-1]], [v_bar,v_bar], marker='', linestyle='-')
# plt.plot(t_intersect, list(v_bar for i in range(len(t_intersect))), marker='o', linestyle = '')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(v_val, w_val, marker='', linestyle='-')
plt.plot(v_null, w_null_1, marker='', linestyle='-')
plt.plot(v_null, w_null_2, marker='', linestyle='-')
plt.plot(list(i[1] for i in intersect), list(i[2] for i in intersect), marker='', linestyle='', color='b')
plt.grid(True)
plt.show()