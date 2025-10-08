import matplotlib.pyplot as plt
import math
import numpy as np

version = 0

# Differential Equation Parameters

if version == 0:
    mu = 1
    alpha = 0.2
    beta = 1.09
    eps = 0.005

if version == 4:
    alpha = 0.1
    gamma = 1
    beta = 0.3
    eps = 0.05
    delta = 0.07

# Stimulation Parameters
I_stim = 0.5
T = 230
duration = 1

t_start = 0
stimulation_on = True

# Time Parameters
dt = 0.1
t_f = 4000
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


intersect = []

def v_diff(v, w, t):
    I = 0
    if t % T <= duration and t >= t_start and stimulation_on == True:
        I = I_stim
        intersect.append([t,v,w])
    if version == 0:
        return mu * v * (1 - v) * (v - alpha) - v * w + I
    if version == 4:
        return v * (1 - v) * (v - alpha) - w + I

def w_diff(v, w):
    if version == 0:
        return eps * (v * (beta - v) - w)
    if version == 4:
        return eps * (beta * v - gamma * w - delta)

def step(v, w, t, dt):
    v += v_diff(v, w, t) * dt
    w += w_diff(v, w) * dt
    t += dt
    return v, w, round(t,1)

for i in range(n):
    v_0, w_0, t_0 = step(v_0, w_0, t_0, dt)
    v_val[i+1] = v_0
    w_val[i+1] = w_0
    t_val[i+1] = t_0

v_null = np.linspace(min(v_val), max(v_val), 500)
print(min(v_val), max(v_val))
if version == 0:
    w_null_1 = list(v*(beta-v) for v in v_null)
    w_null_2 = list(mu*(1-v)*(v-alpha) for v in v_null)
if version == 4:
    w_null_1 = list((beta * v - delta)/gamma for v in v_null)
    w_null_2 = list(v * (1 - v) * (v - alpha) for v in v_null)

graph = True

plt.figure(figsize=(8, 8))
plt.plot(t_val, v_val, marker='', linestyle='-', color='r')
plt.plot(t_val, w_val, marker='', linestyle='-', color='b')
plt.plot(list(i[0] for i in intersect), list(i[1] for i in intersect), marker='x', linestyle='', color='b')
plt.plot(list(i[0] for i in intersect), list(i[2] for i in intersect), marker='x', linestyle='', color='b')
# plt.plot([t_val[0],t_val[-1]], [v_bar,v_bar], marker='', linestyle='-')
# plt.plot(t_intersect, list(v_bar for i in range(len(t_intersect))), marker='o', linestyle = '')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(v_val, w_val, marker='', linestyle='-')
plt.plot(v_null, w_null_1, marker='', linestyle='-')
plt.plot(v_null, w_null_2, marker='', linestyle='-')
plt.plot(list(i[1] for i in intersect), list(i[2] for i in intersect), marker='x', linestyle='', color='b')
plt.grid(True)
plt.show()


# IPY widgets