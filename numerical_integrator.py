import math
import numpy as np

# version = 0

# Differential Equation Parameters

# if version == 0:
#     mu = 1
#     alpha = 0.2
#     beta = 1.09
#     eps = 0.005

# if version == 4:
#     alpha = 0.1
#     gamma = 1
#     beta = 0.3
#     eps = 0.05
#     delta = 0.07

# # Stimulation Parameters
# I_stim = 0.5
# T = 230
# duration = 1

# t_start = 0
# stimulation_on = True

# # Time Parameters
# dt = 0.1
# t_f = 4000
# n = int(np.ceil(t_f / dt))

# # Initial Conditions and Variables Storage Arrays
# v_0 = 0
# w_0 = 0.1
# t_0 = 0
# v_val = np.zeros(n + 1)
# w_val = np.zeros(n + 1)
# t_val = np.zeros(n + 1)
# v_val[0] = v_0
# w_val[0] = w_0
# t_val[0] = t_0

# intersect = []

def v_diff(version, v, w,  *args):
    if version == 0:
        mu, alpha = args
        return mu * v * (1 - v) * (v - alpha) - v * w 
    if version == 4:
        alpha = args
        return v * (1 - v) * (v - alpha) - w

def w_diff(version, v, w, *args):
    if version == 0:
        eps, beta = args
        return eps * (v * (beta - v) - w)
    if version == 4:
        eps, beta, gamma, delta = args
        return eps * (beta * v - gamma * w - delta)

def step(version, v, w, t, dt, v_param, w_param, *args, intersect):
    I = 0
    T, duration, I_stim, t_start = args
    if t % T <= duration and t >= t_start:
        I = I_stim
        intersect.append([t,v,w])
    v += (v_diff(version, v, w, *v_param)+I_stim) * dt
    w += (w_diff(version, v, w, *w_param)+I_stim) * dt
    t += dt
    return v, w, round(t,1)

def run(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param, intersect):
    intersect = []
    v_val = np.zeros(n + 1)
    w_val = np.zeros(n + 1)
    t_val = np.zeros(n + 1)
    v_val[0] = v_0
    w_val[0] = w_0
    t_val[0] = t_0
    for i in range(n):
        v_0, w_0, t_0 = step(version, v_0, w_0, t_0, dt, v_param, w_param, I_param, intersect)
        v_val[i+1] = v_0
        w_val[i+1] = w_0
        t_val[i+1] = t_0
    return v_val, w_val, t_val