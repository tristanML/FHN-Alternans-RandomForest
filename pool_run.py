from numerical_integrator import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

compound_v_vals = []
compound_w_vals = []
compound_t_vals = []
compound_apd_vals = []
alt_list = []
avs_list = []
avg_v_list = []
avg_w_list = []
apd_index_list = []

# T_vals = []
# all_apd_vals = []

version = 0
mu = 1
alpha = 0.2
beta = 1.09
lower_beta = 1.04
mid_beta = 1.0889
max_beta = 1.119
eps = 0.04
v_param = [mu, alpha]
w_param = [eps, beta]

I_stim = 0.5
T = 230
duration = 1
t_start = 0
n_T = 500
specific_T = 600

dt = 1
t_f = T*n_T
n = int(np.ceil(t_f / dt))

v_0 = 0
w_0 = 0.1
t_0 = 0

colors = ["r", "g", "b", "orange", "purple", "violet", "r", "g", "b", "orange", "purple", "violet",]
grad_red = [1,0,0]
grad_blue = [0,0,1]

percent_value = 7/8

if __name__ == '__main__':
#
    params = []
    for en in np.arange(0.0001,0.02,0.0001):
        for Tn in range(specific_T, specific_T+10, 10):
            v_param = [mu, alpha]
            w_param = [en, beta]
            params.append([version, v_0, w_0, t_0, dt, n, v_param, w_param, [Tn, duration, I_stim, t_start], percent_value])
    params = []
    for en in np.arange(0.01,0.02,0.005):
        for Tn in range(specific_T, specific_T+10, 10):
            v_param = [mu, alpha]
            w_param = [en, beta]
            params.append([version, v_0, w_0, t_0, dt, n, v_param, w_param, [Tn, duration, I_stim, t_start], percent_value])

    p = Pool()
    result = p.map(run2, params)

    p.close()
    p.join()

    for i in range(len(result)):
        if result[i][0][8][0] == specific_T:
            compound_t_vals.append(result[i][1])
            compound_v_vals.append(result[i][2])
            compound_w_vals.append(result[i][3])
            compound_apd_vals.append(result[i][4])
            avs_list.append(result[i][5])
            alt_list.append(result[i][6])
            avg_v_list.append(result[i][7])
            avg_w_list.append(result[i][8])
            apd_index_list.append(result[i][9])
        # T_vals.append(1/result[i][0][8][0])
        # all_apd_vals.append(result[i][4])

    # Attempt at creating 1/T vs APD graph fails because data is no longer sorted by EPS value
    # This could be solved by sorting the data, but is not necessary for the current purposes
    # fig = plt.figure(figsize=(8, 8))
    # compound_T_vals = T_vals

    # for i in range(len(compound_T_vals)):
    #     print(compound_T_vals[i])
    #     print()
    #     print(all_apd_vals[i])
    #     plt.plot(compound_T_vals[i], all_apd_vals[i], marker='x', linestyle='-', color=colors[2])

    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(8, 8))

    # for i in range(len(compound_t_vals)):
    #     index_start = apd_index_list[i]
    #     index_time_start = index_start*dt

    #     plt.plot((index_time_start, index_time_start), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    #     plt.plot((index_time_start+T, index_time_start+T), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    #     plt.plot((index_time_start+2*T, index_time_start+2*T), (0,1.1), marker='', linestyle='dashed', color=colors[-2])
    #     if alt_list[i]:
    #         plt.plot((index_time_start, t_f), (avg_v_list[i], avg_v_list[i]), marker='', linestyle='dashed', color=colors[3])
    #         plt.plot((index_time_start, t_f), (avg_w_list[i], avg_w_list[i]), marker='', linestyle='dashed', color=colors[3])
    #         plt.plot(compound_t_vals[i][index_start:], compound_v_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
    #         plt.plot(compound_t_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
    #     else:
    #         plt.plot(compound_t_vals[i][index_start:], compound_v_vals[i][index_start:], marker='', linestyle='-', color=colors[2])
    #         plt.plot(compound_t_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[2])

    # plt.grid(True)
    # plt.show()
    
    v_null = np.linspace(min(compound_v_vals[0]), max(compound_v_vals[0]), 100)
    if version == 0:
        w_null_1 = list(v*(beta-v) for v in v_null)
        w_null_2 = list(mu*(1-v)*(v-alpha) for v in v_null)
        # w_null_3 = list(((v*(1 - v) * (v - alpha) + I_stim)/v) for v in v_null)
    # if version == 4:
    #     w_null_1 = list((beta * v - delta)/gamma for v in v_null)
    #     w_null_2 = list(v * (1 - v) * (v - alpha) for v in v_null)

    plt.figure(figsize=(8, 8))
    for i in range(len(compound_t_vals)):
        index_start = apd_index_list[i]
        if alt_list[i]:
            plt.plot((0,avg_v_list[i]), (avg_w_list[i],avg_w_list[i]), marker='', linestyle='dashed', color=tuple(grad_red))
            grad_red[1] += 0.1/255
            # plt.plot(compound_v_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=colors[0])
        else:
            plt.plot(compound_v_vals[i][index_start:], compound_w_vals[i][index_start:], marker='', linestyle='-', color=tuple(grad_blue))
            grad_blue[0] += 0.1/255

    v = v_null
    w = np.linspace(min(w_null_2),max(w_null_1), 100)

    V, W = np.meshgrid(v, w)
    dv = v_diff(version,V,W,*v_param)
    dv2 = v_diff(version,V,W,*v_param)+I_stim
    dw = w_diff(version,V,W,*[0.005,beta])

    # DV, DW = np.meshgrid(dv, dw)

    step = 5

    plt.quiver(V[::step, ::step], W[::step, ::step], dv[::step, ::step], dw[::step, ::step])
    # plt.quiver(V[::step, ::step], W[::step, ::step], dv2[::step, ::step], dw[::step, ::step])
    plt.plot(v_null, w_null_1, marker='', linestyle='-', color='orange')
    plt.plot(v_null, w_null_2, marker='', linestyle='-', color='green')
    # plt.plot(v_null, w_null_3, marker='', linestyle='-', color='purple')
    # k_p = (beta-1)/4
    # plt.plot((min(compound_v_vals[0]), max(compound_v_vals[0])), (k_p, k_p), marker='', linestyle='dashed', color='purple')

    plt.xlabel("v")
    plt.ylabel("w")

    plt.grid(True)
    plt.show()