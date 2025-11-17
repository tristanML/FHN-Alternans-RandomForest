from numerical_integrator import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sklearn
from time import time


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
I_param = [T, duration, I_stim, t_start]

dt = 1
t_f = T*n_T
n = int(np.ceil(t_f / dt))

v_0 = 0
w_0 = 0.1
t_0 = 0

const_params = []



percent_value = 7/8

if __name__ == '__main__':
#
    params = []
    e_range = np.arange(0.0001,0.0220,0.0005)
    T_range = range(50, 651, 10)
    a_range = np.arange(0.01,0.60,0.01)

    T_range = range(30, 210, 25)
    a_range = np.arange(0.01,0.30,0.025)
    e_range = np.arange(0.0001,0.0220,0.0005)

    T_range = range(30, 210, 10)
    a_range = np.arange(0.01,0.30,0.025)
    e_range = np.arange(0.0001,0.0220,0.001)

    # print(len(e_range)*len(T_range))
    # for en in e_range:
    #     for Tn in T_range:
    #         params.append([version, v_0, w_0, t_0, dt, n, v_param, [en,beta], [Tn,duration,I_stim,t_start], percent_value])

    # print(len(e_range)*len(a_range))
    # for en in e_range:
    #     for an in a_range:
    #         params.append([version, v_0, w_0, t_0, dt, n, [mu,an], [en,beta], [T,duration,I_stim,t_start], percent_value])

    # print(len(a_range)*len(T_range))
    # for an in a_range:
    #     for Tn in T_range:
    #         params.append([version, v_0, w_0, t_0, dt, n, [mu,an], [eps,beta], [Tn,duration,I_stim,t_start], percent_value])
    t1 = time()
    print(len(a_range)*len(T_range)*len(e_range))
    for an in a_range:
        for Tn in T_range:
            for en in e_range:
                params.append([version, v_0, w_0, t_0, dt, n, [mu,an], [en,beta], [Tn,duration,I_stim,t_start], percent_value])
    print(time()-t1)
    t1 = time()
    p = Pool()
    result = p.map(run3, params)
    print(time()-t1)

    # 128.50350904464722 with print
    # 131.5244140625 without print

    p.close()
    p.join()

    param_results = [row[:-1] for row in result]
    alt_results = [row[-1] for row in result]

    for k in ["linear", "poly", "rbf", "sigmoid"]:
        clf = sklearn.svm.SVC(kernel=k)
        clf.fit(param_results,alt_results)
        print(k, clf.score(param_results,alt_results))

        # 94.3 % for rbf
        # 94.3 % for poly

        # svc_results = clf.predict(param_results)

        # accuracy = 100 * sum(1 for a, s in zip(alt_results, svc_results) if a == s) / len(alt_results)

        # print(k, accuracy)

    # fig = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection='3d')

    # for i in range(len(param_results)):
    #     if alt_results[i]:
    #         ax.plot3D(param_results[i][0], param_results[i][1], param_results[i][2], marker='o', color='red')
    #     else:
    #         ax.plot3D(param_results[i][0], param_results[i][1], param_results[i][2], marker='o', color='blue')

    # plt.xlabel("Period (T)")
    # plt.ylabel("Epsilon $\epsilon$")
    # plt.xlabel(r"Alpha $\alpha$")
    # plt.ylabel("Epsilon $\epsilon$")
    # plt.ylabel("Period (T)")

    # plt.grid()
    # plt.show()

