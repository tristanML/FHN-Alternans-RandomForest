from numerical_integrator import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sklearn


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
    for en in np.arange(0.0001,0.02,0.001):
        for Tn in range(100, 600, 50):
            params.append([version, v_0, w_0, t_0, dt, n, v_param, [en,beta], [Tn,duration,I_stim,t_start], percent_value])

    p = Pool()
    result = p.map(run3, params)


    p.close()
    p.join()
    print(result)

    param_results = [[row[2],row[4]] for row in result]
    alt_results = [row[-1] for row in result]
    print(param_results[0])
    print(alt_results[0])

    clf = sklearn.svm.SVC()
    clf.fit(param_results,alt_results)

    fig = plt.figure(figsize=(8, 8))

    for i in range(len(param_results)):
        if alt_results[i]:
            plt.plot(param_results[i][0], param_results[i][1], marker='x', color='red')
        else:
            plt.plot(param_results[i][0], param_results[i][1], marker='x', color='blue')

    plt.grid()
    plt.show()

