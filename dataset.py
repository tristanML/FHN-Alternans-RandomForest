from numerical_integrator import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import accuracy_score, recall_score, f1_score
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
eps = 0.0025
v_param = [mu, alpha]
w_param = [eps, beta]

I_stim = 0.5
T = 230
duration = 1
t_start = 0
n_T = 200
I_param = [T, duration, I_stim, t_start]

dt = 0.5
t_f = T*n_T
n = int(np.ceil(t_f / dt))

v_0 = 0
w_0 = 0.1
t_0 = 0

const_params = []

percent_value = 0.8

if __name__ == '__main__':
#
    params = []
    e_range = np.arange(0.0001,0.0220,0.0005)
    T_range = range(50, 651, 10)
    a_range = np.arange(0.01,0.60,0.01)

    T_range = range(30, 210, 10)
    a_range = np.arange(0.01,0.30,0.025)
    e_range = np.arange(0.0001,0.0220,0.001)
    #4752 samples

    m = 10
    T_range = np.linspace(50, 600, m)
    a_range = np.linspace(0.01,0.5,m)
    e_range = np.linspace(0.001,0.03,m)
    # e_range = [eps]
    #m**3 samples

    # dt = 1, n_T = 500, percent_value 
    # m, m**3, runtime of trials, accuracy, recall
    #15 3375 103.45952105522156 0.8601895734597157 0.3356164383561644
    #12 2197 66.21146988868713 0.889090909090909 0.2328767123287671
    #10 1000 31.66248893737793 0.896 0.38461538461538464
    #5 125 4.874859094619751 0.5625 0.42857142857142855

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
    t0 = time()
    size = len(a_range)*len(T_range)*len(e_range)
    for an in a_range:
        for Tn in T_range:
            for en in e_range:
                params.append([version, v_0, w_0, t_0, dt, n, [mu,an], [en,beta], [Tn,duration,I_stim,t_start], percent_value])

    p = Pool()
    result = p.map(run3, params)
    print(len(result))
    runtime = time() - t0

    p.close()
    p.join()

    param_results = np.array([row[:6] for row in result])
    alt_results = np.array([row[-1] for row in result])

    X_train, X_test, y_train, y_test = train_test_split(param_results, alt_results, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    t0 = time()
    svm_model = SVC(kernel='rbf', C=1000, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    traintime = time() - t0
    print(1/(5*X_train.var()))
    y_pred = svm_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)

    print(m, size, runtime, traintime, accuracy, recall, f1)
    print(T_range[0],T_range[-1],a_range[0],a_range[-1],e_range[0],e_range[-1])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')

    for i in range(len(param_results)):
        if alt_results[i]==1:
            ax.plot3D(param_results[i][1], param_results[i][2], param_results[i][4], marker='o', color='red')
        else:
            ax.plot3D(param_results[i][1], param_results[i][2], param_results[i][4], marker='', color='blue')

    # for i in range(len(X_train)):
    #     if y_train[i]==-1:
    #         ax.plot3D(X_train[i][0], X_train[i][1], X_train[i][2], marker = 'o', color = 'green')

    # for i in range(len(X_test)):
    #     if y_test[i]==-1:
    #         ax.plot3D(X_test[i][0], X_test[i][1], X_test[i][2], marker = 'o', color = 'orange')

    for i in range(len(X_test)):
        if y_pred[i]==1:
            ax.plot3D(X_test[i][1], X_test[i][2], X_test[i][4], marker = 'o', color = 'green')
        else:
            ax.plot3D(X_test[i][1], X_test[i][2], X_test[i][4], marker = '', color = 'blue')
        

        


    plt.grid()
    plt.show()
    # plt.xlabel("Period (T)")
    # plt.ylabel("Epsilon $\epsilon$")
    # plt.xlabel(r"Alpha $\alpha$")
    # plt.ylabel("Epsilon $\epsilon$")
    # plt.ylabel("Period (T)")



