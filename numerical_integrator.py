import math
import numpy as np

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

def step(version, v, w, t, dt, v_param, w_param, intersect, *args):
    I = 0
    T, duration, I_stim, t_start = args
    if t % T <= duration and t >= t_start:
        I = I_stim
        intersect.append([t,v,w])
    v += (v_diff(version, v, w, *v_param)+I) * dt
    w += w_diff(version, v, w, *w_param) * dt
    t += dt
    return v, w, round(t,1)

def run(version, v_0, w_0, t_0, dt, n, v_param, w_param, I_param):
    intersect = []
    v_val = np.zeros(n + 1)
    w_val = np.zeros(n + 1)
    t_val = np.zeros(n + 1)
    v_val[0] = v_0
    w_val[0] = w_0
    t_val[0] = t_0
    for i in range(n):
        v_0, w_0, t_0 = step(version, v_0, w_0, t_0, dt, v_param, w_param, intersect, *I_param)
        v_val[i+1] = v_0
        w_val[i+1] = w_0
        t_val[i+1] = t_0
    return v_val, w_val, t_val

def apd_calc(t_val, v_val, w_val, percent_value, T, n, dt):
    threshold = 0.3*max(v_val)
    threshold_intersections = []
    #F inding times when the v trace crosses the threshold value
    for i in range(0, len(v_val)-1):
        if (v_val[i] - threshold)*(v_val[i+1] - threshold) < 0:
            threshold_intersections.append([((v_val[i+1] - threshold)*t_val[i+1] + (threshold - v_val[i])*t_val[i])/(v_val[i+1]-v_val[i]),threshold])
    # Finding the difference between subsequent intersection times, also know as the APD
    apd = []
    for i in range(0, len(threshold_intersections)-1, 2):
        apd.append(threshold_intersections[i+1][0]-threshold_intersections[i][0])

    apd_calc_start = round(percent_value*len(apd))

    # Neither APD will be exactly equal every time it occurs.
    # This finds the averages of the two APDs in the list of apd by averaging the odd and even items
    # If there are no alternans, the two average apds will be the same.
    apd1 = np.mean(list(apd[i] for i in range(apd_calc_start, len(apd), 2)))
    apd2 = np.mean(list(apd[i] for i in range(apd_calc_start+1, len(apd), 2)))
    if apd1 - 5 <= apd2 and apd2 <= apd1 + 5:
        print(T, "No alternans with APD: ", apd1, apd2)
        in_alt = False
    else:
        print(T, "Alternans with APD: ", apd1, apd2)
        in_alt = True
    
    #Finds the average of the two APDS. This would be the APD if the system was not in alternans 
    avg_apd = (apd1+apd2)/2
    
    # n: the amount of steps/iterations, used for general indices
    # percent_value * n: the selection of relevant general indices for v, w, and t vals. This can occur at anytime
    # apd_calc_start = round(percent_value*len(apd)): the selection of relevant APD indices for APD vals.
    #       Since apd[] represents a list of apds, every one of its items represents an elasped period T, thus
    #       apd_time_start = apd_calc_start * T: This finds the timestamp of the first relevant APD index.
    #       From this time, we know the avg_apd will occur at timestamp:
    #       apd_time_start + avg_apd
    # The first relevant APD index will occur at an general index found by:
    # index_start = apd_time_start/dt
    #       This is so because any time t will take n steps of dt to arrive there.
    # From this logic, we know the first relevant avg_apd will occur at general index:
    # avg_apd_index = (apd_time_start + avg_apd)/dt
    apd_time_start = apd_calc_start * T
    index_start = round(apd_time_start/dt)
    avg_apd_index = round((apd_time_start + avg_apd)/dt)

    v_val_slice_1 = v_val[round(apd_time_start/dt):round((apd_time_start+T)/dt)]
    v_max_1 = max(v_val_slice_1)
    v_val_slice_2 = v_val[round((apd_time_start+T)/dt):round((apd_time_start+2*T)/dt)]
    v_max_2 = max(v_val_slice_2)

    w_val_slice_1 = w_val[round((apd_time_start)/dt):round((apd_time_start+apd1)/dt)]
    w_max_1 = min(w_val_slice_1)
    w_val_slice_2 = w_val[round((apd_time_start+T)/dt):round((apd_time_start+T+apd2)/dt)]
    w_max_2 = min(w_val_slice_2)

    avg_v = (v_max_1 + v_max_2)/2
    avg_w = (w_max_1 + w_max_2)/2

    # print(n, percent_value, len(apd), apd_calc_start, index_start, avg_apd_index, v_val[avg_apd_index], w_val[avg_apd_index])
    # avg_v = v_val[avg_apd_index]
    # avg_w = w_val[avg_apd_index]
    # for i in range(index_start, len(v_val)-1):
    #     if (t_val[i] - avg_apd_time_stamp)*(t_val[i+1] - avg_apd_time_stamp) < 0:
    #         print(t_val[i], t_val[i+1])
    #         avg_v = ((t_val[i+1] - avg_apd_time_stamp)*v_val[i+1] + (avg_apd_time_stamp - t_val[i])*v_val[i])/(t_val[i+1]-t_val[i])
    #         avg_w = ((t_val[i+1] - avg_apd_time_stamp)*w_val[i+1] + (avg_apd_time_stamp - t_val[i])*w_val[i])/(t_val[i+1]-t_val[i])
    #         print(max(v_val), max(w_val), avg_v, avg_w)
    #         break
    
    return sorted([apd1, apd2]), avg_apd, in_alt, avg_v, avg_w, index_start