from numerical_integrator import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

version = 0
mu = 1
alpha = 0.2
beta = 1.09
#[0.005,0.007]
eps = 0.007
v_param = [mu, alpha]
w_param = [eps, beta]
q = 10

v = np.linspace(0,1, q)
w = np.linspace(0,1, q)

V, W = np.meshgrid(v, w)

def f(x,y):
    return x

def f2(x,y):
    return y

dv = f(V,W)
dw = f2(V,W)

print(V.shape, W.shape, dv.shape, dw.shape)

# if version == 0:
#     w_null_1 = list(v*(beta-v) for v in v)
#     w_null_2 = list(mu*(1-v)*(v-alpha) for v in v)

# plt.plot(v, w_null_1, marker='', linestyle='-', color='orange')
# plt.plot(v, w_null_2, marker='', linestyle='-', color='green')

step = 5
# print(len(V[::step,::step]),
#       len(W[::step,::step]),
#       len(DV[::step,::step]),
#       len(DW[::step,::step]))
plt.quiver(V[::step, ::step], W[::step, ::step], dv[::step, ::step], dw[::step, ::step])
# plt.quiver(V, W, DV, DW, units='x')

plt.title('Quiver Plot Example')
plt.show()