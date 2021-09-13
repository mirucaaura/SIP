import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import optimize

def f(x):
    """
    x: 3-dim numpy array
    """
    return x[2]

def z(x, t):
    """
    x: 3-dim numpy array
    t: scalar
    z: R^3 \to R^2
    """
    ret = np.zeros(2)
    ret[0] = x[0] + x[2] * np.cos(t)
    ret[1] = x[1] + x[2] * np.sin(t)
    return ret

def c1(z):
    """
    z: 2-dim numpy array
    """
    return z[0] ** 2 - z[1]

def c2(z):
    """
    z: 2-dim numpy array
    """
#     return z[0] ** 2 + 0.5 * z[1] - 1
    return z[0] ** 2 + z[1] - np.sqrt(2)

def g1(x, t):
    """
    x: 3-dim numpy array
    t: scalar
    return: scalar
    """
    ret = pow(x[0] + x[2] * np.cos(t), 2) - (x[1] + x[2] * np.sin(t))
    return ret

def g2(x, t):
    """
    x: 3-dim numpy array
    t: scalar
    return: scalar
    """
#     ret = pow(x[0] + x[2] * np.cos(t), 2) + 0.5 * (x[1] + x[2] * np.sin(t)) - 1
    ret = pow(x[0] + x[2] * np.cos(t), 2) + (x[1] + x[2] * np.sin(t)) - np.sqrt(2)
    return ret

def f_eps(x, eps):
    """
    x: 3-dim numpy array
    eps: scalar
    """
    ret = -f(x) + 0.5 * eps * cp.norm(x) ** 2
    return ret

def gr1(t, y, gamma):
    """
    y: 3-dim numpy array
    t: scalar
    gamma: scalar
    return: scalar
    """
    return -(g1(y, t) + gamma)

def gr2(t, y, gamma):
    """
    y: 3-dim numpy array
    t: scalar
    gamma: scalar
    return: scalar
    """
    return -(g2(y, t) + gamma)

def solve_sub(obj, constraints, gamma, eps, delta):
    """
    solve RCSIP(T, gamma_k, eps_k)
    """
    # set sub-problem
    obj = f_eps(y, eps) # objective function
    constraints = [] # constraints list
    for t in T1:
        constraints += [g1(y, t) <= -gamma]
    for t in T2:
        constraints += [g2(y, t) <= -gamma]
    #--- Step1-1: solve sub-problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    yr = y.value
    return yr

def find_t(yr, gr, gamma, delta):
    """
    find t
    s.t. gr(yr, t) + gamma_k > delta_k
    """
    div = 50
    t = np.linspace(0, 2*np.pi, div)
    y = gr(t, yr, gamma)
    x0 = (2 * np.pi / div) * (np.argmin(y) + 1)
    bounds = ((0, 2*np.pi),)
    res = optimize.minimize(gr, x0=x0, bounds=bounds, args=(yr, gamma,), method="Nelder-Mead", tol=1e-6)
    res_fun = -res.fun # caution
    res_t = res.x
    return (res_fun, res_t)

# set problem
y = cp.Variable((3,), pos=True)
# initial index set
T1 = [0, np.pi, 2 * np.pi] # initial index set
T2 = [0, np.pi, 2 * np.pi] # initial index set

# parameter
sigma = 1e-2
eps = 1
    
itemax = 100
ite_inner_max = 10

# main loop
for k in range(1, itemax + 1):
    # parameters
    eps *= 3 / 4
    gamma = 1 / (k + 5)
    delta = 101 / (100 * pow(k, 3))
    #--- inner loop
    for r in range(ite_inner_max):
        # set sub-problem
        obj = f_eps(y, eps) # objective function
        constraints = [] # constraints list
        for t in T1:
            constraints += [g1(y, t) <= -gamma]
        for t in T2:
            constraints += [g2(y, t) <= -gamma]
        #--- Step1-1: solve sub-problem
        yr = solve_sub(obj, constraints, gamma, eps, delta)
        #--- Step1-2: search tr \in T
        res1_fun, res1_t = find_t(yr, gr1, gamma, delta)
        res2_fun, res2_t = find_t(yr, gr2, gamma, delta)
        # criteria
        if res1_fun <= delta and res2_fun <= delta:
            inner = r
            break
        # add index of T
        elif res1_fun > delta:
            T1.append(res1_t)
        elif res2_fun > delta:
            T2.append(res2_t)
    if k<=10: print(k, yr, len(T1), len(T2), inner, gamma >= delta)
    #--- step2
    if max(gamma, delta, eps) < sigma:
        print("terminate", k, "times")
        break
print("the optimal value is", yr)

# plot
z1 = np.linspace(-1/np.sqrt(np.sqrt(2)), 1/np.sqrt(np.sqrt(2)), 100)
f1 = pow(z1, 2)
f2 = -pow(z1, 2) + np.sqrt(2)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
plt.xlim([-1, 1])
plt.ylim([-0.2, 1.8])
plt.grid(True)
plt.plot(z1, f1, color="k")
plt.plot(z1, f2, color="k")
plt.fill_between(z1, f1, f2, facecolor="r",alpha=0.5)
# x = np.array([0., 0.70709686, 0.66874631])
x = yr
c = patches.Circle(xy=(x[0], x[1]), radius=x[2], color="b", alpha=0.6, fill=True)
ax.add_patch(c)
ax.set_axisbelow(True)
# plt.savefig("ok.jpg")
plt.show()