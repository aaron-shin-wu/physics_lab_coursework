#Problem 3 - ODE
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#set the constants
gamma = 0.9
w = 2*np.pi
w0 = 1.5*w
b = w0/4

def ode_ddp(y,t):
  dy1dt = y[1]
  dy2dt = gamma*(w0**2)*np.cos(w*t) - 2*b*y[1] - (w0**2)*np.sin(y[0])
  dYdt = [dy1dt, dy2dt]
  return dYdt

# create array of time from 0 to 20 with 2000 elements
t = np.linspace(0,6,2001)
y1_ini = 0
y2_ini = 0
Y_ini = [y1_ini, y2_ini]

#initializing for b1
y1_b1_ini = (1/2)*np.pi
y2_b1_ini = 0
Y_b1_ini = [y1_b1_ini, y2_b1_ini]

#initializing for b2
y1_b2_ini = -(1/2)*np.pi
y2_b2_ini = 0
Y_b2_ini = [y1_b2_ini, y2_b2_ini]

#solve ODE system
solve_ode = odeint(ode_ddp, Y_ini, t)
y1_soln = solve_ode[:,0]
y2_soln = solve_ode[:,1]

solve_ode_b1 = odeint(ode_ddp, Y_b1_ini, t)
y1_b1_soln = solve_ode_b1[:, 0]
y2_b1_soln = solve_ode_b1[:, 1]

solve_ode_b2 = odeint(ode_ddp, Y_b2_ini, t)
y1_b2_soln = solve_ode_b2[:, 0]
y2_b2_soln = solve_ode_b2[:, 1]

fig = plt.figure(figsize =(20,4))
grid = plt.GridSpec(1,2,wspace = 0.25, hspace = 0.0)

plt.subplot(grid[0,0])
plt.xlabel(r"Time (s)", fontsize = 14)
plt.ylabel(r"$\phi$", fontsize = 14)
plt.plot(t, y1_soln, label = r"$\phi (0) = 0$")
#plt.plot(t, y1_b1_soln, label = r"$\phi (0) = \frac{\pi}{2}$")
#plt.plot(t, y1_b2_soln, label = r"r$\phi (0) = - \frac{\pi}{2}$")
plt.grid()
plt.title('Problem 3 part a: DDP solution', fontsize = 16)
#plt.legend(loc=1)


