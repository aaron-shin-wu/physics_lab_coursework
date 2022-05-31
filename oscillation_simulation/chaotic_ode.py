#Problem 5 
###################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#set the constants
gamma = 1.084
w = 2*np.pi
w0 = 1.5*w
b = w0/4

def ode_ddp(y,t):
  dy1dt = y[1]
  dy2dt = gamma*(w0**2)*np.cos(w*t) - 2*b*y[1] - (w0**2)*np.sin(y[0])
  dYdt = [dy1dt, dy2dt]
  return dYdt

# create array of time from 0 to 20 with 2000 elements
t = np.linspace(0,50,2001)

#initializing for phi 1
y1_1_ini = 0
y2_1_ini = 0
Y_1_ini = [y1_1_ini, y2_1_ini]

#initializing for phi 2
y1_2_ini = 0.00001
y2_2_ini = 0
Y_2_ini = [y1_2_ini, y2_2_ini]

#solve ODE system
solve_ode_1 = odeint(ode_ddp, Y_1_ini, t)
y1_1_soln = solve_ode_1[:, 0]
y2_1_soln = solve_ode_1[:, 1]

solve_ode_2 = odeint(ode_ddp, Y_2_ini, t)
y1_2_soln = solve_ode_2[:, 0]
y2_2_soln = solve_ode_2[:, 1]

fig = plt.figure(figsize =(18,5))
grid = plt.GridSpec(1,2,wspace = 0.25, hspace = 0.0)

dy = np.abs(y1_1_soln - y1_2_soln)
log_dy = np.log(dy)


plt.subplot(grid[0,0])
plt.xlabel('Time(s)', fontsize = 14)
plt.ylabel('Amplitude (rad)', fontsize = 14)
plt.plot(t, y1_1_soln, label = r"$\phi_1(0) = 0$")
plt.plot(t, y1_2_soln, label = r"$\phi_2(0) = 0.00001$")
plt.grid()
plt.title(r"Problem 5: $\phi_1$ and $\phi_2$", fontsize = 16)
plt.legend(loc = 1)
plt.xlim([0,7])
