# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:13:35 2021

@author: aaron
"""

import matplotlib.pyplot as plt
import numpy as np

amps= [9, 27.3, 40.9]

red = [4.55, 13.45, 19.11]
green = [6.04, 15.36, 20.88]
blue = [27.12, 63.69, 79.13]

amp0_tot = red[0] + green[0] + blue[0]
amp1_tot = red[1] + green[1] + blue[1]
amp2_tot = red[2] + green[2] + blue[2]

red_frac = [red[0]/amp0_tot, red[1]/amp1_tot, red[2]/amp2_tot]
green_frac = [green[0]/amp0_tot, green[1]/amp1_tot, green[2]/amp2_tot]
blue_frac = [blue[0]/amp0_tot, blue[1]/amp1_tot, blue[2]/amp2_tot]


plt.plot(amps, red_frac, 'ro', label = 'red')
plt.plot(amps, red_frac, 'r--')
plt.plot(amps, green_frac, 'go', label = 'green')
plt.plot(amps, green_frac, 'g--')
plt.plot(amps, blue_frac, 'bo', label = 'blue')
plt.plot(amps, blue_frac, 'b--')

plt.xlabel('Amps (mA)', fontsize = 14)
plt.ylabel('Relative Intensity (detected)', fontsize = 14)
plt.title('Quantifying Nonlinear RGB Response \n of the Camera', fontsize = 15)