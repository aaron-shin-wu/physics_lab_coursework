# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:58:57 2021

@author: aaron
"""

# fitting to Planck's law

import matplotlib.pyplot as plt
import numpy as np
from scipy import curve_fit

h = 0  # planck's constant
c = 3*10**8 # speed of light
k = 0  #boltzmann's constant

#planck's law function
def plancks_law(x, T):
    return ((2*h*c**2)/x**5)(1/(np.exp((h*c)/(x*k*T))-1))

