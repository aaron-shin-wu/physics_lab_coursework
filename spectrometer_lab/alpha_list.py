# -*- coding: utf-8 -*-
"""
Created on Sun May 30 17:37:58 2021

@author: aaron
"""
import numpy as np


a_o = 2.105473863*10**11
temperature = np.linspace(2000, 40000, 39)
wavelengths = np.linspace(100, 10000, 1000)

def blackbody(lam, alpha, T):
    lam2 = 1e-9 * lam
    return  (1/alpha)*((1.92*10**(-16))/ (lam2**5 * (np.exp(0.0143977338997/(T*lam2)) - 1)))

alpha = a_o
alpha_list = np.zeros(len(temperature))
for i in range(0, len(temperature)):
    max_amp = np.max(blackbody(wavelengths, alpha, temperature[i]))
    alpha = alpha * max_amp
    alpha_list[i] = alpha
    
print(alpha_list)
    