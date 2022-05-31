# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:33:59 2021

@author: aaron
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

#REDEFINE WHERE THE PEAKS ARE!!
blue_peak = 0.308
red_peak = 2.339



#necessary constants for fitting planck's law
h = 6.62607*10**(-34)  # planck's constant
c = 3*10**8 # speed of light
k = 1.38065*10**(-23)  #boltzmann's constant
from scipy.constants import h,k,c
#alpha = 2*h*c**2


def blackbody(lam, T):
    lam2 = 1e-9 * lam
    return  (1/4.23721841*10**13)*((1.92*10**(-16))/ (lam2**5 * (np.exp(0.0143977338997/(T*lam2)) - 1)))



def blackbody_alpha(lam, alpha, T):  #for finding all normalization constants
    lam2 = 1e-9 * lam
    return  (1/alpha)*((1.92*10**(-16))/ (lam2**5 * (np.exp(0.0143977338997/(T*lam2)) - 1)))


def blackbody_fit(lam, T, alpha):
    lam = 1e-9 * lam
    return  alpha/ (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))


def normalize(x, y, T, alpha):
    y_sub = y - np.min(y)   #shift everything down so minimum is at 0
    p = y_sub/np.max(y_sub)   #divide everything by the maximum so that max is 1
    p_min_indx = np.argmin(p) #find index of minimum p value
    p_max_indx = np.argmax(p)  #find index of maximum p value
    y1 = blackbody_alpha(x[p_min_indx], alpha, T)#/10**26
    y2 = blackbody_alpha(x[p_max_indx], alpha,  T)#/10**26
    delta_y = np.abs(y2-y1)
    y_intermediate = p*delta_y + y1
    y_sub2 = y_intermediate - np.min(y_intermediate)
    p2 = y_sub2/y_intermediate
    final_add = np.min(y_intermediate)/np.max(y_intermediate)
    y_final = p2+final_add

    return y_final
    
#setting up the fitting function
def func(x, T, alpha):
    return blackbody_fit(x, T, alpha)



#setting up the CFL data file
data_dir = 'C:/Users/aaron/Documents/Spring 21/Phys 128AL/Spectrometer Lab/Photos/data/5-31/7pm/'
file = 'cfl_data.csv'
filename = data_dir + file
df = pd.read_csv(filename)
result = df.head(10)
distances = df['Distance_(inches)'].tolist()
brightness = np.array(df['Gray_Value'].tolist())

max_dist = max(distances)
min_dist = min(distances)


blue_wave = 436.6 #wavelength in nanometers
red_wave = 611.6 #wavelength in nanometers

interval = (red_wave - blue_wave)/(red_peak - blue_peak)  #nanometers per inch (wrt original plot)

#finding the minimum and maximum wavelengths in the spectrum (extrapolated from blue and red)
min_wave = blue_wave - interval*blue_peak
max_wave = red_wave + interval * (max_dist - red_peak)

#setting up the final wavelengths array
wavelengths = np.zeros(len(distances))

#for every value in the distances array, convert it to the corresponding calibrated wavelength
for i in range(0, len(distances)):
    wavelengths[i] = min_wave + interval * distances[i]

wavelengths = np.array(wavelengths)



#Now setting up the actual sun data 
data_file = 'sky_data.csv'
data_filename = data_dir + data_file
df_sky= pd.read_csv(data_filename)
sky_brightness = np.array(df_sky['Gray_Value'].tolist())


#Now setting up the actual sun data 
data_file = 'sky_data.csv'
data_filename = data_dir + data_file
df_sky= pd.read_csv(data_filename)
sky_brightness = np.array(df_sky['Gray_Value'].tolist())

#Trying different temperatures for the planck's law curve, see which one works
# best for the data set. 


a_o = 2.105473863*10**11
temperature = np.linspace(2000, 40000, 153)
wavelengths_test = np.linspace(100, 10000, 10000)

alpha = a_o
alpha_list = np.zeros(len(temperature))
for i in range(0, len(temperature)):
    max_amp = np.max(blackbody_alpha(wavelengths_test, alpha, temperature[i]))
    alpha = alpha * max_amp
    alpha_list[i] = alpha


Temperatures = []
ss_res_vals = []
for t in range(0, len(temperature)):
    print('Temperature: ', t)
    normalized_brightness = normalize(wavelengths, sky_brightness, temperature[t], alpha_list[t])
    
    
    popt, pcov = curve_fit(func, wavelengths, normalized_brightness, p0 = (temperature[t], 1/alpha_list[t]))
    
    residuals = normalized_brightness - blackbody_alpha(wavelengths, alpha_list[t], temperature[t])
    #residuals = normalized_brightness - blackbody_fit(wavelengths, *popt)
    ss_res = np.sum(residuals**2)
    Temperatures.append(temperature[t])
    ss_res_vals.append(ss_res)

Temperatures = np.array(Temperatures)
ss_res_vals = np.array(ss_res_vals)


best_temp = Temperatures[np.argmin(ss_res_vals)]
best_alpha= alpha_list[np.argmin(ss_res_vals)]
normalized_data = normalize(wavelengths, sky_brightness, best_temp, best_alpha)


sky_left = np.zeros(len(wavelengths))
sky_right = np.zeros(len(wavelengths))

#making sky_left
for n in range(0, len(wavelengths)):
    
    if n < len(wavelengths) - 15:
        sky_left[n] = sky_brightness[n+15]  
    else:
        sky_left[n] = sky_brightness[len(wavelengths)-1]

#making sky_right
for n in range(0, len(wavelengths)):
    
    if n > 15:
        sky_right[n] = sky_brightness[n-15]  
    else:
        sky_right[n] = sky_brightness[0]
        
plt.plot(wavelengths, sky_brightness)
plt.plot(wavelengths, sky_left)
plt.plot(wavelengths, sky_right)
