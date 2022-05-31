# -*- coding: utf-8 -*-
"""
Created on Sat May 29 15:01:49 2021

@author: aaron
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

#REDEFINE WHERE THE PEAKS ARE!!
blue_peak = 0.302  
red_peak = 2.343 

#necessary constants for fitting planck's law
h = 6.62607*10**(-34)  # planck's constant
c = 3*10**8 # speed of light
k = 1.38065*10**(-23)  #boltzmann's constant
from scipy.constants import h,k,c
#alpha = 2*h*c**2


def blackbody(lam):
    lam2 = 1e-9 * lam
    return  (1/4.23721841*10**13)*((1.92*10**(-16))/ (lam2**5 * (np.exp(0.0000024918/lam2) - 1)))

def blackbody_fit(lam, T, alpha):
    lam = 1e-9 * lam
    return  alpha/ (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))


def normalize(x, y):
    y_sub = y - np.min(y)
    p = y_sub/np.max(y_sub)
    p_min_indx = np.argmin(p)
    p_max_indx = np.argmax(p)
    y1 = blackbody(x[p_min_indx])/10**26
    y2 = blackbody(x[p_max_indx])/10**26
    delta_y = y2-y1
    y_final = p*delta_y + y1
    return y_final
    
#setting up the fitting function
def func(x, T, alpha):
    return blackbody_fit(x, T, alpha)


#setting up the CFL data file
data_dir = 'C:/Users/aaron/Documents/Spring 21/Phys 128AL/Spectrometer Lab/Photos/data/5-29-cloudy-1226/'
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
#Plotting to check the result
plt.figure(1)
plt.plot(wavelengths, brightness)
plt.xlabel('Wavelength (nm)', fontsize = 14)
plt.ylabel('Grayscale Brightness', fontsize = 14)
plt.title('Calibrated CFL Spectrum', fontsize = 14)
plt.grid()


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


normalized_brightness = normalize(wavelengths, sky_brightness)
plt.figure(2)
plt.plot(wavelengths, normalized_brightness)
plt.xlabel('Wavelength (nm)', fontsize = 14)
plt.ylabel('Normalized Brightness', fontsize = 14)
plt.title('Normalized Sky Data', fontsize = 14)
plt.xlim([100, 2000])
#plt.ylim([0, 1.2])
plt.grid()

popt, pcov = curve_fit(func, wavelengths, normalized_brightness, p0 = (5000, 0.000000000000001))
print(popt)
#plt.plot(wavelengths, func(wavelengths, *popt), 'r-')
lamb = np.linspace(100, 2000, 200)
brightness = blackbody_fit(lamb, popt[0], popt[1])
plt.plot(lamb, brightness, 'r--', label = "Fitted Planck's Law Curve")
plt.legend()
