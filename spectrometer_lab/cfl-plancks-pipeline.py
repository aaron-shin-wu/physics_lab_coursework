# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:49:19 2021

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


def blackbody(lam, T, alpha):
    lam = 1e-9 * lam
    return  alpha/ (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))

#setting up the fitting function
def func(x, T, alpha):
    return blackbody(x, T, alpha)


#setting up the CFL data file
data_dir = 'C:/Users/aaron/Documents/Spring 21/Phys 128AL/Spectrometer Lab/Photos/data/5-26/'#9-cloudy-1226/'
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
'''
plt.figure(1)
plt.plot(wavelengths, brightness)
plt.xlabel('Wavelength (nm)', fontsize = 14)
plt.ylabel('Grayscale Brightness', fontsize = 14)
plt.title('Calibrated CFL Spectrum', fontsize = 14)
plt.grid()
'''



#Now setting up the actual sun data 
data_file = 'sky_data.csv'
data_filename = data_dir + data_file
df_sky= pd.read_csv(data_filename)
sky_brightness = np.array(df_sky['Gray_Value'].tolist())

'''
for a in range(0, len(wavelengths)):
    if wavelengths[0] < 425:
        wavelengths = np.delete(wavelengths, 0)
        sky_brightness = np.delete(sky_brightness, 0)
    if wavelengths[0] > 650:
        break
'''

sky_left = np.zeros(len(wavelengths))
sky_right = np.zeros(len(wavelengths))

#making sky_left
for n in range(0, len(wavelengths)):
    
    if n < len(wavelengths) - 20:
        sky_left[n] = sky_brightness[n+20]  
    else:
        sky_left[n] = sky_brightness[len(wavelengths)-1]

#making sky_right
for n in range(0, len(wavelengths)):
    
    if n > 20:
        sky_right[n] = sky_brightness[n-20]  
    else:
        sky_right[n] = sky_brightness[0]
        


fig, ax = plt.subplots()
ax.plot(wavelengths, sky_brightness, label = 'Spectroscopy Data')
ax.fill_between(wavelengths, sky_left, sky_right, alpha = 0.2)
ax.set_title('Sky Spectrum with Calibration', fontsize = 15)
ax.set_xlabel('Wavelength (nm)', fontsize = 14)
ax.set_ylabel('Grayscale Brightness', fontsize = 14)
ax.legend()
