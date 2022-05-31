# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:10:27 2021

@author: aaron
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#REDEFINE WHERE THE PEAKS ARE!!
blue_peak = 1.031  
red_peak = 3.104  

#setting up the data file
data_dir = 'C:/Users/aaron/Documents/Spring 21/Phys 128AL/Spectrometer Lab/Photos/data/5-26/'
file = 'cfl_data.csv'
filename = data_dir + file
df = pd.read_csv(filename)
result = df.head(10)
distances = df['Distance_(inches)'].tolist()
brightness = df['Gray_Value'].tolist()

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

#Plotting to check the result
plt.plot(wavelengths, brightness)
plt.xlabel('Wavelength (nm)', fontsize = 14)
plt.ylabel('Grayscale Brightness', fontsize = 14)
plt.title('Calibrated CFL Spectrum', fontsize = 14)
plt.grid()


