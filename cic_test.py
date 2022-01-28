# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:59:50 2022

@author: JohnG
"""

import cic
import numpy as np
import pylab as plt


N = 2
R = 64
M = 4

N_segments = 50

input_vec = np.zeros(R * N_segments)
input_vec[200:1800] = 1

input_vec_split = np.array_split(input_vec, N_segments)
baseband_vec = np.empty(N_segments)
interp_vec = []

decim = cic.mov_avg_decim(N,R,M)
interp = cic.mov_avg_interp(N,R,M)

for i in range(N_segments):
    baseband_vec[i] = decim.update(input_vec_split[i])
    interp_out = interp.update(baseband_vec[i])
    interp_vec.extend(interp_out)
    
baseband_vec_repeat = np.repeat(baseband_vec, R)
    
plt.plot(input_vec)
plt.plot(baseband_vec_repeat)
plt.plot(interp_vec)
plt.show()