# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:59:50 2022

@author: JohnG
"""

import iir
import numpy as np
import pylab as plt


T_s = 1
T = 10

N_data = 200

input_vec = np.zeros(N_data)
input_vec[20:100] = 1

output_vec = np.empty(N_data)

hpf = iir.iir_hpf(T, T_s)

for i in range(N_data):
    output_vec[i] = hpf.update(input_vec[i])
    
plt.plot(input_vec)
plt.plot(output_vec)
plt.show()