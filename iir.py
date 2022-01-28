# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:58:37 2022

@author: JohnG
"""

import numpy as np

class iir_hpf:
    def __init__(self, relaxation_time, sample_period):
        self.A = np.exp(-sample_period/relaxation_time)
        self.y_lpf = 0
        self.curr_output = 0
        
    def update(self, data_in):
        self.y_lpf = (1-self.A) * data_in + self.A * self.y_lpf
        self.curr_output = data_in - self.y_lpf
        return self.curr_output