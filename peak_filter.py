# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:58:37 2022

@author: JohnG
"""

import numpy as np
    
class peak_filter:
    def __init__(self, N_window):
        self.window_values = np.zeros(N_window)
        self.curr_output = 0
    
    def update(self, data_in):
        self.window_values[1:] = self.window_values[0:-1] #Move window
        self.window_values[0] = np.max(data_in) #Add new data to window
        self.curr_output = np.max(self.window_values)
        return self.curr_output