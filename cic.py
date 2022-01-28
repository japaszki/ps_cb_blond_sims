# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:49:54 2022

@author: JohnG
"""

import numpy as np
from blond.utils import bmath as bm
            
class mov_avg_decim:
    def __init__(self, num_stages, ratio, comb_length):
        assert num_stages >= 1, "Number of stages must be at least 1."
        assert isinstance(num_stages, int), "Number of stages must be an integer."
        
        assert ratio >= 1, "Downsampling ratio must be at least 1"
        assert isinstance(ratio, int), "Downsampling ratio must be an integer."
        
        assert comb_length >= 1, "Comb length must be at least 1"
        assert isinstance(comb_length, int), "Comb length must be an integer."
        
        self.num_stages = num_stages
        self.ratio = ratio
        self.comb_length = comb_length
        
        self.window_length = comb_length * ratio
        self.window_values = np.zeros([num_stages, self.window_length])
        self.curr_output = 0
        
        
    def update(self, data_in):
        # assert np.size(data_in) == self.ratio, "Incorrect input vector size."
        
        for j in range(self.ratio):   
            stage_input = data_in[j]       
            for i in range(self.num_stages):
                self.window_values[i, 1:] = self.window_values[i, 0:-1] #Move window
                self.window_values[i, 0] = stage_input #Add new data to window
                stage_input = bm.mean(self.window_values[i,:])  #input for next stage
        
        self.curr_output = stage_input
        return self.curr_output
        
class mov_avg_interp:
    def __init__(self, num_stages, ratio, comb_length):
        assert num_stages >= 1, "Number of stages must be at least 1."
        assert isinstance(num_stages, int), "Number of stages must be an integer."
        
        assert ratio >= 1, "Downsampling ratio must be at least 1"
        assert isinstance(ratio, int), "Downsampling ratio must be an integer."
        
        assert comb_length >= 1, "Comb length must be at least 1"
        assert isinstance(comb_length, int), "Comb length must be an integer."
        
        self.num_stages = num_stages
        self.ratio = ratio
        self.comb_length = comb_length
        
        self.window_length = comb_length * ratio
        self.window_values = np.zeros([num_stages, self.window_length])
        self.curr_output = np.zeros(ratio)
        
        
    def update(self, data_in):
        # assert np.isscalar(data_in), "Incorrect input vector size."
        
        for j in range(self.ratio):    
            stage_input = data_in
            for i in range(self.num_stages):
                self.window_values[i, 1:] = self.window_values[i, 0:-1] #Move window
                self.window_values[i, 0] = stage_input #Add new data to window
                stage_input = bm.mean(self.window_values[i,:])  #input for next stage
        
            self.curr_output[j] = stage_input
        return self.curr_output