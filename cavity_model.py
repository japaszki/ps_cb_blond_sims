# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:09:29 2022

@author: JohnG
"""

import numpy as np
from blond.utils import bmath as bm

def resonator_impulse_response(omega_0, Q, dt, length):
    p1 = - omega_0 / (2*Q) * (1 + np.lib.scimath.sqrt(1-4*Q**2))
    p2 = - omega_0 / (2*Q) * (1 - np.lib.scimath.sqrt(1-4*Q**2))
    t_vec = np.arange(length * Q /(omega_0 * dt)) * dt
    h = np.real(omega_0/Q * p1 / (p1-p2) * np.exp(p1 * t_vec) + omega_0/Q * p2 / (p2-p1) * np.exp(p2 * t_vec)) * dt
    return h

class cavity_model_fir:
    def __init__(self, rf_params):
        self.dt = rf_params['dt']
        self.impulse_response = rf_params['impulse_response']
        self.max_voltage = rf_params['max_voltage']
        self.history_length_samples = np.ceil(rf_params['history_length'] / self.dt)
        self.output_delay = rf_params['output_delay']
        
        self.output_hist_v = np.zeros(1)
        self.output_hist_t = np.zeros(1, dtype=int)
        self.latest_input_sample = -1
        self.latest_input_voltage = 0
    
    def update(self, v_in, t_in):
        #Note: input sample times assumed to be in ascending order!
        #t_in should be in absolute time from simulation start
        
        #Resample new input data to cavity sample rate:
        new_data_indices = np.round(t_in / self.dt).astype(int)
        
        #Clamp input voltage to allowed limits:
        self.v_in_clamp = v_in
        self.v_in_clamp[np.argwhere(self.v_in_clamp > self.max_voltage)] = self.max_voltage
        self.v_in_clamp[np.argwhere(self.v_in_clamp < -self.max_voltage)] = -self.max_voltage
        
        #Determine length of new data in Finemet samples:
        input_first_sample = self.latest_input_sample + 1
        input_last_sample = new_data_indices[-1]
        self.input_vec = np.zeros(int(input_last_sample - input_first_sample + 1))
        
        #Construct next segment of input vector using zero-order hold:   
        #Extend last input sample from previous turn   
        self.input_vec[0:(new_data_indices[0]-input_first_sample)] = self.latest_input_voltage
        
        for i in range(1, new_data_indices.shape[0]):
            self.input_vec[(new_data_indices[i-1]-input_first_sample+1):(new_data_indices[i]-input_first_sample+1)] = \
                self.v_in_clamp[i-1]
        
        #Update latest sample for next turn:
        self.latest_input_sample = input_last_sample
        self.latest_input_voltage = self.v_in_clamp[-1]
        
        #Convolve new input vector segment with impulse response:
        self.output_vec = bm.convolve(self.input_vec, self.impulse_response)
        
        output_first_sample = input_first_sample
        output_last_sample = input_first_sample + self.output_vec.shape[0] - 1
        
        #Extend output history as needed to fit new data:
        output_new_samples = np.arange(self.output_hist_t[-1] + 1, output_last_sample + 1)
        self.output_hist_t = np.append(self.output_hist_t, output_new_samples)
        self.output_hist_v = np.append(self.output_hist_v, np.zeros_like(output_new_samples))
        
        #Add output vector to output history:
        self.output_hist_v[np.argwhere(self.output_hist_t == output_first_sample)[0][0]:] += self.output_vec
        
        #Cull old points from history vector:
        history_cull_indices = np.argwhere(self.output_hist_t < input_first_sample - self.history_length_samples)
        self.output_hist_t = np.delete(self.output_hist_t, history_cull_indices)
        self.output_hist_v = np.delete(self.output_hist_v, history_cull_indices)
        
    def get_output_in_window(self, bounds):
        #Note: outputs exceed bounds by 1 sample to allowe interpolation over entire span:
        #the dt output vector gives sample times with respect to lower bound.
        
        output_dt = self.output_hist_t * self.dt - self.output_delay - bounds[0]
        output_mask = ~((output_dt < 0) | (output_dt > (bounds[1] - bounds[0])))
        
        #Return zero if no data exists within requested bounds:
        if output_dt[output_mask].shape[0] <= 1:
            dummy_dt = np.zeros(2)
            dummy_dt[1] = bounds[1] - bounds[0]
            return dummy_dt, np.zeros(2)
        else:
            return output_dt[output_mask], self.output_hist_v[output_mask]