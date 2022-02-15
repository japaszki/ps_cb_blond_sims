# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:03:17 2022

@author: JohnG
"""
import cavity_model
import numpy as np
import pylab as plt


finemet_dt = 5e-9
finemet_f0 = 5e6
finemet_Q = 0.3
finemet_h = cavity_model.resonator_impulse_response(2*np.pi*finemet_f0, finemet_Q, finemet_dt, 20)

plt.figure()
plt.plot(finemet_h)
plt.xlabel('Time [samples]')
plt.ylabel('Impulse response')


rf_params = {'dt' : finemet_dt, 'impulse_response' : finemet_h, 'max_voltage' : 1e3,\
             'history_length' : 1, 'output_delay' : 1e-5}

finemet = cavity_model.cavity_model_fir(rf_params)

N_turns = 100
h_samp = 64
trev = np.logspace(-8, -6, N_turns)

turn_start_time = 0

v_in_full = np.zeros(0)
T_samp_full = np.zeros(0)

for turn in range(N_turns):
    dt_samp = np.linspace(0, trev[turn]*(h_samp-1)/h_samp, h_samp)
    T_samp = turn_start_time + dt_samp
    
    nco_phase = np.linspace(0, 2*np.pi*(h_samp-1)/h_samp, h_samp)
    v_in = 1.0e3 * np.sin(nco_phase)
    
    #Record input vector:
    v_in_full = np.append(v_in_full, v_in)
    T_samp_full = np.append(T_samp_full, T_samp)
    
    finemet.update(v_in, T_samp)
    
    # plt.figure()
    # plt.plot(finemet.input_vec)
    # plt.xlabel('Time [sample]')
    # plt.ylabel('Voltage [V]')
        
    turn_start_time += trev[turn]
    
plt.figure()
plt.plot(T_samp_full, v_in_full, label='input')
plt.plot(finemet.output_hist_t * finemet_dt, finemet.output_hist_v, label='output')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')