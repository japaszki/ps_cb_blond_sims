
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import division
from __future__ import print_function
from builtins import str
import numpy as np
import pylab as plt
import pickle
import os
from run_cb_sim import run_cb_sim
import cavity_model
from scipy.constants import m_p ,c, e


class sim_params:
    pass
        

#Define static parameters:
params = sim_params()

# Tracking details
params.N_t = 10000     # Number of turns to track

# Beam parameters
params.n_particles = 1e10
params.n_macroparticles = 1e3
params.sync_momentum = 15e9 # [eV]
                        
# Machine and RF parameters
radius = 100.0
params.gamma_transition = 6.1
params.circumference = 2 * np.pi * radius  # [m]

# Cavities parameters
params.n_rf_systems = 1
params.harmonic_number = 21
params.voltage_program = 168e3
params.phi_offset = 0

#Wake impedance
params.wake_R_S = 1
params.wake_Q = 100

#Beam parameters:
params.n_bunches = 21
params.bunch_spacing_buckets = 1
params.bunch_length = 4*2/c
params.intensity_list = [84*2.6e11/params.n_bunches] * params.n_bunches
params.minimum_n_macroparticles = [1e4] * params.n_bunches

params.cbfb_params = {'N_channels' : 1,
                      'h_in' : [20],
                      'h_out' : [1],
                      'active' : [False],
                      'sideband_swap' : [True],
                      'gain' : [np.zeros(params.N_t+1, complex)]}

params.cbfb_params['gain'][0][:] = 1e-3 * np.exp(2j * np.pi * 0.26)

finemet_dt = 5e-9
finemet_f0 = 1.96e6
finemet_Q = 0.49
# finemet_h = cavity_model.resonator_impulse_response(2*np.pi*finemet_f0, finemet_Q, finemet_dt, 100)

params.rf_params = {'dt' : finemet_dt, 
                    'impulse_response' : np.ones(1), 
                    'max_voltage' : 1e5, 
                    'output_delay' : 1e-8,
                    'history_length' : 1e-6}

params.start_cbfb_turn = 15000
params.end_cbfb_turn = 20000
params.cbfb_active_mask = [True] #Only these channels get activated on SCBFB

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 200
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.exc_v[:] = 1.5e3
params.fs_exc = 387.29

#Simulation parameters
params.profile_plot_bunch = 0
params.phase_plot_dt = 200
params.phase_plot_max_dE = 100e6
params.tomo_n_slices = 3000
params.tomo_dt = 10
params.fft_n_slices = 64
params.fft_start_turn = 4000
params.fft_end_turn = 10000
params.fft_plot_harmonics = [20]
params.fft_span_around_harmonic = 2000

params.mode_analysis_window = 4000
params.mode_analysis_resolution = 2000
params.N_plt_modes = 4


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
results_dir = 'output_files/exc_harm_scan/'

exc_harm_runs = [1, 20, 22, 41, 43, 62, 64, 83, 85, 104, 106, 125, 127, 146, 148]
# exc_harm_runs = [2, 19, 23, 40, 44, 61, 65, 83, 86, 103, 107, 124, 128, 145, 149]
N_runs = len(exc_harm_runs)

#Dipole excitation at given harmonic:
for run in range(N_runs):
    subdir = 'dipole_exc_h' + str(exc_harm_runs[run])
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = params.fs_exc
    params.exc_harmonic = exc_harm_runs[run]
    
    [dipole_usb_mag, dipole_lsb_mag, quad_usb_mag, quad_lsb_mag,\
     pos_mode_amp, width_mode_amp] = run_cb_sim(params)
        
    cb_data = {'params' : params,
        'dipole_usb_mag' : dipole_usb_mag,
        'dipole_lsb_mag' : dipole_lsb_mag,
        'quad_usb_mag' : quad_usb_mag,
        'quad_lsb_mag' : quad_lsb_mag,
        'pos_mode_amp' : pos_mode_amp,
        'width_mode_amp' : width_mode_amp}
     
    with open(this_directory + results_dir + 'dipole_run_' + str(run) + '.pickle', 'wb') as f:
        pickle.dump(cb_data, f)
        
    print('Dipole run '  + str(run) + ' finished.')

#Quad excitation at given harmonic:
for run in range(N_runs):        
    subdir = 'quad_exc_h_' + str(exc_harm_runs[run])   
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = 2*params.fs_exc
    params.exc_harmonic = exc_harm_runs[run]
    
    [dipole_usb_mag, dipole_lsb_mag, quad_usb_mag, quad_lsb_mag,\
     pos_mode_amp, width_mode_amp] = run_cb_sim(params)
        
    cb_data = {'params' : params,
        'dipole_usb_mag' : dipole_usb_mag,
        'dipole_lsb_mag' : dipole_lsb_mag,
        'quad_usb_mag' : quad_usb_mag,
        'quad_lsb_mag' : quad_lsb_mag,
        'pos_mode_amp' : pos_mode_amp,
        'width_mode_amp' : width_mode_amp}

    with open(this_directory + results_dir + 'quad_run_' + str(run) + '.pickle', 'wb') as f:
        pickle.dump(cb_data, f)

    print('Quad run '  + str(run) + ' finished.')
    

#Read, post-process, and plot data:
dipole_exc_dipole_sideband_mag = np.empty(N_runs)
dipole_exc_quad_sideband_mag = np.empty(N_runs)
dipole_exc_pos_amp = np.empty(N_runs)
dipole_exc_width_amp =  np.empty(N_runs)

quad_exc_dipole_sideband_mag = np.empty(N_runs)
quad_exc_quad_sideband_mag = np.empty(N_runs)
quad_exc_pos_amp = np.empty(N_runs)
quad_exc_width_amp =  np.empty(N_runs)

for run in range(N_runs):
    with open(this_directory + results_dir + 'dipole_run_' + str(run) + '.pickle', 'rb') as f:
        dipole_data = pickle.load(f)    
    
    dipole_exc_dipole_sideband_mag[run] = (dipole_data['dipole_usb_mag'][20] + \
                                           dipole_data['dipole_lsb_mag'][20]) / 2
    dipole_exc_quad_sideband_mag[run] = (dipole_data['quad_usb_mag'][20] + \
                                         dipole_data['quad_lsb_mag'][20]) / 2
    dipole_exc_pos_amp[run] = (dipole_data['pos_mode_amp'][1] + dipole_data['pos_mode_amp'][20]) / 2
    dipole_exc_width_amp[run] = (dipole_data['width_mode_amp'][1] + dipole_data['width_mode_amp'][20]) / 2
    
    with open(this_directory + results_dir + 'quad_run_' + str(run) + '.pickle', 'rb') as f:
        quad_data = pickle.load(f)
        
    quad_exc_dipole_sideband_mag[run] = (quad_data['dipole_usb_mag'][20] + \
                                           quad_data['dipole_lsb_mag'][20]) / 2
    quad_exc_quad_sideband_mag[run] = (quad_data['quad_usb_mag'][20] + \
                                         quad_data['quad_lsb_mag'][20]) / 2
    quad_exc_pos_amp[run] = (quad_data['pos_mode_amp'][1] + quad_data['pos_mode_amp'][20]) / 2
    quad_exc_width_amp[run] = (quad_data['width_mode_amp'][1] + quad_data['width_mode_amp'][20]) / 2


plt.figure('exc_harm_vs_sideband')
plt.plot(exc_harm_runs, dipole_exc_dipole_sideband_mag, label = 'Dipole Exc, Dipole Sideband')
plt.plot(exc_harm_runs, dipole_exc_quad_sideband_mag, label = 'Dipole Exc, Quad Sideband')
plt.plot(exc_harm_runs, quad_exc_dipole_sideband_mag, label = 'Quad Exc, Dipole Sideband')
plt.plot(exc_harm_runs, quad_exc_quad_sideband_mag, label = 'Quad Exc, Quad Sideband')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Excitation harmonic')
plt.ylabel('Sideband magnitude [arb. units.]')
plt.savefig(this_directory + results_dir + 'exc_harm_vs_sideband.png')

plt.figure('exc_harm_vs_osc_amp')
plt.plot(exc_harm_runs, dipole_exc_pos_amp, label = 'Dipole Exc, Position Osc')
plt.plot(exc_harm_runs, dipole_exc_width_amp, label = 'Dipole Exc, Width Osc')
plt.plot(exc_harm_runs, quad_exc_pos_amp, label = 'Quad Exc, Position Osc')
plt.plot(exc_harm_runs, quad_exc_width_amp, label = 'Quad Exc, Width Osc')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Excitation harmonic')
plt.ylabel('Oscillation amplitude [s]')
plt.savefig(this_directory + results_dir + 'exc_harm_vs_osc_amp.png')