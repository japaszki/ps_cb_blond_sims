
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
params.N_t = 16000     # Number of turns to track

# Beam parameters
params.n_particles = 1e10
params.n_macroparticles = 1e3
params.sync_momentum = 15e9#25.92e9 # [eV]
                        
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
params.minimum_n_macroparticles = [4e3] * params.n_bunches

params.cbfb_params = {'N_channels' : 1,
                      'h_in' : [20],
                      'h_out' : [20],
                      'active' : [False],
                      'sideband_swap' : [False],
                      'gain' : [np.zeros(params.N_t+1, complex)]}

finemet_dt = 5e-9
finemet_f0 = 1.96e6
finemet_Q = 0.49
finemet_h = cavity_model.resonator_impulse_response(2*np.pi*finemet_f0, finemet_Q, finemet_dt, 100)

params.rf_params = {'dt' : finemet_dt, 
                    'impulse_response' : finemet_h, 
                    'max_voltage' : 1e5, 
                    'output_delay' : 1e-8,
                    'history_length' : 1e-3}

params.start_cbfb_turn = 8000
params.end_cbfb_turn = 12000
params.cbfb_active_mask = [True] #Only these channels get activated on SCBFB

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 200
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.exc_v[0:6000] = 1.5e3
params.fs_exc = 442.07
params.exc_harmonic = 20

#Simulation parameters
params.profile_plot_bunch = 0
params.phase_plot_dt = 200
params.phase_plot_max_dE = 100e6
params.tomo_n_slices = 3000
params.tomo_dt = 10
params.fft_n_slices = 64
params.fft_start_turn = 8000
params.fft_end_turn = 16000
params.fft_plot_harmonics = [20]
params.fft_span_around_harmonic = 2000

params.mode_analysis_window = 4000
params.mode_analysis_resolution = 2000
params.N_plt_modes = 4


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
results_dir = 'output_files/cbfb_gain_phase_scan/'

N_phases = 16
phase_vals = np.linspace(0, 2*np.pi, N_phases)

N_gains = 3
gain_vals = [1e-4, 3e-4, 1e-3]


#Arrange 2D grid of gain and phase values:
[cbfb_gain_2d, cbfb_phase_2d] = np.meshgrid(gain_vals, phase_vals)

#Convert to 1D arrays, with one run of zero gain to use as reference:
cbfb_gain_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_gain_2d)))
cbfb_phase_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_phase_2d)))

N_runs = cbfb_gain_runs.shape[0]

#Dipole excitation:
for run in range(N_runs):
    subdir = 'dipole_exc_cbfb_gain_' + str(cbfb_gain_runs[run]) + '_phase_' + str(cbfb_phase_runs[run])
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = params.fs_exc
    params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
    
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

#Quad excitation:
for run in range(N_runs):        
    subdir = 'quad_exc_cbfb_gain_' + str(cbfb_gain_runs[run]) + '_phase_' + str(cbfb_phase_runs[run])
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = 2*params.fs_exc
    params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(2j * np.pi * cbfb_phase_runs[run])
    
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


#Relative mode amplitude after CBFB on:
plt.figure('dipole_amp_vs_phase')
for i in range(N_gains):
    plot_indices = (cbfb_gain_runs == gain_vals[i])
    plt.semilogy(phase_vals, dipole_exc_pos_amp[plot_indices] /\
             dipole_exc_pos_amp[0], label='Gain = ' + str(gain_vals[i]))
plt.legend(loc=0, fontsize='medium')
plt.xlabel('CBFB phase [rad]')
plt.ylabel('Relative oscillation amplitude')
plt.title('Dipole mode')
plt.savefig(this_directory + results_dir + 'dipole_amp_vs_phase.png')

plt.figure('quad_amp_vs_phase')
for i in range(N_gains):
    plot_indices = (cbfb_gain_runs == gain_vals[i])
    plt.semilogy(phase_vals, quad_exc_width_amp[plot_indices] /\
             quad_exc_width_amp[0], label='Gain = ' + str(gain_vals[i]))
plt.legend(loc=0, fontsize='medium')
plt.xlabel('CBFB phase [rad]')
plt.ylabel('Relative oscillation amplitude')
plt.title('Quadrupole mode')
plt.savefig(this_directory + results_dir + 'quad_amp_vs_phase.png')
