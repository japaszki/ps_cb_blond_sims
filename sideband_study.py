
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


class sim_params:
    pass
        

#Define static parameters:
params = sim_params()

# Tracking details
params.N_t = 10000     # Number of turns to track

# Beam parameters
params.n_particles = 1e10
params.n_macroparticles = 1e3
params.sync_momentum = 25.92e9 # [eV]
                        
# Machine and RF parameters
radius = 100.0
params.gamma_transition = 6.1
params.circumference = 2 * np.pi * radius  # [m]

# Cavities parameters
params.n_rf_systems = 1
params.harmonic_number = 21
params.voltage_program = 200e3
params.phi_offset = 0

#Wake impedance
params.wake_R_S = 1
params.wake_Q = 100

#Beam parameters:
params.n_bunches = 21
params.bunch_spacing_buckets = 1
params.bunch_length = 15e-9
params.intensity_list = [1e11] * params.n_bunches
params.minimum_n_macroparticles = [4e3] * params.n_bunches

params.cbfb_N_chans = 1
params.cbfb_h_in = [20]
params.cbfb_h_out = [1]
params.cbfb_active = [False]
params.cbfb_sideband_swap = [True]

params.cbfb_gain_vec = [np.zeros(params.N_t+1, complex)]
params.cbfb_gain_vec[0][:] = 1e-3 * np.exp(2j * np.pi * 0.26)

params.start_cbfb_turn = 15000
params.end_cbfb_turn = 20000
params.cbfb_active_mask = [True] #Only these channels get activated on SCBFB

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 200
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.fs_exc = 387.29
params.exc_harmonic = 20

#Simulation parameters
params.profile_plot_bunch = 0
params.phase_plot_dt = 200
params.phase_plot_max_dE = 100e6
params.tomo_n_slices = 3000
params.tomo_dt = 10
params.fft_n_slices = 256
params.fft_start_turn = 4000
params.fft_end_turn = 10000
params.fft_plot_harmonics = [1, 20, 22, 41, 43, 62, 64, 83, 85, 104, 106, 125, 127]
params.fft_span_around_harmonic = 2000


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
results_dir = 'output_files/sideband_study/'

exc_amp_runs = [5e2, 1e3, 2e3]
N_runs = len(exc_amp_runs)

#Dipole excitation at given harmonic:
for run in range(N_runs):
    subdir = 'dipole_exc_amp_' + str(exc_amp_runs[run])
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = params.fs_exc
    params.exc_v[:] = exc_amp_runs[run]
    
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
    subdir = 'quad_exc_amp_' + str(exc_amp_runs[run])
    params.output_dir = this_directory + results_dir + subdir + '/'
    params.exc_delta_freq = 2*params.fs_exc
    params.exc_v[:] = exc_amp_runs[run]
    
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
dipole_exc_dipole_sideband_mag = [None] * N_runs
dipole_exc_quad_sideband_mag = [None] * N_runs
quad_exc_dipole_sideband_mag = [None] * N_runs
quad_exc_quad_sideband_mag = [None] * N_runs

for run in range(N_runs):
    with open(this_directory + results_dir + 'dipole_run_' + str(run) + '.pickle', 'rb') as f:
        dipole_data = pickle.load(f)    
    
    dipole_exc_dipole_sideband_mag[run] = (np.array(dipole_data['dipole_usb_mag']) + \
                                           np.array(dipole_data['dipole_lsb_mag'])) / 2
    dipole_exc_quad_sideband_mag[run] = (np.array(dipole_data['quad_usb_mag']) + \
                                         np.array(dipole_data['quad_lsb_mag'])) / 2
  
    with open(this_directory + results_dir + 'quad_run_' + str(run) + '.pickle', 'rb') as f:
        quad_data = pickle.load(f)
        
    quad_exc_dipole_sideband_mag[run] = (np.array(quad_data['dipole_usb_mag']) + \
                                           np.array(quad_data['dipole_lsb_mag'])) / 2
    quad_exc_quad_sideband_mag[run] = (np.array(quad_data['quad_usb_mag']) + \
                                         np.array(quad_data['quad_lsb_mag'])) / 2

plt.figure('dipole_sideband_spectrum')
for run in range(N_runs):
    plt.plot(params.fft_plot_harmonics, dipole_exc_dipole_sideband_mag[run][params.fft_plot_harmonics], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('Sideband magnitude [arb. units.]')
plt.title('Dipole mode')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/dipole_sideband_spectrum.png')

plt.figure('quad_sideband_spectrum')
for run in range(N_runs):
    plt.plot(params.fft_plot_harmonics, quad_exc_quad_sideband_mag[run][params.fft_plot_harmonics], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('Sideband magnitude [arb. units.]')
plt.title('Quadrupole mode')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/quad_sideband_spectrum.png')

plt.figure('sideband_spectrum_ratio')
for run in range(N_runs):
    plt.plot(params.fft_plot_harmonics, dipole_exc_dipole_sideband_mag[run][params.fft_plot_harmonics] /\
             quad_exc_quad_sideband_mag[run][params.fft_plot_harmonics], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('Dipole / quadrupole sideband ratio')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/sideband_spectrum_ratio.png')