
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
from scipy.constants import c
from blond.impedances.impedance_sources import Resonators


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
params.resonator_list = [Resonators(0.001*10*7.691696828196692195e+02,\
                                    9.860944280723674223e+06, \
                                        8.157582101860359813e+00)]
#Beam parameters:
params.n_bunches = 21
params.bunch_spacing_buckets = 1
params.bunch_length = 4*2/c
params.intensity_list = [84*2.6e11/params.n_bunches] * params.n_bunches
params.minimum_n_macroparticles = [1e4] * params.n_bunches


params.cbfb_params = {'N_channels' : 3,#13,
                      'h_in' : [1, 20, 22, 41, 43, 62, 64, 83, 85, 104, 106, 125, 127],
                      'h_out' : [1] * 13,
                      'active' : [False] * 13,
                      'sideband_swap' : [True, False] * 6 + [True],
                      'gain' : [np.zeros(params.N_t+1, complex)] * 13}

params.rf_params = {'dt' : 5e-9, 
                    'impulse_response' : np.ones(1), 
                    'max_voltage' : 1e5, 
                    'output_delay' : 1e-8,
                    'history_length' : 1e-6}

params.start_cbfb_turn = 15000
params.end_cbfb_turn = 20000
params.cbfb_active_mask = [True] * 13  #Only these channels get activated on SCBFB

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 200
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.fs_exc = 442.07
params.exc_harmonic = 20
params.exc_mod_harm = 0
params.exc_mod_phase = np.pi/2

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
params.fft_span_around_harmonic = 6*params.fs_exc

params.mode_analysis_window = 4000
params.mode_analysis_resolution = 2000
params.N_plt_modes = 4


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
     pos_mode_amp, width_mode_amp, cbfb_usb_mag, cbfb_lsb_mag] = run_cb_sim(params)
        
    cb_data = {'params' : params,
        'dipole_usb_mag' : dipole_usb_mag,
        'dipole_lsb_mag' : dipole_lsb_mag,
        'quad_usb_mag' : quad_usb_mag,
        'quad_lsb_mag' : quad_lsb_mag,
        'pos_mode_amp' : pos_mode_amp,
        'width_mode_amp' : width_mode_amp,
        'cbfb_usb_mag' : cbfb_usb_mag,
        'cbfb_lsb_mag' : cbfb_lsb_mag}
     
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
     pos_mode_amp, width_mode_amp, cbfb_usb_mag, cbfb_lsb_mag] = run_cb_sim(params)
        
    cb_data = {'params' : params,
        'dipole_usb_mag' : dipole_usb_mag,
        'dipole_lsb_mag' : dipole_lsb_mag,
        'quad_usb_mag' : quad_usb_mag,
        'quad_lsb_mag' : quad_lsb_mag,
        'pos_mode_amp' : pos_mode_amp,
        'width_mode_amp' : width_mode_amp,
        'cbfb_usb_mag' : cbfb_usb_mag,
        'cbfb_lsb_mag' : cbfb_lsb_mag}

    with open(this_directory + results_dir + 'quad_run_' + str(run) + '.pickle', 'wb') as f:
        pickle.dump(cb_data, f)

    print('Quad run '  + str(run) + ' finished.')
    

#Read, post-process, and plot data:
dipole_exc_dipole_sideband_mag = [None] * N_runs
dipole_exc_quad_sideband_mag = [None] * N_runs
dipole_exc_cbfb_bb_mag = [None] * N_runs
quad_exc_dipole_sideband_mag = [None] * N_runs
quad_exc_quad_sideband_mag = [None] * N_runs
quad_exc_cbfb_bb_mag = [None] * N_runs

for run in range(N_runs):
    with open(this_directory + results_dir + 'dipole_run_' + str(run) + '.pickle', 'rb') as f:
        dipole_data = pickle.load(f)    
    
    dipole_exc_dipole_sideband_mag[run] = (np.array(dipole_data['dipole_usb_mag']) + \
                                           np.array(dipole_data['dipole_lsb_mag'])) / 2
    dipole_exc_quad_sideband_mag[run] = (np.array(dipole_data['quad_usb_mag']) + \
                                         np.array(dipole_data['quad_lsb_mag'])) / 2
    dipole_exc_cbfb_bb_mag[run] = (np.array(dipole_data['cbfb_usb_mag']) + \
                                         np.array(dipole_data['cbfb_lsb_mag'])) / 2
  
    with open(this_directory + results_dir + 'quad_run_' + str(run) + '.pickle', 'rb') as f:
        quad_data = pickle.load(f)
        
    quad_exc_dipole_sideband_mag[run] = (np.array(quad_data['dipole_usb_mag']) + \
                                           np.array(quad_data['dipole_lsb_mag'])) / 2
    quad_exc_quad_sideband_mag[run] = (np.array(quad_data['quad_usb_mag']) + \
                                         np.array(quad_data['quad_lsb_mag'])) / 2
    quad_exc_cbfb_bb_mag[run] = (np.array(quad_data['cbfb_usb_mag']) + \
                                         np.array(quad_data['cbfb_lsb_mag'])) / 2

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

plt.figure('dipole_fb_baseband_spectrum')
for run in range(N_runs):
    plt.plot(params.cbfb_params['h_in'][0:3], dipole_exc_cbfb_bb_mag[run], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('CBFB baseband magnitude [arb. units.]')
plt.title('Dipole mode')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/dipole_fb_baseband_spectrum.png')

plt.figure('quad_fb_baseband_spectrum')
for run in range(N_runs):
    plt.plot(params.cbfb_params['h_in'][0:3], quad_exc_cbfb_bb_mag[run], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('CBFB baseband magnitude [arb. units.]')
plt.title('Quadrupole mode')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/quad_fb_baseband_spectrum.png')

plt.figure('fb_baseband_spectrum_ratio')
for run in range(N_runs):
    plt.plot(params.cbfb_params['h_in'][0:3], dipole_exc_cbfb_bb_mag[run] /\
             quad_exc_cbfb_bb_mag[run], \
             label = 'Exc amp = ' + str(exc_amp_runs[run]) + ' V')
plt.xlabel('Measurement harmonic')
plt.ylabel('Dipole / quadrupole  CBFB baseband signal ratio')
plt.legend(loc=0, fontsize='medium')
plt.savefig(this_directory + results_dir + '/fb_baseband_spectrum_ratio.png')