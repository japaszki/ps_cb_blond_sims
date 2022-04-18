
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import division
from __future__ import print_function
import os
from builtins import str
import numpy as np
from cavity_model import resonator_impulse_response
from scipy.constants import c
from blond.impedances.impedance_sources import Resonators
from setup_run import setup_run


class sim_params:
    pass
        
#Define static parameters:
params = sim_params()

# Tracking details
params.N_t = 30000     # Number of turns to track

# Beam parameters
params.n_particles = 1e10
params.n_macroparticles = 1e3
params.sync_momentum = 15e9# [eV]
                        
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

params.n_turns_memory = 100
params.filter_front_wake = 0.5

#Beam parameters:
params.n_bunches = 21
params.bunch_spacing_buckets = 1
params.bunch_length = 4*1/c
params.intensity_list = [84*2.6e11/params.n_bunches] * params.n_bunches
params.minimum_n_macroparticles = [1e5] * params.n_bunches

params.cbfb_params = {'N_channels' : 1,
                      'h_in' : [1],
                      'h_out' : [1],
                      'active' : [False],
                      'sideband_swap' : [False],
                      'gain' : [np.zeros(params.N_t+1, complex)],
                      'pre_filter' : 'none',
                      'post_filter' : 'none'}

finemet_dt = 5e-9
finemet_f0 = 1.96e6
finemet_Q = 0.49
finemet_h = resonator_impulse_response(2*np.pi*finemet_f0, finemet_Q, finemet_dt, 100)

params.rf_params = {'dt' : finemet_dt, 
                    'impulse_response' : finemet_h, 
                    'max_voltage' : 1e5, 
                    'output_delay' : 1e-8,
                    'history_length' : 1e-3}

params.start_cbfb_turn = 12000
params.end_cbfb_turn = 30000
params.cbfb_active_mask = [True]

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 500
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.fs_exc = 442.07
params.exc_harmonic = 20
params.exc_mod_harm = 0
params.exc_mod_phase = np.pi/2
params.exc_waveform = 'sine'

#Simulation parameters
params.profile_plot_bunch = 0
params.phase_plot_dt = 500
params.phase_plot_max_dE = 100e6
params.tomo_n_slices = 10000
params.tomo_dt = 10
params.fft_n_slices = 64
params.fft_start_turn = 14000
params.fft_end_turn = 20000
params.fft_plot_harmonics = [20]
params.fft_span_around_harmonic = 6*params.fs_exc

params.mode_analysis_window = 4000
params.mode_analysis_resolution = 2000
params.N_plt_modes = 4

params.cbfb_mag_window = 3001

dipole_exc_v = 2e3
quad_exc_v = 4e3

N_phases = 32
phase_vals = np.linspace(0, 2*np.pi, N_phases)

# N_gains = 3
# gain_vals = [3e-4, 1e-3, 3e-3, 1e-2]


gain_vals = [1e-5, 1e-4]

#Arrange 2D grid of gain and phase values:
[cbfb_gain_2d, cbfb_phase_2d] = np.meshgrid(gain_vals, phase_vals)

#Convert to 1D arrays, with one run of zero gain to use as reference:
cbfb_gain_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_gain_2d)))
cbfb_phase_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_phase_2d)))

N_runs = cbfb_gain_runs.shape[0]

working_dir = os.getcwd()
source_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
job_flavour = '"workday"'



#Baseline CBFB parameters:

# =============================================================================
# scans_dir = '/scans/cbfb_baseline_gain_phase_scan/'
# 
# params.cbfb_params['pre_filter'] = 'none'
# params.cbfb_params['post_filter'] = 'none'
# 
# #Dipole mode runs:
# params.exc_v[0:10000] = dipole_exc_v
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'dipole_run' + str(run) + '/'
#     
#     params.exc_delta_freq = params.fs_exc
#     params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
#     
#     setup_run(run_dir, source_dir, params, job_flavour)
#     
# #Quadrupole mode runs:
# params.exc_v[0:10000] = quad_exc_v
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'quad_run' + str(run) + '/'
#     
#     #Set swept parameters for this run:
#     params.exc_delta_freq = 2*params.fs_exc
#     params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
#     
#     setup_run(run_dir, source_dir, params, job_flavour)
#     
# =============================================================================



#Peak detector and h21 modulation:
gain_vals = [2e-6, 5e-6]

#Arrange 2D grid of gain and phase values:
[cbfb_gain_2d, cbfb_phase_2d] = np.meshgrid(gain_vals, phase_vals)

#Convert to 1D arrays, with one run of zero gain to use as reference:
cbfb_gain_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_gain_2d)))
cbfb_phase_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_phase_2d)))

N_runs = cbfb_gain_runs.shape[0]


scans_dir = '/scans/cbfb_peak_h21_gain_phase_scan/'

params.cbfb_params['pre_filter'] = 'peak'
params.cbfb_params['post_filter'] = 'h21_mod_h256_clk_saw'

#Dipole mode runs:
params.exc_v[0:10000] = dipole_exc_v
for run in range(N_runs):
    run_dir = working_dir + scans_dir + 'dipole_run' + str(run) + '/'
    
    params.exc_delta_freq = params.fs_exc
    params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
    
    setup_run(run_dir, source_dir, params, job_flavour)
    
#Quadrupole mode runs:
params.exc_v[0:10000] = quad_exc_v
for run in range(N_runs):
    run_dir = working_dir + scans_dir + 'quad_run' + str(run) + '/'
    
    #Set swept parameters for this run:
    params.exc_delta_freq = 2*params.fs_exc
    params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
    
    setup_run(run_dir, source_dir, params, job_flavour)



# =============================================================================
# # #Width measurement and h21 modulation:
# gain_vals = [1e14, 3e14]
# 
# #Arrange 2D grid of gain and phase values:
# [cbfb_gain_2d, cbfb_phase_2d] = np.meshgrid(gain_vals, phase_vals)
# 
# #Convert to 1D arrays, with one run of zero gain to use as reference:
# cbfb_gain_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_gain_2d)))
# cbfb_phase_runs = np.concatenate((np.zeros(1), np.ndarray.flatten(cbfb_phase_2d)))
# 
# N_runs = cbfb_gain_runs.shape[0]
# 
# 
# scans_dir = '/scans/cbfb_width_h21_gain_phase_scan/'
# 
# params.cbfb_params['pre_filter'] = 'width'
# params.cbfb_params['post_filter'] = 'h21_mod_h256_clk'
# 
# #Dipole mode runs:
# params.exc_v[0:10000] = dipole_exc_v
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'dipole_run' + str(run) + '/'
#     
#     params.exc_delta_freq = params.fs_exc
#     params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
#     
#     setup_run(run_dir, source_dir, params, job_flavour)
#     
# #Quadrupole mode runs:
# params.exc_v[0:10000] = quad_exc_v
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'quad_run' + str(run) + '/'
#     
#     #Set swept parameters for this run:
#     params.exc_delta_freq = 2*params.fs_exc
#     params.cbfb_params['gain'][0][:] = cbfb_gain_runs[run] * np.exp(1j * cbfb_phase_runs[run])
#     
#     setup_run(run_dir, source_dir, params, job_flavour)
# =============================================================================
