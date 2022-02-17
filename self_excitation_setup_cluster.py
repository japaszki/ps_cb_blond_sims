
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/blond')

from builtins import str
import numpy as np
import pickle
import os
import cavity_model
from scipy.constants import c
from blond.impedances.impedance_sources import Resonators


class sim_params:
    pass
        
#Define static parameters:
params = sim_params()

# Tracking details
params.N_t = 80000     # Number of turns to track

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
params.resonator_list = [Resonators(5*10*7.691696828196692195e+02,\
                                    9.860944280723674223e+06, \
                                        8.157582101860359813e+00)]

params.n_turns_memory = 100
params.filter_front_wake = 0.5
    
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

params.cbfb_params['gain'][0][:] = 1e-3 * np.exp(2j * np.pi * 0.0)

finemet_dt = 5e-9
finemet_f0 = 1.96e6
finemet_Q = 0.49
finemet_h = cavity_model.resonator_impulse_response(2*np.pi*finemet_f0, finemet_Q, finemet_dt, 100)

params.rf_params = {'dt' : finemet_dt, 
                    'impulse_response' : finemet_h, 
                    'max_voltage' : 1e5, 
                    'output_delay' : 1e-8,
                    'history_length' : 1e-6}

params.start_cbfb_turn = 65000
params.end_cbfb_turn = 80000
params.cbfb_active_mask = [True] #Only these channels get activated on SCBFB

params.fb_diag_dt = 25
params.fb_diag_plot_dt = 200
params.fb_diag_start_delay = 100

# Excitation parameters:
params.exc_v = np.zeros(params.N_t+1)
params.exc_v[:] = 0
params.fs_exc = 442.07
params.exc_harmonic = 20
params.exc_delta_freq = params.fs_exc
params.exc_mod_harm = 0
params.exc_mod_phase = np.pi/2

#Simulation parameters
params.profile_plot_bunch = 0
params.phase_plot_dt = 200
params.phase_plot_max_dE = 100e6
params.tomo_n_slices = 3000
params.tomo_dt = 10
params.fft_n_slices = 64
params.fft_start_turn = 52000
params.fft_end_turn = 58000
params.fft_plot_harmonics = [20]
params.fft_span_around_harmonic = 6*params.fs_exc

params.mode_analysis_window = 4000
params.mode_analysis_resolution = 2000
params.N_plt_modes = 4

params.cbfb_mag_window = 3001

working_dir = os.getcwd()
scans_dir = '/scans/self_exc_test/'
source_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

N_runs = 1

for run in range(N_runs):
    run_dir = working_dir + scans_dir + 'run' + str(run) + '/'
    
    #Create root directory for this run:
    try:
        os.makedirs(run_dir)
    except:
        pass   
    
    #Create directories for Condor outputs:
    try:
        os.makedirs(run_dir + 'log/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'output/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'error/')
    except:
        pass
    
    #Copy bash script:
    os.system('cp run.sh ' + run_dir)
    #Copy submit file:
    os.system('cp run.sub ' + run_dir)
    
    #Create pickle file with parameters:
    with open(run_dir + '/input_params.pickle', 'wb') as f:
        pickle.dump(params, f)

    #Submit job:
    os.system('condor_submit run.sub')