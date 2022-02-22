
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

class sim_params:
    pass

N_runs_fb = 160
plot_modes = [20]
    
working_dir = os.getcwd() + '/'
scans_dir = '/scans/cbfb_peak_h21_gain_phase_scan/'

#Baseline dipole run:
run_dir = working_dir + scans_dir + 'dipole_run' + str(0) + '/'

with open(run_dir + 'results.pickle', 'rb') as f:
     data = pickle.load(f)
    
base_dipole_cbfb_usb = data['cbfb_usb_mag']
base_dipole_cbfb_lsb = data['cbfb_lsb_mag']

base_dipole_pos_mode_amp = data['pos_mode_amp']
base_dipole_width_mode_amp = data['width_mode_amp']
N_modes = len(base_dipole_pos_mode_amp)

#Baseline quad run:
run_dir = working_dir + scans_dir + 'quad_run' + str(0) + '/'

with open(run_dir + 'results.pickle', 'rb') as f:
     data = pickle.load(f)
    
base_quad_cbfb_usb = data['cbfb_usb_mag']
base_quad_cbfb_lsb = data['cbfb_lsb_mag']

base_quad_pos_mode_amp = data['pos_mode_amp']
base_quad_width_mode_amp = data['width_mode_amp']

#Paramter scans with feedback:

#Get data for dipole runs:
dipole_fb_gains = np.empty(N_runs_fb)
dipole_fb_phases = np.empty(N_runs_fb)
dipole_exc_dipole_sideband_mag = np.empty(N_runs_fb)
dipole_exc_quad_sideband_mag = np.empty(N_runs_fb)
dipole_exc_pos_amp = np.empty((N_runs_fb, N_modes))
dipole_exc_width_amp = np.empty((N_runs_fb, N_modes))

for run in range(N_runs_fb):
    run_dir = working_dir + scans_dir + 'dipole_run' + str(run+1) + '/'
    print('Reading dipole run ' + str(run))
    
    with open(run_dir + 'results.pickle', 'rb') as f:
        data = pickle.load(f)
     
    dipole_fb_gains[run] = np.abs(data['params'].cbfb_params['gain'][0][0])
    dipole_fb_phases[run] = np.angle(data['params'].cbfb_params['gain'][0][0])
    
    dipole_exc_pos_amp[run, :] = data['pos_mode_amp']
    dipole_exc_width_amp[run, :] = data['width_mode_amp']

dipole_unique_gains = np.unique(dipole_fb_gains.round(decimals=8))

#Get data for quadrupole runs:
quad_fb_gains = np.empty(N_runs_fb)
quad_fb_phases = np.empty(N_runs_fb)
quad_exc_dipole_sideband_mag = np.empty(N_runs_fb)
quad_exc_quad_sideband_mag = np.empty(N_runs_fb)
quad_exc_pos_amp = np.empty((N_runs_fb, N_modes))
quad_exc_width_amp =  np.empty((N_runs_fb, N_modes))

for run in range(N_runs_fb):
    run_dir = working_dir + scans_dir + 'quad_run' + str(run+1) + '/'
    print('Reading quadrupole run ' + str(run))
    
    with open(run_dir + 'results.pickle', 'rb') as f:
        data = pickle.load(f)
    
    quad_fb_gains[run] = np.abs(data['params'].cbfb_params['gain'][0][0])
    quad_fb_phases[run] = np.angle(data['params'].cbfb_params['gain'][0][0])
    quad_exc_pos_amp[run, :] = data['pos_mode_amp']
    quad_exc_width_amp[run, :] = data['width_mode_amp']

quad_unique_gains = np.unique(quad_fb_gains.round(decimals=8))

try:
    os.makedirs(working_dir + 'cbfb_gain_phase_scan_plots/')
except:
    pass

#Relative mode amplitude after CBFB on:
for mode in plot_modes:
    plt.figure('dipole_amp_vs_phase')
    for i in range(dipole_unique_gains.shape[0]):
        plot_indices = np.isclose(dipole_fb_gains, dipole_unique_gains[i], rtol=1e-3)
        phase_plt = dipole_fb_phases[plot_indices]
        osc_plt = dipole_exc_pos_amp[plot_indices, mode] / base_dipole_pos_mode_amp[mode]
        sort_indices = np.argsort(phase_plt)
        
        plt.semilogy(phase_plt[sort_indices], osc_plt[sort_indices], \
                     label='Gain = ' + str(dipole_unique_gains[i]))
    plt.legend(loc=0, fontsize='medium')
    plt.xlabel('CBFB phase [rad]')
    plt.ylabel('Relative oscillation amplitude')
    plt.title('Dipole mode ' + str(mode))
    plt.savefig(working_dir + 'cbfb_gain_phase_scan_plots/dipole_amp_vs_phase_mode_' + str(mode) + '.png')
    plt.close()

for mode in plot_modes:
    plt.figure('quad_amp_vs_phase')
    for i in range(quad_unique_gains.shape[0]):
        plot_indices = np.isclose(quad_fb_gains, quad_unique_gains[i], rtol=1e-3)
        phase_plt = quad_fb_phases[plot_indices]
        osc_plt = quad_exc_width_amp[plot_indices, mode] / base_quad_width_mode_amp[mode]
        sort_indices = np.argsort(phase_plt)
        
        plt.semilogy(phase_plt[sort_indices], osc_plt[sort_indices], \
                     label='Gain = ' + str(quad_unique_gains[i]))
    plt.legend(loc=0, fontsize='medium')
    plt.xlabel('CBFB phase [rad]')
    plt.ylabel('Relative oscillation amplitude')
    plt.title('Qudrupole mode ' + str(mode))
    plt.savefig(working_dir + 'cbfb_gain_phase_scan_plots/quad_amp_vs_phase_mode_' + str(mode) + '.png')
    plt.close()