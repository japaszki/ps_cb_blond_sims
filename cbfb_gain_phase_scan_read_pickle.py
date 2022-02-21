
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
# from coupled_bunch_diag import plot_modes_vs_time

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

class sim_params:
    pass
        
with open('self_exc_test_results.pickle', 'rb') as f:
        data = pickle.load(f)

N_runs_fb = 160
plot_mode = 20
    
working_dir = os.getcwd()
scans_dir = '/scans/cbfb_baseline_gain_phase_scan/'
source_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

#Baseline dipole run:
run_dir = working_dir + scans_dir + 'dipole_run' + str(0) + '/'

with open(run_dir + 'input_params.pickle', 'rb') as f:
     data = pickle.load(f)
    
base_dipole_cbfb_usb = data['cbfb_usb_mag']
base_dipole_cbfb_lsb = data['cbfb_lsb_mag']

base_dipole_pos_mode_amp = data['pos_mode_amp'][plot_mode]
base_dipole_width_mode_amp = data['width_mode_amp'][plot_mode]

#Baseline quad run:
run_dir = working_dir + scans_dir + 'quad_run' + str(0) + '/'

with open(run_dir + 'input_params.pickle', 'rb') as f:
     data = pickle.load(f)
    
base_quad_cbfb_usb = data['cbfb_usb_mag']
base_quad_cbfb_lsb = data['cbfb_lsb_mag']

base_quad_pos_mode_amp = data['pos_mode_amp'][plot_mode]
base_quad_width_mode_amp = data['width_mode_amp'][plot_mode]

#Paramter scans with feedback:

#Get data for dipole runs:
dipole_fb_gains = np.empty(N_runs_fb)
dipole_fb_phases = np.empty(N_runs_fb)
dipole_exc_dipole_sideband_mag = np.empty(N_runs_fb)
dipole_exc_quad_sideband_mag = np.empty(N_runs_fb)
dipole_exc_pos_amp = np.empty(N_runs_fb)
dipole_exc_width_amp =  np.empty(N_runs_fb)

for run in range(N_runs_fb):
    run_dir = working_dir + scans_dir + 'dipole_run' + str(run+1) + '/'
    
    with open(run_dir + 'input_params.pickle', 'rb') as f:
        data = pickle.load(f)
     
    dipole_fb_gains[run] = np.abs(data['params'].cbfb_params['gain'][0][0])
    dipole_fb_phases[run] = np.angle(data['params'].cbfb_params['gain'][0][0])
    dipole_exc_pos_amp[run] = data['pos_mode_amp'][plot_mode]
    dipole_exc_quad_sideband_mag[run] = data['width_mode_amp'][plot_mode]

dipole_unique_gains = np.unique(dipole_fb_gains)

#Get data for quadrupole runs:
quad_fb_gains = np.empty(N_runs_fb)
quad_fb_phases = np.empty(N_runs_fb)
quad_exc_dipole_sideband_mag = np.empty(N_runs_fb)
quad_exc_quad_sideband_mag = np.empty(N_runs_fb)
quad_exc_pos_amp = np.empty(N_runs_fb)
quad_exc_width_amp =  np.empty(N_runs_fb)

for run in range(N_runs_fb):
    run_dir = working_dir + scans_dir + 'quad_run' + str(run+1) + '/'
    
    with open(run_dir + 'input_params.pickle', 'rb') as f:
        data = pickle.load(f)
    
    quad_fb_gains[run] = np.abs(data['params'].cbfb_params['gain'][0][0])
    quad_fb_phases[run] = np.angle(data['params'].cbfb_params['gain'][0][0])
    quad_exc_pos_amp[run] = data['pos_mode_amp'][plot_mode]
    dipole_exc_quad_sideband_mag[run] = data['width_mode_amp'][plot_mode]

quad_unique_gains = np.unique(quad_fb_gains)



#Relative mode amplitude after CBFB on:
plt.figure('dipole_amp_vs_phase')
for i in range(dipole_unique_gains.shape[0]):
    plot_indices = (dipole_fb_gains == dipole_unique_gains[i])
    plt.semilogy(dipole_fb_phases[plot_indices], dipole_exc_pos_amp[plot_indices] /\
             base_dipole_pos_mode_amp, label='Gain = ' + str(dipole_unique_gains[i]))
plt.legend(loc=0, fontsize='medium')
plt.xlabel('CBFB phase [rad]')
plt.ylabel('Relative oscillation amplitude')
plt.title('Dipole mode')
plt.savefig(this_directory + 'cbfb_gain_phase_scan_plots/dipole_amp_vs_phase.png')

plt.figure('quad_amp_vs_phase')
for i in range(quad_unique_gains.shape[0]):
    plot_indices = (quad_fb_gains == quad_unique_gains[i])
    plt.semilogy(quad_fb_phases[plot_indices], quad_exc_pos_amp[plot_indices] /\
             base_quad_pos_mode_amp, label='Gain = ' + str(quad_unique_gains[i]))
plt.legend(loc=0, fontsize='medium')
plt.xlabel('CBFB phase [rad]')
plt.ylabel('Relative oscillation amplitude')
plt.title('Qudrupole mode')
plt.savefig(this_directory + 'cbfb_gain_phase_scan_plots/quad_amp_vs_phase.png')