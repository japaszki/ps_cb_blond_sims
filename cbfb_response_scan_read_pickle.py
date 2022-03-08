
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

N_runs = 41
N_modes = 21
plot_modes = [1, 20]
    
working_dir = os.getcwd() + '/'

# =============================================================================
# #Get data for baseline runs:
# scans_dir = '/scans/cbfb_response_scan_baseline/'
# baseline_pos_amp = np.empty((N_runs, N_modes))
# baseline_width_amp = np.empty((N_runs, N_modes))
# baseline_bb = np.empty(N_runs)
# 
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'run' + str(run) + '/'
#     print('Reading baseline run ' + str(run))
#     
#     with open(run_dir + 'results.pickle', 'rb') as f:
#         data = pickle.load(f)
#     
#     baseline_pos_amp[run, :] = data['pos_mode_amp']
#     baseline_width_amp[run, :] = data['width_mode_amp']
#     baseline_bb[run] = data['cbfb_usb_mag'][0] + data['cbfb_lsb_mag'][0]
# =============================================================================


# =============================================================================
# #Get data for peak detector runs:
# scans_dir = '/scans/cbfb_response_scan_peak/'
# peak_pos_amp = np.empty((N_runs, N_modes))
# peak_width_amp = np.empty((N_runs, N_modes))
# peak_bb = np.empty(N_runs)
# 
# for run in range(N_runs):
#     run_dir = working_dir + scans_dir + 'run' + str(run) + '/'
#     print('Reading peak detector run ' + str(run))
#     
#     with open(run_dir + 'results.pickle', 'rb') as f:
#         data = pickle.load(f)
#     
#     peak_pos_amp[run, :] = data['pos_mode_amp']
#     peak_width_amp[run, :] = data['width_mode_amp']
#     peak_bb[run] = data['cbfb_usb_mag'][0] + data['cbfb_lsb_mag'][0]
# =============================================================================

#Get data for peak detector runs:
scans_dir = '/scans/cbfb_response_scan_width/'
peak_pos_amp = np.empty((N_runs, N_modes))
peak_width_amp = np.empty((N_runs, N_modes))
peak_bb = np.empty(N_runs)

for run in range(N_runs):
    run_dir = working_dir + scans_dir + 'run' + str(run) + '/'
    print('Reading width measurement run ' + str(run))
    
    with open(run_dir + 'results.pickle', 'rb') as f:
        data = pickle.load(f)
    
    peak_pos_amp[run, :] = data['pos_mode_amp']
    peak_width_amp[run, :] = data['width_mode_amp']
    peak_bb[run] = data['cbfb_usb_mag'][0] + data['cbfb_lsb_mag'][0]



try:
    os.makedirs(working_dir + 'cbfb_response_scan_plots/')
except:
    pass

#Response magnitude vs position amplitude:
for mode in plot_modes:
# =============================================================================
#     plt.figure('baseline_bb_vs_pos_osc_amp')
#     baseline_pos_amp_mode = baseline_pos_amp[:, mode]
#     sort_indices = np.argsort(baseline_pos_amp_mode)   
#     plt.plot(baseline_pos_amp_mode[sort_indices], baseline_bb[sort_indices])
#     # plt.legend(loc=0, fontsize='medium')
#     plt.xlabel('Position oscillation amplitude [s]')
#     plt.ylabel('Baseband response magnitude')
#     plt.title('Baseline CBFB, mode ' + str(mode))
#     plt.savefig(working_dir + 'cbfb_response_scan_plots/baseline_bb_vs_pos_osc_amp_mode_' + str(mode) + '.png')
#     plt.close()
#     
#     plt.figure('baseline_bb_vs_width_osc_amp')
#     baseline_width_amp_mode = baseline_width_amp[:, mode]
#     sort_indices = np.argsort(baseline_width_amp_mode)   
#     plt.plot(baseline_width_amp_mode[sort_indices], baseline_bb[sort_indices])
#     # plt.legend(loc=0, fontsize='medium')
#     plt.xlabel('Width oscillation amplitude [s]')
#     plt.ylabel('Baseband response magnitude')
#     plt.title('Baseline CBFB, mode ' + str(mode))
#     plt.savefig(working_dir + 'cbfb_response_scan_plots/baseline_bb_vs_width_osc_amp_mode_' + str(mode) + '.png')
#     plt.close()
# =============================================================================
    
# =============================================================================
#     plt.figure('peak_bb_vs_pos_osc_amp')
#     peak_pos_amp_mode = peak_pos_amp[:, mode]
#     sort_indices = np.argsort(peak_pos_amp_mode)   
#     plt.plot(peak_pos_amp_mode[sort_indices], peak_bb[sort_indices])
#     # plt.legend(loc=0, fontsize='medium')
#     plt.xlabel('Position oscillation amplitude [s]')
#     plt.ylabel('Baseband response magnitude')
#     plt.title('Peak-detector CBFB, mode ' + str(mode))
#     plt.savefig(working_dir + 'cbfb_response_scan_plots/peak_bb_vs_pos_osc_amp_mode_' + str(mode) + '.png')
#     plt.close()
#     
#     plt.figure('peak_bb_vs_width_osc_amp')
#     peak_width_amp_mode = peak_width_amp[:, mode]
#     sort_indices = np.argsort(peak_width_amp_mode)   
#     plt.plot(peak_width_amp_mode[sort_indices], peak_bb[sort_indices])
#     # plt.legend(loc=0, fontsize='medium')
#     plt.xlabel('Width oscillation amplitude [s]')
#     plt.ylabel('Baseband response magnitude')
#     plt.title('Peak-detector CBFB, mode ' + str(mode))
#     plt.savefig(working_dir + 'cbfb_response_scan_plots/peak_bb_vs_width_osc_amp_mode_' + str(mode) + '.png')
#     plt.close()
# =============================================================================

    plt.figure('width_bb_vs_pos_osc_amp')
    peak_pos_amp_mode = peak_pos_amp[:, mode]
    sort_indices = np.argsort(peak_pos_amp_mode)   
    plt.plot(peak_pos_amp_mode[sort_indices], peak_bb[sort_indices])
    # plt.legend(loc=0, fontsize='medium')
    plt.xlabel('Position oscillation amplitude [s]')
    plt.ylabel('Baseband response magnitude')
    plt.title('Width filter CBFB, mode ' + str(mode))
    plt.savefig(working_dir + 'cbfb_response_scan_plots/width_bb_vs_pos_osc_amp_mode_' + str(mode) + '.png')
    plt.close()
    
    plt.figure('width_bb_vs_width_osc_amp')
    peak_width_amp_mode = peak_width_amp[:, mode]
    sort_indices = np.argsort(peak_width_amp_mode)   
    plt.plot(peak_width_amp_mode[sort_indices], peak_bb[sort_indices])
    # plt.legend(loc=0, fontsize='medium')
    plt.xlabel('Width oscillation amplitude [s]')
    plt.ylabel('Baseband response magnitude')
    plt.title('Width filter CBFB, mode ' + str(mode))
    plt.savefig(working_dir + 'cbfb_response_scan_plots/width_bb_vs_width_osc_amp_mode_' + str(mode) + '.png')
    plt.close()
