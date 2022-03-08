# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:07:01 2022

@author: JohnG
"""
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import numpy as np
import pylab as plt
import scipy.signal

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.beam.distributions_multibunch \
                            import matched_from_distribution_density_multibunch
from blond.monitors.monitors import BunchMonitor
from blond.trackers.utilities import synchrotron_frequency_distribution
from blond.plots.plot import Plot
import cbfb
import coupled_bunch_diag as cbd
from blond.utils import bmath as bm

#Dummy class to store paramters:
class sim_params:
    pass

# os.chdir('scans/self_exc_test/run2/')

working_dir = os.getcwd()
output_dir = working_dir + '/sim_outputs/'
print("Working directory is " + working_dir)

#Load input parameters:
print("Loading input paramters from " + working_dir + '/input_params.pickle ...')
with open(working_dir + '/input_params.pickle', 'rb') as f:
        params = pickle.load(f)
        
#Create directory to store results:
try:
    print("Creating directory " + output_dir)
    os.makedirs(output_dir)
except:
    pass

#Initialise variables to store data:     
long_beam_signal = []
cbfb_baseband_vec = [ [] for x in range(params.cbfb_params['N_channels']) ]
fft_n_harmonics = int(np.floor(params.fft_n_slices/2))
dipole_usb_mag = [None] * fft_n_harmonics
dipole_lsb_mag = [None] * fft_n_harmonics
quad_usb_mag = [None] * fft_n_harmonics
quad_lsb_mag = [None] * fft_n_harmonics

tomo_N_lines = int(np.ceil(params.N_t / params.tomo_dt) + 1)
tomogram = np.empty([tomo_N_lines, params.tomo_n_slices])
        
#Define ring and RF systems:
momentum_compaction = 1 / params.gamma_transition**2
ring = Ring(params.circumference, momentum_compaction, params.sync_momentum, Proton(), params.N_t)
beam = Beam(ring, params.n_macroparticles, params.n_particles)
rf = RFStation(ring, [params.harmonic_number], [params.voltage_program],
                       [params.phi_offset], params.n_rf_systems)
tracker = RingAndRFTracker(rf, beam)
full_tracker = FullRingAndRF([tracker]) 

bucket_length = rf.t_rf[0, 0]
turn_length = params.harmonic_number * bucket_length
f_rev = 1 / turn_length
exc_freq = params.exc_harmonic * f_rev + params.exc_delta_freq
fft_dt = turn_length / params.fft_n_slices

tomo_profile = Profile(beam, CutOptions(cut_left=0, 
                    cut_right=turn_length, n_slices=params.tomo_n_slices))

long_fft_profile = Profile(beam, CutOptions(cut_left=0, 
                    cut_right=turn_length, n_slices=params.fft_n_slices))

# WAKE IMPEDANCE -------------------------------------------------------
frequency_step = 1/(ring.t_rev[0]*params.n_turns_memory) # [Hz]
front_wake_length = params.filter_front_wake * ring.t_rev[0]*params.n_turns_memory

intensity_freq = InducedVoltageFreq(beam,
                                   tomo_profile,
                                   params.resonator_list,
                                   frequency_step,
                                   RFParams=rf,
                                   multi_turn_wake=True,
                                   front_wake_length=front_wake_length)

longitudinal_intensity = TotalInducedVoltage(beam, tomo_profile, [intensity_freq])

# BEAM GENERATION -------------------------------------------------------------
distribution_options_list = {'bunch_length': params.bunch_length,
                             'type': 'parabolic_amplitude',
                             'density_variable': 'Hamiltonian'}

matched_from_distribution_density_multibunch(beam, ring,
                             full_tracker, distribution_options_list,
                             params.n_bunches, params.bunch_spacing_buckets,
                             intensity_list=params.intensity_list,
                             minimum_n_macroparticles=params.minimum_n_macroparticles,
                             TotalInducedVoltage=longitudinal_intensity,
                             n_iterations_input=8, seed=7878)

bunchmonitor = BunchMonitor(ring, rf, beam, output_dir, Profile=tomo_profile)

plots = Plot(ring, rf, beam, params.phase_plot_dt, params.N_t, 
             2*params.profile_plot_bunch*np.pi,
              2*(params.profile_plot_bunch + 1)*np.pi,
              -params.phase_plot_max_dE, params.phase_plot_max_dE, xunit='rad', 
              separatrix_plot=True,
              Profile=tomo_profile,
              sampling = 1,
              format_options={'dirname': output_dir + 'profile_plots/'})


#Set up coupled-bunch feedback parameters:
fb = cbfb.feedback(beam, ring, full_tracker, params.cbfb_params, params.rf_params)
    
fb_diag = cbd.coupled_bunch_diag(beam, ring, full_tracker, fb,\
                                 {'dirname': output_dir + 'cb_plots/'}, \
                                 params.harmonic_number, params.fb_diag_dt, params.fb_diag_plot_dt, params.N_t)
    
plt.figure('finemet_impulse_response')
plt.plot(np.arange(params.rf_params['impulse_response'].shape[0]) * params.rf_params['dt'],\
         params.rf_params['impulse_response'])
plt.xlabel('Time [s]')
plt.ylabel('Finemet impulse response')
plt.savefig(output_dir + 'finemet_impulse_response.png')

plt.figure('finemet_transfer_function')
plt.semilogx(np.fft.rfftfreq(params.rf_params['impulse_response'].shape[0], params.rf_params['dt']),\
         20*np.log10(np.absolute(np.fft.rfft(params.rf_params['impulse_response']))))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Finemet frequency response [dB]')
plt.savefig(output_dir + 'finemet_frequency_response.png')

# Accelerator map
map_ = [full_tracker] + [tomo_profile] + [long_fft_profile] + \
    [longitudinal_intensity] + [bunchmonitor] + [fb] + [fb_diag] + [plots]
print("Map set")
print("")


# Tracking --------------------------------------------------------------------
turn_start_time = 0

for turn in range(1, params.N_t+1):
    # Track
    for m in map_:
        m.track()
        
    #Apply excitation:
    turn_length = ring.t_rev[turn]
    f_rev = 1 / turn_length
    exc_freq = params.exc_harmonic * f_rev + params.exc_delta_freq
    mod_freq = params.exc_mod_harm * f_rev
        
    particle_t = turn_start_time + beam.dt
    beam.dE += params.exc_v[turn] * bm.sin(2*np.pi*exc_freq*particle_t) * \
        bm.sin(2*np.pi*mod_freq*particle_t + params.exc_mod_phase)
    turn_start_time += turn_length
    
    #Record CBFB baseband signals:
    for channel in range(params.cbfb_params['N_channels']):
        cbfb_baseband_vec[channel].append(fb.dipole_channels[channel].cic_decim_out)
    
    if turn == params.start_cbfb_turn:
        for channel in range(params.cbfb_params['N_channels']):
            if params.cbfb_active_mask[channel]:
                fb.active[channel] = True
    
    if turn == params.end_cbfb_turn:
        for channel in range(params.cbfb_params['N_channels']):
            fb.active[channel] = False
    
    #Collect data for long FFT
    if turn >= params.fft_start_turn and turn < params.fft_end_turn:
        long_beam_signal.extend(long_fft_profile.n_macroparticles)
        
    #Collect data for tomoscope
    if (turn % params.tomo_dt) == 0:
        tomo_line = int(turn / params.tomo_dt)
        tomogram[tomo_line, :] = tomo_profile.n_macroparticles
            
print("Tracking Done")
            
# Plot narrowband beam spectrum                    
long_fft = bm.rfft(long_beam_signal * np.hanning(len(long_beam_signal)))
long_fft_freq = bm.rfftfreq(len(long_beam_signal), fft_dt)

print("FFT Calculated")

#Plot FFTs:
for h in params.fft_plot_harmonics:
    f_plot_centre = f_rev * h
    f_plot_min = f_plot_centre - params.fft_span_around_harmonic / 2
    f_plot_max = f_plot_centre + params.fft_span_around_harmonic / 2
    plot_mask = (long_fft_freq >= f_plot_min) & (long_fft_freq < f_plot_max)
    
    plt.figure('beam_spectrum', figsize=(8,6))
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(long_fft_freq[plot_mask] - f_plot_centre, np.absolute(long_fft[plot_mask]), '-')
    ax.set_xlabel("Relative frequency [Hz]")
    ax.set_ylabel('Beam spectrum, absolute value [arb. units]')
    plt.title('Beam spectrum around h' + str(h))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(output_dir + 'beam_spectrum_h' + str(h) + '.png')
    plt.clf()

plot_mask = (long_fft_freq >= 0) & (long_fft_freq < params.harmonic_number*f_rev)

plt.figure('beam_spectrum_full', figsize=(8,6))
ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
ax.plot(long_fft_freq[plot_mask], np.absolute(long_fft[plot_mask]), '-')
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel('Beam spectrum, absolute value [arb. units]')
plt.title('Beam spectrum')
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(output_dir + 'beam_spectrum_full.png')
plt.close()

# Plot tomograms
plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*params.tomo_dt):params.tomo_dt, 0:params.tomo_n_slices]
plt.pcolormesh(plot_x, plot_y, tomogram, cmap='hot', shading='flat')
plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
plt.xlabel('Time [sample]')
plt.ylabel('Turn')
plt.colorbar()
plt.rc('font', size=16)
plt.savefig(output_dir + 'tomogram.png')
plt.close()

tomo_zoom_start = round(params.tomo_n_slices * params.profile_plot_bunch / params.harmonic_number)
tomo_zoom_end = round(params.tomo_n_slices * (params.profile_plot_bunch+1) / params.harmonic_number)

plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*params.tomo_dt):params.tomo_dt, tomo_zoom_start:tomo_zoom_end]
plt.pcolormesh(plot_x, plot_y, tomogram[:, tomo_zoom_start:tomo_zoom_end], cmap='hot', shading='flat')
plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
plt.xlabel('Time [sample]')
plt.ylabel('Turn')
plt.colorbar()
plt.rc('font', size=16)
plt.savefig(output_dir + 'tomogram_zoom.png')
plt.close()

#Coupled-bunch analysis

#Sideband magnitudes:
#Measure up to half of FFT sample rate:
for h in range(fft_n_harmonics):
    rel_freq = long_fft_freq - f_rev * h
    
    dipole_usb_mag[h] = np.mean(np.abs(long_fft[(rel_freq >= 0.5*abs(params.fs_exc)) & \
                                         (rel_freq < 1.5*abs(params.fs_exc))]))
    quad_usb_mag[h] = np.mean(np.abs(long_fft[(rel_freq >= 0.5*2*abs(params.fs_exc)) & \
                                       (rel_freq < 1.5*2*abs(params.fs_exc))]))
    dipole_lsb_mag[h] = np.mean(np.abs(long_fft[(rel_freq >= -1.5*abs(params.fs_exc)) & \
                                         (rel_freq < -0.5*abs(params.fs_exc))]))
    quad_lsb_mag[h] = np.mean(np.abs(long_fft[(rel_freq >= -1.5*2*abs(params.fs_exc)) & \
                                       (rel_freq < -0.5*2*abs(params.fs_exc))]))

plt.figure('cbfb_baseband_raw')
plt.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
        np.real(cbfb_baseband_vec[0][params.fb_diag_start_delay:]), label='I')
plt.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
        np.imag(cbfb_baseband_vec[0][params.fb_diag_start_delay:]), label='Q')
plt.xlabel("Turn")
plt.ylabel('CBFB baseband signal [arb. units]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(output_dir + 'cbfb_baseband_raw.png')
plt.close()

#Calculate CBFB baseband magnitude:
cbfb_bb_usb = [[] for i in range(params.cbfb_params['N_channels'])]
cbfb_bb_lsb = [[] for i in range(params.cbfb_params['N_channels'])]
cbfb_bb_usb_mag = [[] for i in range(params.cbfb_params['N_channels'])]
cbfb_bb_lsb_mag = [[] for i in range(params.cbfb_params['N_channels'])]
cbfb_bb_usb_mag_mean = [[] for i in range(params.cbfb_params['N_channels'])]
cbfb_bb_lsb_mag_mean = [[] for i in range(params.cbfb_params['N_channels'])]

for channel in range(params.cbfb_params['N_channels']):
    hilb = np.imag(scipy.signal.hilbert(np.imag(cbfb_baseband_vec[channel])))
    cbfb_bb_usb[channel] = np.real(cbfb_baseband_vec[channel]) + hilb
    cbfb_bb_lsb[channel] = np.real(cbfb_baseband_vec[channel]) - hilb

    cbfb_bb_usb_mag[channel] = scipy.signal.savgol_filter(np.abs(cbfb_bb_usb[channel]),\
                                                          params.cbfb_mag_window, 5)
    cbfb_bb_lsb_mag[channel] = scipy.signal.savgol_filter(np.abs(cbfb_bb_lsb[channel]), \
                                                          params.cbfb_mag_window, 5)
    
    cbfb_bb_usb_mag_mean[channel] = np.mean(cbfb_bb_usb_mag[channel][params.fft_start_turn:params.fft_end_turn])
    cbfb_bb_lsb_mag_mean[channel] = np.mean(cbfb_bb_lsb_mag[channel][params.fft_start_turn:params.fft_end_turn])
    
plt.figure('cbfb_baseband_magnitudes')
for channel in range(params.cbfb_params['N_channels']):
    plt.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
            cbfb_bb_usb_mag[channel][params.fb_diag_start_delay:], label='Ch. '+str(channel)+' USB')
    plt.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
            cbfb_bb_lsb_mag[channel][params.fb_diag_start_delay:], label='Ch. '+str(channel)+' LSB')
plt.legend(loc=0, fontsize='medium')
plt.xlabel("Turn")
plt.ylabel('CBFB baseband signal [arb. units]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(output_dir + 'cbfb_baseband_magnitudes.png')
plt.close()
     
#Plot bunch width and position history
fb_diag.plot_size_width(params.fb_diag_start_delay, params.N_t)
[pos_mode_spectrum, width_mode_spectrum] = fb_diag.mode_analysis(params.fft_start_turn, params.fft_end_turn, True)

[mode_times, pos_modes_vs_time, width_modes_vs_time] = fb_diag.modes_vs_time(params.fb_diag_start_delay, params.N_t,\
                      params.mode_analysis_window, params.mode_analysis_resolution, params.N_plt_modes)

#Get magnitudes of desired modes:                             
pos_mode_amp = [np.abs(pos_mode_spectrum[m]) for m in range(params.harmonic_number)]
width_mode_amp = [np.abs(width_mode_spectrum[m]) for m in range(params.harmonic_number)]

# Plot synchrotron frequency distribution
[sync_freq_distribution_left, sync_freq_distribution_right], \
    [emittance_array_left, emittance_array_right], \
    [delta_time_left, delta_time_right], \
    particleDistributionFreq, synchronous_time = \
                          synchrotron_frequency_distribution(beam, full_tracker,
                                  TotalInducedVoltage=None)

# Plot of the synchrotron frequency distribution
plt.figure('fs_distribution')
plt.plot(delta_time_left, sync_freq_distribution_left, lw=2, label='Left')
plt.plot(delta_time_right, sync_freq_distribution_right, 'r--', lw=2,
          label='Right')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Amplitude of particle oscillations [s]')
plt.ylabel('Synchrotron frequency [Hz]')
plt.title('Synchrotron frequency distribution, no induced voltage')
plt.savefig(output_dir + 'fs_distribution.png')
plt.close()

print('Zero-amplitude synchrotron frequency : ' + str(sync_freq_distribution_left[0]) + ' Hz')

cb_data = {'params' : params,
    'dipole_usb_mag' : dipole_usb_mag,
    'dipole_lsb_mag' : dipole_lsb_mag,
    'quad_usb_mag' : quad_usb_mag,
    'quad_lsb_mag' : quad_lsb_mag,
    'pos_mode_amp' : pos_mode_amp,
    'width_mode_amp' : width_mode_amp,
    'cbfb_usb_mag' : cbfb_bb_usb_mag_mean,
    'cbfb_lsb_mag' : cbfb_bb_lsb_mag_mean,
    'mode_times' : mode_times,
    'pos_modes_vs_time' : pos_modes_vs_time,
    'width_modes_vs_time' : width_modes_vs_time}
 
with open(working_dir + '/results.pickle', 'wb') as f:
    pickle.dump(cb_data, f)