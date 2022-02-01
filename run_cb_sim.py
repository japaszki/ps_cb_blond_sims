# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:07:01 2022

@author: JohnG
"""
import os
import numpy as np
import pylab as plt
# from scipy.constants import m_p ,c, e

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance_sources import Resonators
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.beam.distributions_multibunch \
                            import matched_from_distribution_density_multibunch
from blond.monitors.monitors import BunchMonitor
from blond.trackers.utilities import synchrotron_frequency_distribution
from blond.plots.plot import Plot
import cbfb
import coupled_bunch_diag as cbd
from blond.utils import bmath as bm

def run_cb_sim(params):
    #Create directory to store results:
    try:
        os.makedirs(params.output_dir)
    except:
        pass
    
    #Initialise variables to store data:     
    long_beam_signal = []
    cbfb_baseband_vec = []
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
    frequency_R = 2*rf.omega_rf[0,0] / 2.0 / np.pi
    resonator = Resonators(params.wake_R_S, frequency_R, params.wake_Q)    
    imp_list = [resonator]
    ind_volt_freq = InducedVoltageFreq(beam, tomo_profile, imp_list,
                                        frequency_resolution=5e4)
    total_ind_volt = TotalInducedVoltage(beam, tomo_profile, [ind_volt_freq])
    
    # BEAM GENERATION -------------------------------------------------------------
    distribution_options_list = {'bunch_length': params.bunch_length,
                                 'type': 'parabolic_amplitude',
                                 'density_variable': 'Hamiltonian'}
    
    matched_from_distribution_density_multibunch(beam, ring,
                                 full_tracker, distribution_options_list,
                                 params.n_bunches, params.bunch_spacing_buckets,
                                 intensity_list=params.intensity_list,
                                 minimum_n_macroparticles=params.minimum_n_macroparticles,
                                 TotalInducedVoltage=total_ind_volt,
                                 n_iterations_input=4, seed=7878)
    
    bunchmonitor = BunchMonitor(ring, rf, beam,
                                params.output_dir, Profile=tomo_profile)
    
    plots = Plot(ring, rf, beam, params.phase_plot_dt, params.N_t, 
                 2*params.profile_plot_bunch*np.pi,
                  2*(params.profile_plot_bunch + 1)*np.pi,
                  -params.phase_plot_max_dE, params.phase_plot_max_dE, xunit='rad', 
                  separatrix_plot=True,
                  Profile=tomo_profile,
                  sampling = 1,
                  format_options={'dirname': params.output_dir + 'profile_plots/'})
    
    
    #Set up coupled-bunch feedback parameters:
    fb = cbfb.feedback(beam, ring, full_tracker, params.cbfb_N_chans, params.cbfb_h_in, params.cbfb_h_out, \
                       params.cbfb_gain_vec, params.cbfb_active, params.cbfb_sideband_swap)
        
    fb_diag = cbd.coupled_bunch_diag(beam, ring, full_tracker, fb,\
                                     {'dirname': params.output_dir + 'cb_plots/'}, \
                                     params.harmonic_number, params.fb_diag_dt, params.fb_diag_plot_dt, params.N_t)
    
    # Accelerator map
    map_ = [full_tracker] + [tomo_profile] + [long_fft_profile] + \
        [total_ind_volt] + [bunchmonitor] + [fb] + [fb_diag] + [plots]
    print("Map set")
    print("")
    
    
    # Tracking --------------------------------------------------------------------
    turn_start_time = 0
    
    for turn in range(1, params.N_t+1):
        # Track
        for m in map_:
            m.track()
            
        #Apply excitation:
        particle_t = turn_start_time + beam.dt
        beam.dE += params.exc_v[turn] * bm.sin(2*np.pi*exc_freq*particle_t)
        turn_start_time += turn_length
        
        #Record CBFB ch1 baseband:
        cbfb_baseband_vec.append(fb.dipole_channels[0].cic_decim_out)
        
        if turn == params.start_cbfb_turn:
            for channel in range(params.cbfb_N_chans):
                if params.cbfb_active_mask[channel]:
                    fb.active[channel] = True
        
        if turn == params.end_cbfb_turn:
            for channel in range(params.cbfb_N_chans):
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
        plt.savefig(params.output_dir + 'beam_spectrum_h' + str(h) + '.png')
        plt.clf()
    
    plot_mask = (long_fft_freq >= 0) & (long_fft_freq < params.harmonic_number*f_rev)
    
    plt.figure('beam_spectrum_full', figsize=(8,6))
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(long_fft_freq[plot_mask], np.absolute(long_fft[plot_mask]), '-')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel('Beam spectrum, absolute value [arb. units]')
    plt.title('Beam spectrum')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(params.output_dir + 'beam_spectrum_full.png')
    plt.clf()
    
    # Plot tomograms
    plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*params.tomo_dt):params.tomo_dt, 0:params.tomo_n_slices]
    plt.pcolormesh(plot_x, plot_y, tomogram, cmap='hot', shading='flat')
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('Time [sample]')
    plt.ylabel('Turn')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig(params.output_dir + 'tomogram.png')
    plt.show()
    
    tomo_zoom_start = round(params.tomo_n_slices * params.profile_plot_bunch / params.harmonic_number)
    tomo_zoom_end = round(params.tomo_n_slices * (params.profile_plot_bunch+1) / params.harmonic_number)
    
    plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*params.tomo_dt):params.tomo_dt, tomo_zoom_start:tomo_zoom_end]
    plt.pcolormesh(plot_x, plot_y, tomogram[:, tomo_zoom_start:tomo_zoom_end], cmap='hot', shading='flat')
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('Time [sample]')
    plt.ylabel('Turn')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig(params.output_dir + 'tomogram_zoom.png')
    plt.show()
    

    #Coupled-bunch analysis
    
    #Sideband magnitudes:
    #Measure up to half of FFT sample rate:
    for h in range(fft_n_harmonics):
        rel_freq = long_fft_freq - f_rev * h
        
        dipole_usb_mag[h] = np.mean(np.absolute(long_fft[(rel_freq >= 0.5*abs(params.fs_exc)) & \
                                             (rel_freq < 1.5*abs(params.fs_exc))]))
        quad_usb_mag[h] = np.mean(np.absolute(long_fft[(rel_freq >= 0.5*2*abs(params.fs_exc)) & \
                                           (rel_freq < 1.5*2*abs(params.fs_exc))]))
        dipole_lsb_mag[h] = np.mean(np.absolute(long_fft[(rel_freq >= -1.5*abs(params.fs_exc)) & \
                                             (rel_freq < -0.5*abs(params.fs_exc))]))
        quad_lsb_mag[h] = np.mean(np.absolute(long_fft[(rel_freq >= -1.5*2*abs(params.fs_exc)) & \
                                           (rel_freq < -0.5*2*abs(params.fs_exc))]))
    
    plt.figure('cbfb_baseband')
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
            np.real(cbfb_baseband_vec[params.fb_diag_start_delay:]), '-')
    ax.plot([x for x in range(params.fb_diag_start_delay, params.N_t)], \
            np.imag(cbfb_baseband_vec[params.fb_diag_start_delay:]), '-')
    ax.set_xlabel("Turn")
    ax.set_ylabel('CBFB baseband signal [arb. units]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(params.output_dir + 'cbfb_baseband.png')
    plt.clf()
    
    #Plot bunch width and position history
    fb_diag.plot_size_width(params.fb_diag_start_delay, params.N_t)
    [pos_mode_spectrum, width_mode_spectrum] = fb_diag.mode_analysis(params.fft_start_turn, params.fft_end_turn)
    
    #Get magnitudes of desired modes:                             
    pos_mode_amp = [np.absolute(pos_mode_spectrum[m]) for m in range(params.harmonic_number)]
    width_mode_amp = [np.absolute(width_mode_spectrum[m]) for m in range(params.harmonic_number)]
    
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
    plt.savefig(params.output_dir + 'fs_distribution.png')
    
    print('Zero-amplitude synchrotron frequency : ' + str(sync_freq_distribution_left[0]) + ' Hz')
    
    return [dipole_usb_mag, dipole_lsb_mag, quad_usb_mag, quad_lsb_mag, pos_mode_amp, width_mode_amp]