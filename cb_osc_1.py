
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
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.beam import Beam, Proton
from blond.beam.distributions_multibunch \
                            import matched_from_distribution_density_multibunch
from blond.beam.distributions_multibunch import matched_from_line_density_multibunch
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import Resonators
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot, fig_folder
from scipy.constants import c, e, m_p
from blond.utils import bmath as bm
from blond.trackers.utilities import synchrotron_frequency_distribution
import os
import cbfb
import coupled_bunch_diag as cbd

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

# cbfb_phase_vals = np.linspace(0,1,16)
N_runs = 20
exc_amp = np.logspace(1, 5, N_runs) #[1e3]
dipole_sideband_mag = np.empty(N_runs)
quad_sideband_mag = np.empty(N_runs)
pos_amp = np.empty(N_runs)
width_amp =  np.empty(N_runs)
fs_exc = 387.29
dipole_exc = True

for run in range(N_runs):
    
    cbfb_gain = 1e-3
    # cbfb_phase = cbfb_phase_vals[run]
    cbfb_phase = 0.26
    # subdir = 'quad_gain_' + str(cbfb_gain) + '_phase_' + "{:.3f}".format(cbfb_phase)
    if dipole_exc:
        subdir = 'dipole_exc_h20_amp_' + str(exc_amp[run])
    else:
        subdir = 'quad_exc_h20_amp_' + str(exc_amp[run])
        
    full_dir = this_directory + 'output_files/' + subdir + '/'
    
    try:
        os.mkdir(full_dir)
    except:
        pass

    # SIMULATION PARAMETERS -------------------------------------------------------
    #Note: FFT assumes fixed frev!
    
    # Tracking details
    N_t = 10000 #15000    # Number of turns to track

    # Beam parameters
    n_particles = 1e10
    n_macroparticles = 1e3
    sync_momentum = 25.92e9 # [eV]
                            
    # Machine and RF parameters
    radius = 100.0
    gamma_transition = 6.1
    C = 2 * np.pi * radius  # [m]
    
    # Cavities parameters
    n_rf_systems = 1
    harmonic_number = 21
    voltage_program = 200e3
    phi_offset = 0
    
    #Wake impedance
    wake_R_S = 1#1e6
    wake_Q = 100
    
    #Beam parameters:
    n_bunches = 21
    bunch_spacing_buckets = 1
    bunch_intensity = 1e11
    bunch_length = 15e-9
    intensity_list = [bunch_intensity] * n_bunches
    minimum_n_macroparticles = [4e3] * n_bunches
    
    cbfb_N_chans = 1
    cbfb_h_in = [20]
    cbfb_h_out = [1]
    cbfb_active = [False]
    cbfb_sideband_swap = [True]
    
    cbfb_gain_vec = [np.zeros(N_t+1, complex)]
    cbfb_gain_vec[0][:] = cbfb_gain * np.exp(2j * np.pi * cbfb_phase)
    
    start_cbfb_turn = 10000
    end_cbfb_turn = 20000
    cbfb_active_mask = [True] #Only these channels get activated on SCBFB
    
    fb_diag_dt = 100
    fb_diag_start_delay = 100
    
    # Excitation parameters:
    exc_v = np.zeros(N_t+1)
    exc_v[0:7500] = exc_amp[run]
    exc_harmonic = 20
    
    if dipole_exc:
        exc_delta_freq = fs_exc
    else:
        exc_delta_freq = 2*fs_exc
         
    #Simulation parameters
    profile_plot_bunch = 0
    phase_plot_dt = 200
    phase_plot_max_dE = 100e6
    tomo_n_slices = 3000
    tomo_dt = 10
    fft_n_slices = 64
    fft_start_turn = 4000
    fft_end_turn = 10000 
    fft_span_around_harmonic = 2000
    fft_plot_harmonic = harmonic_number - exc_harmonic
    
    
    
    
    
    #Initialise variables to store data:     
    long_beam_signal = []
    cbfb_baseband_vec = []
    
    tomo_N_lines = int(np.ceil(N_t / tomo_dt) + 1)
    tomogram = np.empty([tomo_N_lines, tomo_n_slices])
        
    # Derived parameters
    E_0 = m_p * c**2 / e    # [eV]
    tot_beam_energy =  np.sqrt(sync_momentum**2 + E_0**2) # [eV]
    momentum_compaction = 1 / gamma_transition**2
    gamma = tot_beam_energy / E_0
    beta = np.sqrt(1.0-1.0/gamma**2.0)    
    momentum_compaction = 1 / gamma_transition**2
    
    #Define ring and RF systems:
    ring = Ring(C, momentum_compaction, sync_momentum, Proton(), N_t)
    beam = Beam(ring, n_macroparticles, n_particles)
    rf = RFStation(ring, [harmonic_number], [voltage_program],
                           [phi_offset], n_rf_systems)
    tracker = RingAndRFTracker(rf, beam)
    full_tracker = FullRingAndRF([tracker]) 
    
    bucket_length = rf.t_rf[0, 0]
    turn_length = harmonic_number * bucket_length
    f_rev = 1 / turn_length
    exc_freq = exc_harmonic * f_rev + exc_delta_freq
    fft_dt = turn_length / fft_n_slices
    
    tomo_profile = Profile(beam, CutOptions(cut_left=0, 
                        cut_right=turn_length, n_slices=tomo_n_slices))
    
    long_fft_profile = Profile(beam, CutOptions(cut_left=0, 
                        cut_right=turn_length, n_slices=fft_n_slices))
    
    # WAKE IMPEDANCE -------------------------------------------------------
    frequency_R = 2*rf.omega_rf[0,0] / 2.0 / np.pi
    resonator = Resonators(wake_R_S, frequency_R, wake_Q)    
    imp_list = [resonator]
    ind_volt_freq = InducedVoltageFreq(beam, tomo_profile, imp_list,
                                        frequency_resolution=5e4)
    total_ind_volt = TotalInducedVoltage(beam, tomo_profile, [ind_volt_freq])
    
    # BEAM GENERATION -------------------------------------------------------------
    distribution_options_list = {'bunch_length': bunch_length,
                                 'type': 'parabolic_amplitude',
                                 'density_variable': 'Hamiltonian'}
    
    matched_from_distribution_density_multibunch(beam, ring,
                                 full_tracker, distribution_options_list,
                                 n_bunches, bunch_spacing_buckets,
                                 intensity_list=intensity_list,
                                 minimum_n_macroparticles=minimum_n_macroparticles,
                                 TotalInducedVoltage=total_ind_volt,
                                 n_iterations_input=4, seed=7878)
    
    bunchmonitor = BunchMonitor(ring, rf, beam,
                                full_dir, Profile=tomo_profile)
    
    plots = Plot(ring, rf, beam, phase_plot_dt, N_t, 
                 2*profile_plot_bunch*np.pi,
                  2*(profile_plot_bunch + 1)*np.pi,
                  -phase_plot_max_dE, phase_plot_max_dE, xunit='rad', 
                  separatrix_plot=True,
                  Profile=tomo_profile,
                  sampling = 1,
                  format_options={'dirname': full_dir + 'profile_plots/'})
    
    
    #Set up coupled-bunch feedback parameters:
    fb = cbfb.feedback(beam, ring, full_tracker, cbfb_N_chans, cbfb_h_in, cbfb_h_out, \
                       cbfb_gain_vec, cbfb_active, cbfb_sideband_swap)
        
    fb_diag = cbd.coupled_bunch_diag(beam, ring, full_tracker, fb,\
                                     {'dirname': full_dir + 'cb_plots/'}, \
                                     harmonic_number, fb_diag_dt, N_t)
    
    # Accelerator map
    map_ = [full_tracker] + [tomo_profile] + [long_fft_profile] + \
        [total_ind_volt] + [bunchmonitor] + [fb] + [fb_diag] + [plots]
    print("Map set")
    print("")
    
    
    # Tracking --------------------------------------------------------------------
    turn_start_time = 0
    
    for turn in range(1, N_t+1):
        # Track
        for m in map_:
            m.track()
            
        #Apply excitation:
        particle_t = turn_start_time + beam.dt
        beam.dE += exc_v[turn] * bm.sin(2*np.pi*exc_freq*particle_t)
        turn_start_time += turn_length
        
        #Record CBFB ch1 baseband:
        cbfb_baseband_vec.append(fb.dipole_channels[0].cic_decim_out)
        
        if turn == start_cbfb_turn:
            for channel in range(cbfb_N_chans):
                if cbfb_active_mask[channel]:
                    fb.active[channel] = True
        
        if turn == end_cbfb_turn:
            for channel in range(cbfb_N_chans):
                fb.active[channel] = True
        
        #Collect data for long FFT
        if turn >= fft_start_turn and turn < fft_end_turn:
            long_beam_signal.extend(long_fft_profile.n_macroparticles)
            
        #Collect data for tomoscope
        if (turn % tomo_dt) == 0:
            tomo_line = int(turn / tomo_dt)
            tomogram[tomo_line, :] = tomo_profile.n_macroparticles
                
    print("Tracking Done")
                
    # Plot narrowband beam spectrum                    
    long_fft = bm.rfft(long_beam_signal * np.hanning(len(long_beam_signal)))
    long_fft_freq = bm.rfftfreq(len(long_beam_signal), fft_dt)
    
    print("FFT Calculated")
    
    fft_N = len(long_beam_signal)
    fft_T = fft_dt * fft_N
    fft_df = 1 / fft_T
    
    #Plot FFTs:
    f_plot_centre = f_rev * fft_plot_harmonic
    f_plot_min = f_plot_centre - fft_span_around_harmonic / 2
    f_plot_max = f_plot_centre + fft_span_around_harmonic / 2
    plot_mask = (long_fft_freq >= f_plot_min) & (long_fft_freq < f_plot_max)
    
    plt.figure('beam_spectrum', figsize=(8,6))
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(long_fft_freq[plot_mask] - f_plot_centre, np.absolute(long_fft[plot_mask]), '-')
    ax.set_xlabel("Relative frequency [Hz]")
    ax.set_ylabel('Beam spectrum, absolute value [arb. units]')
    plt.title('Beam spectrum around h' + str(f_plot_centre/f_rev))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(full_dir + 'beam_spectrum.png')
    plt.clf()
    
    plot_mask = (long_fft_freq >= 0) & (long_fft_freq < harmonic_number*f_rev)
    
    plt.figure('beam_spectrum_full', figsize=(8,6))
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(long_fft_freq[plot_mask], np.absolute(long_fft[plot_mask]), '-')
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel('Beam spectrum, absolute value [arb. units]')
    plt.title('Beam spectrum')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(full_dir + 'beam_spectrum_full.png')
    plt.clf()
    
    # Plot tomograms
    plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*tomo_dt):tomo_dt, 0:tomo_n_slices]
    plt.pcolormesh(plot_x, plot_y, tomogram, cmap='hot', shading='flat')
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('Time [s]')
    plt.ylabel('Turn')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig(full_dir + 'tomogram.png')
    plt.show()
    
    
    # tomo_zoom_slices = round(tomo_n_slices / harmonic_number)
    tomo_zoom_start = round(tomo_n_slices * (profile_plot_bunch-1) / harmonic_number)
    tomo_zoom_end = round(tomo_n_slices * profile_plot_bunch / harmonic_number)
    
    plot_y, plot_x = np.mgrid[0:((tomo_N_lines)*tomo_dt):tomo_dt, tomo_zoom_start:tomo_zoom_end]
    plt.pcolormesh(plot_x, plot_y, tomogram[:, tomo_zoom_start:tomo_zoom_end], cmap='hot', shading='flat')
    plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
    plt.xlabel('Time [s]')
    plt.ylabel('Turn')
    plt.colorbar()
    plt.rc('font', size=16)
    plt.savefig(full_dir + 'tomogram_zoom.png')
    plt.show()
    

    #Coupled-bunch analysis
    
    #Average both upper and lower sidebands:    
    rel_freq = long_fft_freq - f_rev * exc_harmonic #frequency relative to carrier
       
    dipole_usb = np.max(np.absolute(long_fft[(rel_freq >= 0.75*abs(fs_exc)) & \
                                             (rel_freq < 1.25*abs(fs_exc))]))
    dipole_lsb = np.max(np.absolute(long_fft[(rel_freq >= -1.25*abs(fs_exc)) & \
                                             (rel_freq < -0.75*abs(fs_exc))]))        
    quad_usb = np.max(np.absolute(long_fft[(rel_freq >= 0.75*2*abs(fs_exc)) & \
                                           (rel_freq < 1.25*2*abs(fs_exc))]))
    quad_lsb = np.max(np.absolute(long_fft[(rel_freq >= -1.25*2*abs(fs_exc)) & \
                                           (rel_freq < -0.75*2*abs(fs_exc))]))
    
    dipole_sideband_mag[run] = (dipole_usb + dipole_lsb) / 2    
    quad_sideband_mag[run] = (quad_usb + quad_lsb) / 2
    
    plt.figure('cbfb_baseband')
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot([x for x in range(fb_diag_start_delay, N_t)], \
            np.real(cbfb_baseband_vec[fb_diag_start_delay:]), '-')
    ax.plot([x for x in range(fb_diag_start_delay, N_t)], \
            np.imag(cbfb_baseband_vec[fb_diag_start_delay:]), '-')
    ax.set_xlabel("Turn")
    ax.set_ylabel('CBFB baseband signal [arb. units]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig(full_dir + 'cbfb_baseband.png')
    plt.clf()
    
    #Plot bunch width and position history
    fb_diag.plot_size_width(fb_diag_start_delay, N_t)
    [pos_mode_spectrum, width_mode_spectrum] = fb_diag.mode_analysis(fft_start_turn, fft_end_turn)
    
    #Average both upper and lower mode, as with sidebands:
    pos_amp[run] = (np.absolute(pos_mode_spectrum[exc_harmonic]) + \
                    np.absolute(pos_mode_spectrum[harmonic_number - exc_harmonic]))/2
    width_amp[run] = (np.absolute(width_mode_spectrum[exc_harmonic]) + \
                      np.absolute(width_mode_spectrum[harmonic_number - exc_harmonic]))/2
    
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
    phi = delta_time_left * 2.0 * np.pi / bucket_length
    plt.legend(loc=0, fontsize='medium')
    plt.xlabel('Amplitude of particle oscillations [s]')
    plt.ylabel('Synchrotron frequency [Hz]')
    plt.title('Synchrotron frequency distribution, no induced voltage')
    plt.savefig(full_dir + 'fs_distribution.png')
    
    print('Zero-amplitude synchrotron frequency : ' + str(sync_freq_distribution_left[0]) + ' Hz') 
    print('Run '  + str(run) + ' finished.')
    
    
cb_data = {'exc_harmonic' : exc_harmonic, 
           'exc_delta_freq' : exc_delta_freq,
           'exc_amp' : exc_amp,
           'dipole_sideband_mag' : dipole_sideband_mag,
           'quad_sideband_mag' : quad_sideband_mag,
           'pos_amp' : pos_amp,
           'width_amp' : width_amp}
    
if dipole_exc:
    with open(this_directory + 'output_files/' + 'dipole_exc_scan.pickle', 'wb') as f:
        pickle.dump(cb_data, f)
else:
    with open(this_directory + 'output_files/' + 'quad_exc_scan.pickle', 'wb') as f:
        pickle.dump(cb_data, f)



with open(this_directory + 'output_files/' + 'dipole_exc_scan.pickle', 'rb') as f:
    dipole_data = pickle.load(f)

with open(this_directory + 'output_files/' + 'quad_exc_scan.pickle', 'rb') as f:
    quad_data = pickle.load(f)

plt.figure('exc_amp_vs_sideband')
plt.loglog(dipole_data['exc_amp'], dipole_data['dipole_sideband_mag'], label = 'Dipole Exc, Dipole Sideband')
plt.loglog(dipole_data['exc_amp'], dipole_data['quad_sideband_mag'], label = 'Dipole Exc, Quad Sideband')
plt.loglog(quad_data['exc_amp'], quad_data['dipole_sideband_mag'], label = 'Quad Exc, Dipole Sideband')
plt.loglog(quad_data['exc_amp'], quad_data['quad_sideband_mag'], label = 'Quad Exc, Quad Sideband')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Excitation amplitude [V]')
plt.ylabel('Sideband magnitude [arb. units.]')
plt.savefig(this_directory + 'output_files/' + 'exc_amp_vs_sideband.png')

plt.figure('exc_amp_vs_osc_amp')
plt.loglog(dipole_data['exc_amp'], dipole_data['pos_amp'], label = 'Dipole Exc, Position Osc')
plt.loglog(dipole_data['exc_amp'], dipole_data['width_amp'], label = 'Dipole Exc, Width Osc')
plt.loglog(quad_data['exc_amp'], quad_data['pos_amp'], label = 'Quad Exc, Position Osc')
plt.loglog(quad_data['exc_amp'], quad_data['width_amp'], label = 'Quad Exc, Width Osc')
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Excitation amplitude [V]')
plt.ylabel('Oscillation amplitude [s]')
plt.savefig(this_directory + 'output_files/' + 'exc_amp_vs_osc_amp.png')

plt.figure('dipole_quad_ratio')
plt.semilogx(dipole_data['exc_amp'], \
           np.divide(dipole_data['dipole_sideband_mag'], quad_data['quad_sideband_mag']), \
               label = 'Sideband magnitude')
plt.semilogx(dipole_data['exc_amp'], \
           np.divide(dipole_data['pos_amp'], quad_data['width_amp']), \
               label = 'Bunch position and width')
plt.ylim(bottom=0)
plt.legend(loc=0, fontsize='medium')
plt.xlabel('Excitation amplitude [V]')
plt.ylabel('Ratio of dipole to quadrupole oscillation')
plt.savefig(this_directory + 'output_files/' + 'dipole_vs_quad_ratio.png')