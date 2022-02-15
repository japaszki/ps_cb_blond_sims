# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:18:45 2022

@author: JohnG
"""

from blond.beam.profile import Profile, CutOptions
from blond.utils import bmath as bm
import numpy as np
import cic
import iir
import cavity_model

class feedback:
    def __init__(self, beam, ring, tracker, cbfb_params, rf_params):
        self.N_channels = cbfb_params['N_channels']
        self.h_in = cbfb_params['h_in']
        self.h_out = cbfb_params['h_out']
        self.sideband_swap = cbfb_params['sideband_swap']
        self.active = cbfb_params['active']
        self.gain = cbfb_params['gain']
        
        self.tracker = tracker
        self.ring = ring
        self.beam = beam
        
        #Sampling of beam signal:
        self.h_samp = 64
        turn = self.tracker.RingAndRFSection_list[0].counter
        self.profile = Profile(self.beam, CutOptions(cut_left=0, 
                    cut_right=ring.t_rev[turn], n_slices=self.h_samp))
        
        #Initialise each channel:
        self.dipole_channels = []
        for i in range(self.N_channels):
            self.dipole_channels.append(dipole_channel(self.h_in[i], self.h_out[i], self.gain[i], self.sideband_swap[i]))
        
        #Initialise Finement cavity model:
        self.finemet = cavity_model.cavity_model_fir(rf_params)
        
        #Tracking of absolute time:
        self.turn_start_time = 0
        
    def track(self):
        #Get current turn from tracker:
        turn = self.tracker.RingAndRFSection_list[0].counter[0]
        turn_length = self.ring.t_rev[turn]
        
        #Simulate sampling with beam-synchronous clock:
        self.profile.cut_right = turn_length
        self.profile.set_slices_parameters()
        self.profile.track()
        self.beam_signal = self.beam.ratio * self.profile.n_macroparticles
        self.sample_dt = np.linspace(0, turn_length * (self.h_samp - 1) / self.h_samp, self.h_samp)
        
        #Update each channel:
        self.output_sum = np.zeros(self.h_samp)
        self.output_sum_all_chans = np.zeros(self.h_samp)
        for i in range(self.N_channels):
            channel_output = self.dipole_channels[i].update_output(turn, self.beam_signal)
            
            #Sum outputs of all channels for diagnotics:
            self.output_sum_all_chans += channel_output
            
            #Sum outputs of all active channels:
            if self.active[i]:
                self.output_sum += channel_output
                
        #Update Finemet model:
        self.finemet.update(self.output_sum, self.turn_start_time + self.sample_dt)
        
        #Perform cubic interpolation of h64 output to simulate finite bandwidth of high power system:
        # self.output_spline = scipy.interpolate.CubicSpline(self.sample_dt, self.cbfb_output_sum)
        # self.output_spline_all_chans = scipy.interpolate.CubicSpline(self.sample_dt, self.cbfb_output_sum_all_chans)
        
        #Ignore particles not within this turn:
        # particle_mask = (self.beam.dt >= 0) & (self.beam.dt < turn_length)
        
        #Apply kick to particles:
        # self.beam.dE[particle_mask] += self.output_spline(self.beam.dt[particle_mask])
        

        #Get part of finemet output history corresponding to current turn:
        # finemet_out_dt = self.finemet.output_hist_t * self.finemet.dt - self.turn_start_time - self.rf_output_delay
        # finemet_out_turn_mask = ~((finemet_out_dt < 0) | (finemet_out_dt > turn_length))

        # self.finemet_out_turn_dt = finemet_out_dt[finemet_out_turn_mask]
        # self.finemet_out_turn_v = self.finemet.output_hist_v[finemet_out_turn_mask]
        
        [self.finemet_dt, self.finemet_v] = self.finemet.get_output_in_window([self.turn_start_time, self.turn_start_time + turn_length])
        
        #Apply kick to particles:
        bm.linear_interp_kick(dt=self.beam.dt, dE=self.beam.dE,
                              voltage=self.finemet_v,
                              bin_centers=self.finemet_dt,
                              charge=self.beam.Particle.charge,
                              acceleration_kick=0.)
        
        #Tracking of absolute time:
        self.turn_start_time += turn_length
        

class dipole_channel:
    def __init__(self, h_in, h_out, gain, sideband_swap):        
        self.h_in = h_in
        self.h_out = h_out
        self.sideband_swap = sideband_swap
        self.gain = gain
        
        #IIR notch filter properties:
        self.hpf_i = iir.iir_hpf(1024, 1)
        self.hpf_q = iir.iir_hpf(1024, 1)
        
        #Baseband filter properties:
        self.cic_n = 2
        self.cic_r = 64
        self.cic_m = 4
        
        self.decim_i = cic.mov_avg_decim(self.cic_n, self.cic_r, self.cic_m)
        self.decim_q = cic.mov_avg_decim(self.cic_n, self.cic_r, self.cic_m)
        self.interp_i = cic.mov_avg_interp(self.cic_n, self.cic_r, self.cic_m)
        self.interp_q = cic.mov_avg_interp(self.cic_n, self.cic_r, self.cic_m)
        self.cic_decim_out = 0
        
        self.h_samp = 64
        
    def update_output(self, turn, beam_signal):
        #Generate NCO phase values for 1 turn:
        self.nco_phase = 2*np.pi*np.arange(self.h_samp)/self.h_samp
        
        #Downmix input signal at selected harmonic:
        self.downmix_lo_i = np.cos(self.h_in * self.nco_phase)
        self.downmix_lo_q = np.sin(self.h_in * self.nco_phase)
        self.downmix_out_i = bm.mul(beam_signal, self.downmix_lo_i)
        self.downmix_out_q = bm.mul(beam_signal, self.downmix_lo_q)
        
        #Notch out f_rev harmonic:
        hpf_out_i = np.zeros(self.h_samp)
        hpf_out_q = np.zeros(self.h_samp)
        for i in range(self.h_samp):
            hpf_out_i[i] = self.hpf_i.update(self.downmix_out_i[i])
            hpf_out_q[i] = self.hpf_q.update(self.downmix_out_q[i])
        
        #Low-pass filter and decimate mixer output:
        #Note, by coincidence h_samp and cic_r are equal, hence output is a scalar
        #I.e., one baseband sample per turn.
        cic_decim_out_i = self.decim_i.update(hpf_out_i)
        cic_decim_out_q = self.decim_q.update(hpf_out_q)
        
        self.cic_decim_out = cic_decim_out_i + 1j * cic_decim_out_q
        
        #Apply gain control:
        gain_ctrl_out = self.cic_decim_out * self.gain[turn]
        
        #Interpolate back to initial sample rate:
        interp_out_i = self.interp_i.update(np.real(gain_ctrl_out))
        interp_out_q = self.interp_q.update(np.imag(gain_ctrl_out))
        
        #Upmix to output harmonic number:
        self.upmix_lo_i = np.cos(self.h_out * self.nco_phase)
        self.upmix_lo_q = np.sin(self.h_out * self.nco_phase)
        
        if self.sideband_swap:
            self.upmix_out = bm.add(bm.mul(interp_out_i, self.upmix_lo_q), \
                               bm.mul(interp_out_q, self.upmix_lo_i))
        else:
            self.upmix_out = bm.add(bm.mul(interp_out_i, self.upmix_lo_i), \
                               bm.mul(interp_out_q, self.upmix_lo_q))
        
        return self.upmix_out