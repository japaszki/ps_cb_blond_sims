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
import peak_filter


class feedback_peak_det:
    def __init__(self, beam, ring, tracker, N_channels, h_in, h_out, gain, active, sideband_swap):
        self.N_channels = N_channels
        self.h_in = h_in
        self.h_out = h_out
        self.sideband_swap = sideband_swap
        self.active = active
        self.gain = gain
        
        self.tracker = tracker
        self.ring = ring
        self.beam = beam
        
        #Sampling of beam signal:
        self.h_samp_pkdet = 256
        self.pkdet_decim = 4
        turn = self.tracker.RingAndRFSection_list[0].counter
        self.profile = Profile(self.beam, CutOptions(cut_left=0, 
                    cut_right=ring.t_rev[turn], n_slices=self.h_samp_pkdet))
        
        #Initialise peak detector:
        self.peak_filter = peak_filter.peak_filter(2)
        
        #Initialise each channel:
        self.h_samp = 64
        self.dipole_channels = []
        for i in range(self.N_channels):
            self.dipole_channels.append(dipole_channel(h_in[i], h_out[i], gain[i], sideband_swap[i]))
        
    def track(self):
        #Get current turn from tracker:
        turn = self.tracker.RingAndRFSection_list[0].counter[0]
        
        #Simulate sampling with beam-synchronous clock:
        self.profile.cut_right = self.ring.t_rev[turn]
        self.profile.track()
        self.beam_signal = self.beam.ratio * self.profile.n_macroparticles
        
        #Decimate with peak detection:
        for i in range(self.h_samp):
            self.beam_signal_pkdet[i] = self.peak_filter.update(self.beam_signal[self.pkdet_decim*i:self.pkdet_decim*(i+1)])
        
        #Update each channel:
        self.cbfb_output_sum = np.zeros(self.h_samp)
        for i in range(self.N_channels):
            cbfb_channel_output = self.dipole_channels[i].update_output(turn, self.beam_signal_pkdet)
            
            #Sum outputs of all active channels:
            if self.active[i]:
                self.cbfb_output_sum += cbfb_channel_output
            
        #Rasterise particle times to sample clock:
        particle_time_sampled = np.floor(self.beam.dt * self.h_samp / self.ring.t_rev[turn])
        time_indices = particle_time_sampled.astype(int)
        indices_mask = (time_indices >= 0) & (time_indices < self.h_samp) #only apply kick to particles within turn
            
        #Apply kick to particles:
        self.beam.dE[indices_mask] += self.cbfb_output_sum[time_indices[indices_mask]]
        

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
        nco_phase = 2*np.pi*np.arange(self.h_samp)/self.h_samp
        
        #Downmix input signal at selected harmonic:
        self.downmix_lo_i = np.cos(self.h_in * nco_phase)
        self.downmix_lo_q = np.sin(self.h_in * nco_phase)
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
        upmix_lo_i = np.cos(self.h_out * nco_phase)
        upmix_lo_q = np.sin(self.h_out * nco_phase)
        
        if self.sideband_swap:
            self.upmix_out = bm.add(bm.mul(interp_out_i, upmix_lo_q), \
                               bm.mul(interp_out_q, upmix_lo_i))
        else:
            self.upmix_out = bm.add(bm.mul(interp_out_i, upmix_lo_i), \
                               bm.mul(interp_out_q, upmix_lo_q))
        
        return self.upmix_out
        