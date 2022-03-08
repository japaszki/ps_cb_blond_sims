# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:18:45 2022

@author: JohnG
"""

from blond.beam.profile import Profile, CutOptions
from blond.utils import bmath as bm
import numpy as np
import scipy.interpolate
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
        self.pre_filter = cbfb_params['pre_filter']
        self.post_filter = cbfb_params['post_filter']
        
        self.tracker = tracker
        self.ring = ring
        self.beam = beam
        
        #Sampling of beam signal:
        if self.pre_filter == 'none':
            self.h_samp_adc = 64
        elif self.pre_filter == 'peak':
            self.samples_per_bucket = 40
            self.h_samp_adc = 21 * self.samples_per_bucket
            self.bucket_max = np.zeros(21)
        elif self.pre_filter == 'width':
            self.h_samp_adc = 64
            
        turn = self.tracker.RingAndRFSection_list[0].counter
        self.profile = Profile(self.beam, CutOptions(cut_left=0, 
                    cut_right=ring.t_rev[turn], n_slices=self.h_samp_adc))
        
        if self.post_filter == 'none':
            self.h_samp_dsp = 64
        elif self.post_filter == 'h21_mod_h64_clk':
            self.h_samp_dsp = 64
        elif self.post_filter == 'h21_mod_h256_clk':
            self.h_samp_dsp = 256
        
        #Initialise each DSP channel:
        self.dipole_channels = []
        for i in range(self.N_channels):
            self.dipole_channels.append(dipole_channel(self.h_samp_dsp, self.h_in[i],\
                                    self.h_out[i], self.gain[i], self.sideband_swap[i]))
        
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
        self.adc_sample_dt = np.linspace(0, turn_length * (self.h_samp_adc - 1) / self.h_samp_adc, self.h_samp_adc)
        self.dsp_sample_dt = np.linspace(0, turn_length * (self.h_samp_dsp - 1) / self.h_samp_dsp, self.h_samp_dsp)
        
        #Perform pre-filtering of beam signal:
        if self.pre_filter == 'none':
            self.beam_signal_filt = self.beam_signal
        elif self.pre_filter == 'peak':
            #Perform peak detection per bucket:
            for i in range(21):
                bucket_indices = range(i*self.samples_per_bucket, (i+1)*self.samples_per_bucket)
                self.bucket_max[i] = np.max(self.beam_signal[bucket_indices])
            
            #Resample to DSP sample rate:
            bucket_dt = np.linspace(0, turn_length * 20 / 21, 21)
            interp_spline = scipy.interpolate.CubicSpline(bucket_dt, self.bucket_max)
            self.beam_signal_filt = interp_spline(self.dsp_sample_dt)
        elif self.pre_filter == 'width':
            bucket_length = turn_length / 21
            
            #Measure bunch width in each bucket:
            for i in range(21):
                particle_indices = (self.beam.dt >= i * bucket_length) & (self.beam.dt < (i+1) * bucket_length)
                self.bucket_width[i] = bm.std(self.beam.dt[particle_indices])
                
            #Resample to DSP sample rate:
            bucket_dt = np.linspace(0, turn_length * 20 / 21, 21)
            interp_spline = scipy.interpolate.CubicSpline(bucket_dt, self.bucket_width)
            self.beam_signal_filt = interp_spline(self.dsp_sample_dt)
    
        #Update each channel:
        #Note that the DSP channels expect an input of h_samp_dsp samples per turn
        #If beam is sampled with a different sample rate, the pre-filter has to re-sample the signal appropriately.
        
        self.output_sum = np.zeros(self.h_samp_dsp)
        self.output_sum_all_chans = np.zeros(self.h_samp_dsp)
        for i in range(self.N_channels):
            channel_output = self.dipole_channels[i].update_output(turn, self.beam_signal_filt)
            
            #Sum outputs of all channels for diagnotics:
            self.output_sum_all_chans += channel_output
            
            #Sum outputs of all active channels:
            if self.active[i]:
                self.output_sum += channel_output
           
        #Perform post_filtering of output signal:
        if self.post_filter == 'none':
            self.output_sum_filt = self.output_sum
        elif self.post_filter == 'h21_mod_h64_clk' or self.post_filter == 'h21_mod_h256_clk':
            nco_phase = 2*np.pi*np.arange(self.h_samp_dsp)/self.h_samp_dsp
            self.output_sum_filt = bm.mul(self.output_sum, np.sin(21 * nco_phase))
            
        #Update Finemet model:
        self.finemet.update(self.output_sum_filt, self.turn_start_time + self.dsp_sample_dt)
        [self.finemet_dt, self.finemet_v] = self.finemet.get_output_in_window([self.turn_start_time, \
                                                                               self.turn_start_time + turn_length])
        
        #Apply kick to particles:
        bm.linear_interp_kick(dt=self.beam.dt, dE=self.beam.dE,
                              voltage=self.finemet_v,
                              bin_centers=self.finemet_dt,
                              charge=self.beam.Particle.charge,
                              acceleration_kick=0.)
        
        #Tracking of absolute time:
        self.turn_start_time += turn_length
        

class dipole_channel:
    def __init__(self, h_samp, h_in, h_out, gain, sideband_swap):        
        self.h_in = h_in
        self.h_out = h_out
        self.sideband_swap = sideband_swap
        self.gain = gain
        
        #IIR notch filter properties:
        #scale transfer function according to sample rate
        self.hpf_i = iir.iir_hpf(16*h_samp, 1)
        self.hpf_q = iir.iir_hpf(16*h_samp, 1)
        
        #Baseband filter properties:
        self.cic_n = 2
        self.cic_r = h_samp
        self.cic_m = 4
        
        self.decim_i = cic.mov_avg_decim(self.cic_n, self.cic_r, self.cic_m)
        self.decim_q = cic.mov_avg_decim(self.cic_n, self.cic_r, self.cic_m)
        self.interp_i = cic.mov_avg_interp(self.cic_n, self.cic_r, self.cic_m)
        self.interp_q = cic.mov_avg_interp(self.cic_n, self.cic_r, self.cic_m)
        self.cic_decim_out = 0
        
        self.h_samp = h_samp
        
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