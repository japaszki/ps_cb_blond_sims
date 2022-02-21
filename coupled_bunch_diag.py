# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:11:12 2022

@author: JohnG
"""

import pylab as plt
import numpy as np
from blond.utils import bmath as bm
from blond.plots.plot import fig_folder
import scipy.interpolate

def sine(t, amplitude, frequency, phase):
    return amplitude * np.sin(2*np.pi*frequency*t + phase)

def sine_fit(xdata, ydata, dt):
    #Remove DC offset:
    ydata_nodc = ydata - bm.mean(ydata)

    #Initial guess for amplitude:
    amp_guess = (max(ydata_nodc) - min(ydata_nodc))/2
    
    #Initial guess for frequency peak value of FFT:
    fft_N = len(ydata_nodc)
    fft = bm.rfft(ydata_nodc)
    fft_freq = bm.rfftfreq(fft_N, dt)
    freq_guess = fft_freq[bm.argmax(np.absolute(fft))]
        
    #Scan phase and frequency values for best fit to data:
    N_ph = 200
    N_freq = 20
    error = np.empty([N_ph, N_freq])
    ph = np.linspace(0, 2*np.pi, N_ph)
    #Frequency scan range is plus or minus one FFT bin from FFT peak:
    freq = np.linspace(freq_guess - fft_freq[1], freq_guess + fft_freq[1], N_freq)
    for j in range(N_freq):
        for k in range(N_ph):
            sine_vals = sine(xdata, amp_guess, freq[j], ph[k])
            error[k,j] = np.sum(np.power(sine_vals - ydata_nodc, 2))
        
    #Find phase and frequency that gave least sum of squares error:
    fit_indices = np.unravel_index(error.argmin(), error.shape)
    ph_fit = ph[fit_indices[0]]
    freq_fit = freq[fit_indices[1]]
    #Use guessed amplitude as final amplitude for now:
    
    #Fit amplitude:
    N_amp = 50
    error = np.empty(N_amp)
    amp = amp_guess * np.logspace(-0.5, 0.1, N_amp)
    for j in range(N_amp):
        sine_vals = sine(xdata, amp[j], freq_fit, ph_fit)
        error[j] = np.sum(np.power(sine_vals - ydata_nodc, 2))
        
    amp_fit = amp[error.argmin()]
    
    return [amp_fit, freq_fit, ph_fit]


def bunch_statistics(dt, dE, method='mean_std'):
    if dt.size == 0 or dE.size == 0:
        return {'dt_mean' : 0, 'dt_std' : 0, \
                  'dE_mean' : 0, 'dE_std' : 0}
    else:
        dt_mean = bm.mean(dt)
        dt_std = bm.std(dt)
        dE_mean = bm.mean(dE)
        dE_std = bm.std(dE)
    
        if method == 'mean_std':
            result = {'dt_mean' : dt_mean, 'dt_std' : dt_std, \
                      'dE_mean' : dE_mean, 'dE_std' : dE_std}    
            
        return result

def plot_modes_vs_time(window_centres, pos_modes, width_modes, harmonic_number, N_modes_plt, dirname):
    #Find dominant oscillation modes:
    pos_dom_modes = np.argsort(np.amax(pos_modes, axis=1))
    width_dom_modes = np.argsort(np.amax(width_modes, axis=1))
    
    #Plot mode amplitudes vs time:
    if N_modes_plt > 0:
        plt.figure('pos_modes_vs_turn', figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        for i in range(1,N_modes_plt+1):
            ax.plot(window_centres, pos_modes[pos_dom_modes[-i], :], label = 'Mode ' + str(pos_dom_modes[-i]))
        ax.plot(window_centres, np.sum(pos_modes[pos_dom_modes[0:(harmonic_number-N_modes_plt)], :], axis=0), \
                label = 'Remaining modes')
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mode magnitude [s]")
        plt.title('Bunch position modes')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(dirname + '/pos_modes_vs_turn.png')
        plt.close()
        
        plt.figure('width_modes_vs_turn', figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        for i in range(1,N_modes_plt+1):
            ax.plot(window_centres, width_modes[width_dom_modes[-i], :], label = 'Mode ' + str(width_dom_modes[-i]))
        ax.plot(window_centres, np.sum(width_modes[width_dom_modes[0:(harmonic_number-N_modes_plt)], :], axis=0), \
                label = 'Remaining modes')
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mode magnitude [s]")
        plt.title('Bunch width modes')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.legend(loc=0, fontsize='medium')
        plt.savefig(dirname + '/width_modes_vs_turn.png')
        plt.close()


class coupled_bunch_diag:
    def __init__(self, beam, ring, tracker, cbfb, format_options, harmonic_number, dt, plot_dt, N_t):
        self.beam = beam
        self.ring = ring
        self.tracker = tracker
        self.cbfb = cbfb
        self.harmonic_number = harmonic_number
        self.dt = dt
        self.plot_dt = plot_dt
        self.N_t = N_t
        
        N_lines = int(np.ceil(N_t / dt) + 1)
        self.turn_vec = np.array([self.dt * i for i in range(N_lines)])
        self.bunch_pos_dt = np.empty([harmonic_number, N_lines])
        self.bunch_pos_dE = np.empty([harmonic_number, N_lines])
        self.bunch_width_dt = np.empty([harmonic_number, N_lines])
        self.bunch_width_dE = np.empty([harmonic_number, N_lines])
        
        if format_options == None:
            format_options = {'dummy': 0} 

        if 'dirname' not in format_options:  
            self.dirname = 'fig'
        else: 
            self.dirname = format_options['dirname']
            
        fig_folder(self.dirname)
            
    def track(self):
        #Get current turn from tracker:
        turn = self.tracker.RingAndRFSection_list[0].counter[0]
        bucket_length = self.ring.t_rev[turn] / self.harmonic_number

        #Record data if sampling on this turn:
        if (turn % self.dt) == 0:
            line = int(turn / self.dt)
            
            for i in range(self.harmonic_number):
                #Pick particles in this bucket:
                particle_indices = (self.beam.dt >= i * bucket_length) & (self.beam.dt < (i+1) * bucket_length)
                
                #Obtain statistics of each bunch:
                bunch_result = bunch_statistics(self.beam.dt[particle_indices], \
                                                self.beam.dE[particle_indices], method='mean_std')
                self.bunch_pos_dt[i, line] = bunch_result['dt_mean']
                self.bunch_width_dt[i, line] = 2 * bunch_result['dt_std']
                self.bunch_pos_dE[i, line] = bunch_result['dE_mean']
                self.bunch_width_dE[i, line] = 2 * bunch_result['dE_std']
        
        if (turn % self.plot_dt) == 0:
            line = int(turn / self.dt)
            
            #Fit cubic spline to all channel sum:
            output_spline_all_chans = scipy.interpolate.CubicSpline(self.cbfb.dsp_sample_dt, self.cbfb.output_sum_all_chans)
                   
            #Plot bunch energy deviation vs kick:
            plt.figure('bunch_dE', figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
            ax.plot(self.bunch_pos_dt[:, line], self.bunch_pos_dE[:,line], label='Mean bunch energy offset')
            ax.plot(self.bunch_pos_dt[:, line], 100*output_spline_all_chans(self.bunch_pos_dt[:, line]),\
                    label='100x kick at bunch centre')
            ax.plot(self.cbfb.finemet_dt, 100*self.cbfb.finemet_v, label='100x cavity voltage')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("dE [eV]")
            plt.title('Turn = ' + str(turn))
            ax.legend()
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(self.dirname + '/bunch_dE_turn_' + str(turn) + '.png')
            plt.close()
            
            #Plot bunch energy spread vs kick gradient:                
            plt.figure('bunch_sE', figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
            ax.plot(self.bunch_pos_dt[:, line], \
                    (self.bunch_width_dE[:,line] - np.mean(self.bunch_width_dE[:,line])),\
                        label='Bunch energy spread deviation')
            ax.plot(self.bunch_pos_dt[:, line], 1e-6*output_spline_all_chans(self.bunch_pos_dt[:, line], 1),\
                    label='1e-6 x kick gradient at bunch centre')
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("dE [eV]")
            plt.title('Turn = ' + str(turn))
            ax.legend()
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(self.dirname + '/bunch_sigmaE_turn_' + str(turn) + '.png')
            plt.close()
    
    def plot_size_width(self, start_turn, end_turn): 
        turn_indices = (self.turn_vec >= start_turn) & (self.turn_vec < end_turn)
        bunch_mean_pos_dt = [bm.mean(self.bunch_pos_dt[i,turn_indices]) for i in range(self.harmonic_number)]
        
        #Plot bunch position vs time:
        for i in range(self.harmonic_number):
            plt.figure('bunch_pos', figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(self.turn_vec[turn_indices], self.bunch_pos_dt[i,turn_indices] - bunch_mean_pos_dt[i])
            ax.set_xlabel("Turn")
            ax.set_ylabel("Bunch relative position")
            plt.title('Bunch ' + str(i))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(self.dirname + '/bunch_' + str(i) + '_offset.png')
            plt.close()
        
        #Plot bunch width vs time:
        for i in range(self.harmonic_number):
            plt.figure('bunch_width', figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(self.turn_vec[turn_indices], self.bunch_width_dt[i,turn_indices])
            ax.set_xlabel("Turn")
            ax.set_ylabel("Bunch width")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.title('Bunch ' + str(i))        
            plt.savefig(self.dirname + '/bunch_' + str(i) + '_width.png')
            plt.close()
        
    def mode_analysis(self, start_turn, end_turn, plots, method='fft'):
        #fit to sinousoid of constant amplitude and frequency.
        #output fitted frequency
        #separate analysis on bunch widths and positions.
        turn_indices = (self.turn_vec >= start_turn) & (self.turn_vec < end_turn)
        
        pos_amp_fit = np.empty(self.harmonic_number)
        pos_freq_fit = np.empty(self.harmonic_number)
        pos_ph_fit = np.empty(self.harmonic_number)
        width_amp_fit = np.empty(self.harmonic_number)
        width_freq_fit = np.empty(self.harmonic_number)
        width_ph_fit = np.empty(self.harmonic_number)
        
        if method == 'sine_fit':
            #Fit sinusoids to bunch positions and widths:
            for i in range(self.harmonic_number):
                pos_data = self.bunch_pos_dt[i,turn_indices]
                width_data = self.bunch_width_dt[i,turn_indices]
                
                [pos_amp_fit[i], pos_freq_fit[i], pos_ph_fit[i]] = sine_fit(self.turn_vec[turn_indices], pos_data, self.dt)
                [width_amp_fit[i], width_freq_fit[i], width_ph_fit[i]] = sine_fit(self.turn_vec[turn_indices], width_data, self.dt)
                
                if plots:
                    plt.figure('bunch_pos_fit', figsize=(8,6))
                    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
                    ax.plot(self.turn_vec[turn_indices], pos_data, label='Data')
                    ax.plot(self.turn_vec[turn_indices], \
                            sine(self.turn_vec[turn_indices], pos_amp_fit[i], pos_freq_fit[i], pos_ph_fit[i]) +\
                                bm.mean(pos_data), label='Fit')
                    ax.set_xlabel("Turn")
                    ax.set_ylabel("Bunch relative position")
                    plt.legend(loc=0, fontsize='medium')
                    plt.title('Bunch ' + str(i))
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    plt.savefig(self.dirname + '/bunch_' + str(i) + '_offset_fit.png')
                    plt.close()
                    
                    plt.figure('bunch_width_fit', figsize=(8,6))
                    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
                    ax.plot(self.turn_vec[turn_indices], width_data, label='Data')
                    ax.plot(self.turn_vec[turn_indices], \
                            sine(self.turn_vec[turn_indices], width_amp_fit[i], width_freq_fit[i], width_ph_fit[i]) +\
                            bm.mean(width_data), label='Fit')
                    ax.set_xlabel("Turn")
                    ax.set_ylabel("Bunch width")
                    plt.legend(loc=0, fontsize='medium')
                    plt.title('Bunch ' + str(i))
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    plt.savefig(self.dirname + '/bunch_' + str(i) + '_width_fit.png')
                    plt.close()
              
            #Take FFT of fitted amplitude and phase values:
            pos_complex = np.multiply(pos_amp_fit, np.exp(1j * pos_ph_fit))
            width_complex = np.multiply(width_amp_fit, np.exp(1j * width_ph_fit))
            
            pos_mode_spectrum = np.fft.fft(pos_complex) / self.harmonic_number
            width_mode_spectrum = np.fft.fft(width_complex) / self.harmonic_number
            
            if plots:
                plt.figure('bunch_pos_phases', figsize=(8,6))
                ax = plt.axes([0.15, 0.1, 0.8, 0.8])
                ax.plot(pos_ph_fit)
                ax.set_xlabel("Bunch")
                ax.set_ylabel("Phase [rad]")
                plt.title('Bunch position oscillation')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(self.dirname + '/bunch_pos_phases.png')
                plt.close()
                
                plt.figure('bunch_width_phases', figsize=(8,6))
                ax = plt.axes([0.15, 0.1, 0.8, 0.8])
                ax.plot(pos_ph_fit)
                ax.set_xlabel("Bunch")
                ax.set_ylabel("Phase [rad]")
                plt.title('Bunch width oscillation')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.savefig(self.dirname + '/bunch_width_phases.png')
                plt.close()
            
        elif method == 'fft':
            #Calculate time span of FFT window and thus frequency resolution:
            turn_span = np.max(self.turn_vec[turn_indices]) - np.min(self.turn_vec[turn_indices])
            df = 1 / turn_span
            
            #Subtract mean position from each bunch position:
            bunch_rel_pos = np.empty_like(self.bunch_pos_dt[:,turn_indices])
            for bunch in range(self.harmonic_number):
                bunch_rel_pos[bunch,:] = np.hanning(self.bunch_pos_dt[bunch,turn_indices].shape[0]) * \
                    (self.bunch_pos_dt[bunch,turn_indices] - np.mean(self.bunch_pos_dt[bunch,turn_indices]))
                        
            #Calculate normalised 2D FFT:
            fft_pos = np.fft.rfft2(bunch_rel_pos) * 4 / np.size(bunch_rel_pos)

            #Find frequency peak, estimate of synchrotron frequency:
            max_pos_freq_index = np.argmax(np.mean(np.abs(fft_pos), axis=0))
            pos_mode_spectrum = fft_pos[:, max_pos_freq_index]
            
            #Subtract mean width from each bunch width:
            bunch_rel_width = np.empty_like(self.bunch_width_dt[:,turn_indices])
            for bunch in range(self.harmonic_number):
                bunch_rel_width[bunch,:] = np.hanning(self.bunch_width_dt[bunch,turn_indices].shape[0]) * \
                    (self.bunch_width_dt[bunch,turn_indices] - np.mean(self.bunch_width_dt[bunch,turn_indices]))
                        
            #Calculate normalised 2D FFT:
            fft_width = np.fft.rfft2(bunch_rel_width) * 4 / np.size(bunch_rel_width)

            #Find frequency peak, estimate of synchrotron frequency:
            max_width_freq_index = np.argmax(np.mean(np.abs(fft_width), axis=0))
            width_mode_spectrum = fft_width[:, max_width_freq_index]
            
            if plots:
                plt.figure('bunch_pos_img')
                plot_y, plot_x = np.meshgrid(self.turn_vec[turn_indices], np.arange(self.harmonic_number+1))
                plt.pcolormesh(plot_x, plot_y, bunch_rel_pos, cmap='hot', shading='flat')
                plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
                plt.title('Bunch position offset')
                plt.xlabel('Bunch')
                plt.ylabel('Turn')
                plt.colorbar()
                plt.rc('font', size=16)
                plt.savefig(self.dirname + '/bunch_pos_img.png')
                plt.close()
                
                plt.figure('bunch_width_img')
                plot_y, plot_x = np.meshgrid(self.turn_vec[turn_indices], np.arange(self.harmonic_number+1))
                plt.pcolormesh(plot_x, plot_y, bunch_rel_width, cmap='hot', shading='flat')
                plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
                plt.title('Bunch width offset')
                plt.xlabel('Bunch')
                plt.ylabel('Turn')
                plt.colorbar()
                plt.rc('font', size=16)
                plt.savefig(self.dirname + '/bunch_width_img.png')
                plt.close()
                
                plt.figure('bunch_pos_2dfft')  
                plot_y, plot_x = np.meshgrid(np.arange(fft_pos.shape[1]) * df, np.arange(self.harmonic_number+1))
                plt.pcolormesh(plot_x, plot_y, np.abs(fft_pos), cmap='hot', shading='flat')
                plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
                plt.xlabel('Mode')
                plt.ylabel('Frequency [1/turn]')
                plt.colorbar()
                plt.rc('font', size=16)
                plt.savefig(self.dirname + '/bunch_pos_2dfft.png')
                plt.close()
                
                plt.figure('bunch_width_2dfft')  
                plot_y, plot_x = np.meshgrid(np.arange(fft_width.shape[1]) * df, np.arange(self.harmonic_number+1))
                plt.pcolormesh(plot_x, plot_y, np.abs(fft_width), cmap='hot', shading='flat')
                plt.axis([plot_x.min(), plot_x.max(), plot_y.min(), plot_y.max()])
                plt.xlabel('Mode')
                plt.ylabel('Frequency [1/turn]')
                plt.colorbar()
                plt.rc('font', size=16)
                plt.savefig(self.dirname + '/bunch_width_2dfft.png')
                plt.close()
                
                            
        if plots:            
            plt.figure('bunch_pos_fft', figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.bar([x for x in range(self.harmonic_number)], np.absolute(pos_mode_spectrum))
            ax.set_xlabel("Mode")
            ax.set_ylabel("Amplitude [s]")
            plt.title('Bunch position oscillation')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(self.dirname + '/bunch_pos_modes.png')
            plt.close()
            
            plt.figure('bunch_width_fft')
            plt.bar([x for x in range(self.harmonic_number)], np.absolute(width_mode_spectrum))
            plt.xlabel("Mode")
            plt.ylabel("Amplitude [s]")
            plt.title('Bunch width oscillation')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.savefig(self.dirname + '/bunch_width_modes.png')
            plt.close()
            
            
        return [pos_mode_spectrum, width_mode_spectrum]
     
    def modes_vs_time(self, start_turn, end_turn, window, resolution, N_modes_plt, method='fft'):
        #Define time windows
        window_centres = np.round(np.arange(start_turn+window/2, end_turn-window/2, resolution))
        N_windows = window_centres.shape[0]

        #Calculate modes for each window
        pos_modes = np.empty([self.harmonic_number, N_windows])
        width_modes = np.empty([self.harmonic_number, N_windows])
        
        for i in range(N_windows):
            [pos_modes_curr, width_modes_curr] = \
                self.mode_analysis(window_centres[i]-window/2, window_centres[i]+window/2, False, method)
                
            pos_modes[:,i] = np.absolute(pos_modes_curr)
            width_modes[:,i] = np.absolute(width_modes_curr)
        
        plot_modes_vs_time(window_centres, pos_modes, width_modes,\
                                self.harmonic_number, N_modes_plt, self.dirname)
                
        return [window_centres, pos_modes, width_modes]
    
