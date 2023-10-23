from abc import ABC

import numpy as np
from numpy.fft import fftfreq
from scipy.fft import rfft, rfftfreq, fft


class Signal(ABC):

    def __init__(self, **kwargs):
        try:
            self.r_frequencies = None
            self.rfft_result_spectrum = None
            self.signal = None
            self.time_interval = None
            self.start_time = kwargs['start_time']
            self.end_time = kwargs['end_time']
            self.time_step = kwargs['time_step']
            self.frequency = kwargs['frequency']
            self.amplitude = kwargs['amplitude']
            self.fft_result_spectrum = None
            self.frequencies = None
        except KeyError:
            print("unexpected key")

    def count_signal_function(self):
        """""count signal values"""""
        pass

    def count_r_spectrum(self):
        """""count r_spectrum of current signal, can be override"""""
        r_spectrum = rfft(self.signal)
        r_spectrum[0] = 0
        freqs = rfftfreq(len(self.time_interval), self.time_step)

        self.rfft_result_spectrum = r_spectrum
        self.r_frequencies = freqs

    def count_full_spectrum(self):
        """""count full spectrum of current signal, can be override"""""
        spectrum = fft(self.signal)
        freqs = fftfreq(len(self.time_interval), self.time_step)

        self.fft_result_spectrum = spectrum
        self.frequencies = freqs


class DigitalSignal(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal_function()
        self.count_r_spectrum()

    def count_signal_function(self):
        time_interval = np.arange(self.start_time, self.end_time, self.time_step)

        signal = []
        for time_moment in time_interval:
            func_value = np.sign(np.sin(2 * np.pi * self.frequency * time_moment))
            value = 0 if func_value < 0 else func_value
            signal.append(value)
        signal = np.array(signal)

        self.time_interval = time_interval
        self.signal = signal


class HarmonicSignal(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal_function()
        self.count_r_spectrum()

    def count_signal_function(self):
        time_interval = np.arange(self.start_time, self.end_time, self.time_step)

        signal = self.amplitude * np.sin(2 * np.pi * self.frequency * time_interval)

        self.time_interval = time_interval
        self.signal = signal
