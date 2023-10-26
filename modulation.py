import math

import numpy as np
from scipy.signal import hilbert

from signals import Signal


class Modulation:
    def __init__(self, **kwargs):
        try:
            self.main_signal = kwargs['main_signal']
            self.modeling_signal = kwargs['modeling_signal']
            self.modulated_result_signal = Signal(start_time=self.main_signal.start_time,
                                                  end_time=self.main_signal.end_time,
                                                  time_step=self.main_signal.time_step,
                                                  frequency=self.main_signal.frequency, amplitude=0)
            self.synthesized_signal = None
            self.filtered_signal = None
            self.enveloping_signal = None

        except KeyError:
            print("Key not found")

    def count_signal(self):
        """""count signal values"""""
        pass

    def count_r_spectrum(self):
        self.modulated_result_signal.count_r_spectrum()

    def count_full_spectrum(self):
        self.modulated_result_signal.count_full_spectrum()

    def cut_full_spectrum(self):
        m = max(self.modulated_result_signal.fft_result_spectrum)

        for i in range(len(self.modulated_result_signal.fft_result_spectrum)):
            if np.abs(self.modulated_result_signal.fft_result_spectrum[i]) < 0.25 * m:
                self.modulated_result_signal.fft_result_spectrum[i] = 0
            else:
                self.modulated_result_signal.fft_result_spectrum[i] = np.abs(
                    self.modulated_result_signal.fft_result_spectrum[i])
            if np.abs(self.modulated_result_signal.fft_result_spectrum[i]) == m:
                self.modulated_result_signal.fft_result_spectrum[i] = 0

    def spectrum_synthesys(self):
        self.synthesized_signal = Signal(start_time=self.modulated_result_signal.start_time,
                                         end_time=self.modulated_result_signal.end_time,
                                         time_step=self.modulated_result_signal.time_step,
                                         frequency=self.modulated_result_signal.frequency, amplitude=0)

        self.synthesized_signal.signal = np.fft.ifft(self.modulated_result_signal.fft_result_spectrum).real
        self.synthesized_signal.time_interval = self.modulated_result_signal.time_interval


class PhaseModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_r_spectrum()

    def count_signal(self):
        k = 20

        self.modulated_result_signal.signal = self.main_signal.amplitude * np.cos(2 * math.pi *
                                                                                  self.main_signal.frequency
                                                                                  * self.main_signal.time_interval + k * self.modeling_signal.signal)

        self.modulated_result_signal.time_interval = self.main_signal.time_interval


class FrequencyModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_r_spectrum()

    def count_signal(self):
        signal = []
        for i in range(len(self.main_signal.time_interval)):
            signal.append(
                self.modeling_signal.amplitude * math.cos(
                    2 * math.pi * (self.main_signal.frequency + self.modeling_signal.signal[
                        i] * self.main_signal.frequency / 2) *
                    self.main_signal.time_interval[i]))
        self.modulated_result_signal.signal = np.array(signal)

        self.modulated_result_signal.time_interval = self.main_signal.time_interval


class AmplitudeModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_r_spectrum()
        self.count_full_spectrum()
        self.cut_full_spectrum()
        self.spectrum_synthesys()
        self.filter_signal()

    def count_signal(self):
        self.modulated_result_signal.signal = self.main_signal.signal * self.modeling_signal.signal

        self.modulated_result_signal.time_interval = self.main_signal.time_interval

    def filter_signal(self):
        self.filtered_signal = Signal(start_time=self.modulated_result_signal.start_time,
                                      end_time=self.modulated_result_signal.end_time,
                                      time_step=self.modulated_result_signal.time_step,
                                      frequency=self.modulated_result_signal.frequency, amplitude=0)

        self.enveloping_signal = Signal(start_time=self.synthesized_signal.start_time,
                                        end_time=self.synthesized_signal.end_time,
                                        time_step=self.synthesized_signal.time_step,
                                        frequency=self.synthesized_signal.frequency, amplitude=0)

        self.enveloping_signal.signal = np.abs(hilbert(self.synthesized_signal.signal))
        self.enveloping_signal.time_interval = self.synthesized_signal.time_interval

        m = max(np.abs(self.synthesized_signal.signal)) * 0.35
        self.filtered_signal.signal = np.array([1 if i > m else 0 for i in self.enveloping_signal.signal])

        self.filtered_signal.time_interval = self.synthesized_signal.time_interval
