import math

import numpy as np
from scipy import signal
from scipy.signal import medfilt

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
        min_index = np.argmin(np.abs(self.modulated_result_signal.fft_result_spectrum))
        max_index = np.argmax(np.abs(self.modulated_result_signal.fft_result_spectrum))

        self.modulated_result_signal.fft_result_spectrum[min_index] = 0
        self.modulated_result_signal.fft_result_spectrum[max_index] = 0

        for i in range(len(self.modulated_result_signal.frequencies)):
            if self.modulated_result_signal.frequencies[i] < self.modulated_result_signal.frequencies[max_index] / 2:
                self.modulated_result_signal.frequencies[i] = 0

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
        # k = 100 * math.pi
        #
        # modeling_int = [0, ]
        # for i in range(1, len(self.modeling_signal.time_interval)):
        #     modeling_int.append(simps(self.modeling_signal.signal[:i], self.modeling_signal.time_interval[:i]))
        #
        # s_FM = []
        # for (ti, m) in zip(self.modeling_signal.time_interval, modeling_int):
        #     s_FM.append(self.main_signal.amplitude * math.sin(2 * math.pi * self.main_signal.frequency * ti + k * m))
        #
        # self.modulated_result_signal.signal = np.array(s_FM)
        #
        # self.modulated_result_signal.time_interval = self.main_signal.time_interval

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

        # y = np.zeros_like(self.synthesized_signal.signal)
        # win = 200
        # for i in range(len(self.synthesized_signal.signal) - win + 1):
        #     y[i] = np.mean(self.synthesized_signal.signal[i:i + win])
        # s = 1.2 * np.mean(y)

        #self.filtered_signal.signal = np.array([1 if i >= s else 0 for i in y])

        sig = medfilt(self.synthesized_signal.signal, kernel_size=131)
        s = 1.5 * np.mean(sig)
        self.filtered_signal.signal = np.array([1 if i >= s else 0 for i in sig])

        self.filtered_signal.time_interval = self.synthesized_signal.time_interval
