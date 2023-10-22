import math
import tkinter as tk
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftfreq
from scipy.fft import rfft, rfftfreq, fft
from scipy.integrate import simps


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

    def cut_r_spectrum(self):
        min_index = np.argmin(np.abs(self.modulated_result_signal.rfft_result_spectrum))
        max_index = np.argmax(np.abs(self.modulated_result_signal.rfft_result_spectrum))

        self.modulated_result_signal.rfft_result_spectrum[min_index] = 0
        self.modulated_result_signal.rfft_result_spectrum[max_index] = 0

    def cut_full_spectrum(self):
        min_index = np.argmin(np.abs(self.modulated_result_signal.fft_result_spectrum))
        max_index = np.argmax(np.abs(self.modulated_result_signal.fft_result_spectrum))

        self.modulated_result_signal.fft_result_spectrum[min_index] = 0
        self.modulated_result_signal.fft_result_spectrum[max_index] = 0

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
        self.cut_r_spectrum()
        self.spectrum_synthesys()

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
        self.cut_r_spectrum()
        self.spectrum_synthesys()

    def count_signal(self):
        k = 300

        modeling_int = [0, ]
        for i in range(1, len(self.modeling_signal.time_interval)):
            modeling_int.append(simps(self.modeling_signal.signal[:i], self.modeling_signal.time_interval[:i]))

        s_FM = []
        for (ti, m) in zip(self.modeling_signal.time_interval, modeling_int):
            s_FM.append(self.main_signal.amplitude * math.sin(2 * math.pi * self.main_signal.frequency * ti + k * m))

        self.modulated_result_signal.signal = np.array(s_FM)

        self.modulated_result_signal.time_interval = self.main_signal.time_interval


class AmplitudeModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_r_spectrum()
        self.count_full_spectrum()
        self.cut_r_spectrum()
        self.cut_full_spectrum()
        self.spectrum_synthesys()
        self.filter_signal()

    def count_signal(self):
        # self.modulated_result_signal.signal = self.main_signal.signal * (
        #         1 + self.modeling_signal.signal / self.main_signal.amplitude)

        self.modulated_result_signal.signal = self.main_signal.signal * self.modeling_signal.signal

        self.modulated_result_signal.time_interval = self.main_signal.time_interval

    def filter_signal(self):
        self.filtered_signal = Signal(start_time=self.modulated_result_signal.start_time,
                                      end_time=self.modulated_result_signal.end_time,
                                      time_step=self.modulated_result_signal.time_step,
                                      frequency=self.modulated_result_signal.frequency, amplitude=0)

        y = np.zeros_like(self.synthesized_signal.signal)
        win = 100
        for i in range(len(self.synthesized_signal.signal) - win + 1):
            y[i] = np.mean(self.synthesized_signal.signal[i:i + win])
        s = np.mean(y)
        self.filtered_signal.signal = np.array([1 if i >= s else 0 for i in y])

        self.filtered_signal.time_interval = self.synthesized_signal.time_interval


class DrawSignal:

    def draw_harmonic_signal(self):
        Hs_1Hz = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                                frequency=1, amplitude=1)

        Hs_2Hz = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                                frequency=2, amplitude=1)

        Hs_4Hz = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                                frequency=4, amplitude=1)

        Hs_8Hz = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                                frequency=8, amplitude=1)

        self.__draw_plots(Hs_1Hz, Hs_2Hz, Hs_4Hz, Hs_8Hz)

    def draw_digital_signal(self):
        Ds_1Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=1, amplitude=1)

        Ds_2Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=2, amplitude=1)

        Ds_4Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=4, amplitude=1)

        Ds_8Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=8, amplitude=1)

        self.__draw_plots(Ds_1Hz, Ds_2Hz, Ds_4Hz, Ds_8Hz)

    @staticmethod
    def __draw_plots(signal1Hz: Signal, signal2Hz: Signal, signal4Hz: Signal, signal8Hz: Signal):
        fig, axs = plt.subplots(nrows=4, ncols=2)
        fig.set_size_inches(15, 6)

        axs[0, 0].plot(signal1Hz.time_interval, signal1Hz.signal)
        axs[0, 1].set_xlim(0, 50)
        axs[0, 1].plot(signal1Hz.r_frequencies, np.abs(signal1Hz.rfft_result_spectrum))

        axs[1, 0].plot(signal2Hz.time_interval, signal2Hz.signal)
        axs[1, 1].set_xlim(0, 50)
        axs[1, 1].plot(signal2Hz.r_frequencies, np.abs(signal2Hz.rfft_result_spectrum))

        axs[2, 0].plot(signal4Hz.time_interval, signal4Hz.signal)
        axs[2, 1].set_xlim(0, 50)
        axs[2, 1].plot(signal4Hz.r_frequencies, np.abs(signal4Hz.rfft_result_spectrum))

        axs[3, 0].plot(signal8Hz.time_interval, signal8Hz.signal)
        axs[3, 1].set_xlim(0, 50)
        axs[3, 1].plot(signal8Hz.r_frequencies, np.abs(signal8Hz.rfft_result_spectrum))

        plt.show()

    def draw_phase_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                            frequency=30, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                           frequency=3, amplitude=1)

        modulation = PhaseModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    def draw_amplitude_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                            frequency=15, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                           frequency=3, amplitude=0.2)

        modulation = AmplitudeModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    def draw_frequency_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                            frequency=30, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                           frequency=10, amplitude=1)

        modulation = FrequencyModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    @staticmethod
    def __draw_plots_modulation(main_signal: Signal, modeling_signal: Signal, modulation: Modulation):
        fig, axs = plt.subplots(nrows=6, ncols=1)
        fig.set_size_inches(15, 6)

        axs[0].plot(main_signal.time_interval, main_signal.signal)

        axs[1].plot(modeling_signal.time_interval, modeling_signal.signal)

        axs[2].plot(modulation.modulated_result_signal.time_interval, modulation.modulated_result_signal.signal)

        axs[3].plot(modulation.modulated_result_signal.r_frequencies,
                    np.abs(modulation.modulated_result_signal.rfft_result_spectrum))
        axs[3].set_xlim(0, 50)

        axs[4].plot(modulation.synthesized_signal.time_interval, modulation.synthesized_signal.signal)

        axs[5].plot(modulation.filtered_signal.time_interval, modulation.filtered_signal.signal)

        plt.show()


class Window:
    root = tk.Tk()

    def __init__(self):
        self.root.title("signal graph")
        self.root.geometry('500x500')
        self.path = None
        self.root.config(background="light green")
        draw = DrawSignal()

        button_amplitude_modulation = tk.Button(self.root, text="Draw amplitude modulation",
                                                command=draw.draw_amplitude_modulation_signal,
                                                background="light pink")

        button_frequency_modulation = tk.Button(self.root, text="Draw frequency modulation",
                                                command=draw.draw_frequency_modulation_signal,
                                                background="light green")

        button_phase_modulation = tk.Button(self.root, text="Draw phase modulation",
                                            command=draw.draw_phase_modulation_signal,
                                            background="light blue")

        button_amplitude_modulation.pack()
        button_frequency_modulation.pack()
        button_phase_modulation.pack()

    def start(self):
        self.root.mainloop()


window = Window()
window.start()
