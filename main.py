import math
import tkinter as tk
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.fft import rfft, rfftfreq


class Signal(ABC):

    def __init__(self, **kwargs):
        try:
            self.frequencies = None
            self.fft_result_signal = None
            self.signal = None
            self.time_interval = None
            self.start_time = kwargs['start_time']
            self.end_time = kwargs['end_time']
            self.time_step = kwargs['time_step']
            self.frequency = kwargs['frequency']
            self.amplitude = kwargs['amplitude']
        except KeyError:
            print("unexpected key")

    def count_signal_function(self):
        """""count signal values"""""
        pass

    def count_spectrum(self):
        """""count spectrum of current signal, can be override"""""
        fft_result_signal = rfft(self.signal)
        frequencies = rfftfreq(len(self.time_interval), self.time_step)

        self.fft_result_signal = fft_result_signal
        self.frequencies = frequencies


class DigitalSignal(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal_function()
        self.count_spectrum()

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
        self.count_spectrum()

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
            self.result_signal = Signal(start_time=self.main_signal.start_time, end_time=self.main_signal.end_time,
                                        time_step=self.main_signal.time_step,
                                        frequency=self.main_signal.frequency, amplitude=0)

        except KeyError:
            print("Key not found")

    def count_signal(self):
        """""count signal values"""""
        pass


class FrequencyModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_spectrum()

    def count_signal(self):
        # signal = 1 * np.sin(2 * math.pi * 2 * self.main_signal.time_interval)
        b = 3

        self.result_signal.signal = np.sin(2 * math.pi * self.main_signal.frequency * self.main_signal.time_interval
                                           + b * self.modeling_signal.signal)

        self.result_signal.time_interval = self.main_signal.time_interval

    def count_spectrum(self):
        self.result_signal.count_spectrum()


class AmplitudeModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_signal()
        self.count_spectrum()

    def count_signal(self):
        # signal = 1 * np.sin(2 * math.pi * 2 * self.main_signal.time_interval)

        self.result_signal.signal = self.main_signal.signal * (
                1 + self.modeling_signal.signal / self.main_signal.amplitude)

        self.result_signal.time_interval = self.main_signal.time_interval

    def count_spectrum(self):
        self.result_signal.count_spectrum()


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
        axs[0, 1].plot(signal1Hz.frequencies, np.abs(signal1Hz.fft_result_signal))

        axs[1, 0].plot(signal2Hz.time_interval, signal2Hz.signal)
        axs[1, 1].set_xlim(0, 50)
        axs[1, 1].plot(signal2Hz.frequencies, np.abs(signal2Hz.fft_result_signal))

        axs[2, 0].plot(signal4Hz.time_interval, signal4Hz.signal)
        axs[2, 1].set_xlim(0, 50)
        axs[2, 1].plot(signal4Hz.frequencies, np.abs(signal4Hz.fft_result_signal))

        axs[3, 0].plot(signal8Hz.time_interval, signal8Hz.signal)
        axs[3, 1].set_xlim(0, 50)
        axs[3, 1].plot(signal8Hz.frequencies, np.abs(signal8Hz.fft_result_signal))

        plt.show()

    def draw_amplitude_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                            frequency=30, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                           frequency=2, amplitude=0.2)

        modulation = AmplitudeModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    def draw_frequency_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=0.5, time_step=0.001,
                            frequency=100, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=0.5, time_step=0.001,
                           frequency=10, amplitude=0.2)

        modulation = FrequencyModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    @staticmethod
    def __draw_plots_modulation(main_signal: Signal, modeling_signal: Signal, modulation: Modulation):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        fig.set_size_inches(15, 6)

        axs[0].plot(main_signal.time_interval, main_signal.signal)

        axs[1].plot(modeling_signal.time_interval, modeling_signal.signal)

        axs[2].plot(modulation.result_signal.time_interval, modulation.result_signal.signal)

        # axs[3].plot(modulation.result_signal.frequencies, np.abs(modulation.result_signal.fft_result_signal))
        # axs[3].set_xlim(0, 50)

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

        button_amplitude_modulation.pack()
        button_frequency_modulation.pack()

    def start(self):
        self.root.mainloop()


# При частотной модуляции
# (ЧМ, англ. FM - Frequency Modulation) несущий сигнал является более высокочастотным по отношению к информационному сигналу и амплитуда частотно-модулированного сигнала является неизменной.
# Частотно модулированный сигнал отличается высокой помехозащищенностью и используется для высококачественной передачи информации: в радиовещании, телевидении, радиотелефонии и др.


window = Window()
window.start()
