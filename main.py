import tkinter as tk
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
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
            self.start_time = None
            self.end_time = None
            self.time_step = None
            self.frequency = None
            self.amplitude = None

    @abstractmethod
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


class Modulation(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.main_signal = kwargs['main_signal']
            self.modeling_signal = kwargs['modeling_signal']
            self.time_step = self.modeling_signal.time_step
            self.count_signal_function()
            self.count_spectrum()

        except KeyError:
            print("Key not found")

    @abstractmethod
    def count_signal_function(self):
        """""count signal values"""""
        pass


class AmplitudeModulation(Modulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_signal_function(self):
        signal = []
        for i in range(len(self.main_signal.signal)):
            value = self.main_signal.signal[i] * self.modeling_signal.signal[i]
            signal.append(value)

        self.signal = np.array(signal)
        self.time_interval = self.main_signal.time_interval


class DrawSignal:

    def draw_modulation_signal(self):
        Hs = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                            frequency=7, amplitude=1)
        Ds = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                           frequency=5, amplitude=1)

        modulation = AmplitudeModulation(main_signal=Hs, modeling_signal=Ds)

        self.__draw_plots_modulation(Hs, Ds, modulation)

    @staticmethod
    def __draw_plots_modulation(main_signal: Signal, modeling_signal: Signal, modulation: Modulation):
        fig, axs = plt.subplots(nrows=4, ncols=2)
        fig.set_size_inches(15, 6)

        axs[0, 0].plot(main_signal.time_interval, main_signal.signal)

        axs[1, 0].plot(modeling_signal.time_interval, modeling_signal.signal)

        axs[2, 0].plot(modulation.time_interval, modulation.signal)

        axs[3, 0].plot(modulation.frequencies, np.abs(modulation.fft_result_signal))
        axs[3, 0].set_xlim(0, 50)

        plt.show()


class Window:
    root = tk.Tk()

    def __init__(self):
        self.root.title("signal graph")
        self.root.geometry('500x500')
        self.path = None
        self.root.config(background="light green")
        draw = DrawSignal()

        button_modulation = tk.Button(self.root, text="Draw amplitude modulation", command=draw.draw_modulation_signal,
                                      background="light pink")

        button_modulation.pack()

    def start(self):
        self.root.mainloop()


# При частотной модуляции
# (ЧМ, англ. FM - Frequency Modulation) несущий сигнал является более высокочастотным по отношению к информационному сигналу и амплитуда частотно-модулированного сигнала является неизменной.
# Частотно модулированный сигнал отличается высокой помехозащищенностью и используется для высококачественной передачи информации: в радиовещании, телевидении, радиотелефонии и др.


window = Window()
window.start()
