import tkinter as tk

import numpy as np
from matplotlib import pyplot as plt

from modulation import PhaseModulation, AmplitudeModulation, FrequencyModulation
from signals import Signal
from task_processor import DataProcessor


class DrawSignal:
    def __init__(self):
        self.t_processor = DataProcessor()

    def draw_harmonic_signal(self):
        self.__draw_plots(self.t_processor.harmonic_signals[0], self.t_processor.harmonic_signals[1],
                          self.t_processor.harmonic_signals[2], self.t_processor.harmonic_signals[3])

    def draw_digital_signal(self):
        self.__draw_plots(self.t_processor.digital_signals[0], self.t_processor.digital_signals[1],
                          self.t_processor.digital_signals[2], self.t_processor.digital_signals[3])

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

    def draw_modulation_signal(self):
        self.__draw_plots_modulation(self.t_processor.modulations.get("amplitude_modulation"),
                                     self.t_processor.modulations.get("frequency_modulation"),
                                     self.t_processor.modulations.get("phase_modulation"))

    def draw_spectrum_modulation(self):
        self.__draw_plots_modulation_spectrum(self.t_processor.modulations.get("amplitude_modulation"),
                                              self.t_processor.modulations.get("frequency_modulation"),
                                              self.t_processor.modulations.get("phase_modulation"))

    def draw_synth_signal(self):
        self.__draw_plots_synth_signal(self.t_processor.modulations.get("amplitude_modulation").synthesized_signal)

    def draw_filtered_signal(self):
        self.__draw_plots_filtered_signal(self.t_processor.modulations.get("amplitude_modulation").filtered_signal)

    @staticmethod
    def __draw_plots_modulation(ampl_modulation: AmplitudeModulation, freq_modulation: FrequencyModulation,
                                phase_modulation: PhaseModulation):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        fig.set_size_inches(15, 6)

        axs[0].plot(ampl_modulation.modulated_result_signal.time_interval,
                    ampl_modulation.modulated_result_signal.signal)
        axs[1].plot(freq_modulation.modulated_result_signal.time_interval,
                    freq_modulation.modulated_result_signal.signal)
        axs[2].plot(phase_modulation.modulated_result_signal.time_interval,
                    phase_modulation.modulated_result_signal.signal)

        plt.show()

    @staticmethod
    def __draw_plots_synth_signal(synth_signal: Signal):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(15, 3)

        axs.plot(synth_signal.time_interval, synth_signal.signal)

        plt.show()

    @staticmethod
    def __draw_plots_filtered_signal(filtered_signal: Signal):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(15, 3)

        axs.plot(filtered_signal.time_interval, filtered_signal.signal)

        plt.show()

    @staticmethod
    def __draw_plots_modulation_spectrum(ampl_modulation: AmplitudeModulation, freq_modulation: FrequencyModulation,
                                         phase_modulation: PhaseModulation):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        fig.set_size_inches(15, 6)

        axs[0].plot(ampl_modulation.modulated_result_signal.r_frequencies,
                    np.abs(ampl_modulation.modulated_result_signal.rfft_result_spectrum))

        axs[1].plot(freq_modulation.modulated_result_signal.r_frequencies,
                    np.abs(freq_modulation.modulated_result_signal.rfft_result_spectrum))

        axs[2].plot(phase_modulation.modulated_result_signal.r_frequencies,
                    np.abs(phase_modulation.modulated_result_signal.rfft_result_spectrum))

        for i in range(3):
            axs[i].set_xlim(0, 60)

        plt.show()


class Window:
    root = tk.Tk()

    def __init__(self):
        self.root.title("signal graph")
        self.root.geometry('500x500')
        self.path = None
        self.root.config(background="light green")
        draw = DrawSignal()

        button_harmonic_signal = tk.Button(self.root, text="Draw harmonic signal", command=draw.draw_harmonic_signal,
                                           background="red")

        button_digital_signal = tk.Button(self.root, text="Draw digital signal", command=draw.draw_digital_signal,
                                          background="blue")

        button_modulation = tk.Button(self.root, text="Draw modulation",
                                      command=draw.draw_modulation_signal,
                                      background="light pink")
        button_modulation_spectrum = tk.Button(self.root, text="Draw modulation spectrum",
                                               command=draw.draw_spectrum_modulation,
                                               background="white")

        button_spectrum_and_synthesys = tk.Button(self.root, text="Draw synth signal", command=draw.draw_synth_signal,
                                                  background="orange")

        button_filtered_signal = tk.Button(self.root, text="Draw filtered signal", command=draw.draw_filtered_signal,
                                           background="light green")

        button_harmonic_signal.pack()
        button_digital_signal.pack()
        button_modulation.pack()
        button_modulation_spectrum.pack()
        button_spectrum_and_synthesys.pack()
        button_filtered_signal.pack()

    def start(self):
        self.root.mainloop()


window = Window()
window.start()
