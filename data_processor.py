from modulation import AmplitudeModulation, FrequencyModulation, PhaseModulation
from signals import HarmonicSignal, DigitalSignal


class DataProcessor:
    def __init__(self):
        self.harmonic_signals = []
        self.digital_signals = []
        self.modulations = None
        self.init_task1()
        self.init_task2()

    def init_task1(self):
        Hs_1Hz = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                frequency=1, amplitude=1)

        Hs_2Hz = HarmonicSignal(start_time=0, end_time=5, time_step=0.001,
                                frequency=2, amplitude=1)

        Hs_4Hz = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                frequency=4, amplitude=1)

        Hs_8Hz = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                frequency=8, amplitude=1)

        self.harmonic_signals.append(Hs_1Hz)
        self.harmonic_signals.append(Hs_2Hz)
        self.harmonic_signals.append(Hs_4Hz)
        self.harmonic_signals.append(Hs_8Hz)

        Ds_1Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=1, amplitude=1)

        Ds_2Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=2, amplitude=1)

        Ds_4Hz = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                               frequency=4, amplitude=1)

        Ds_8Hz = DigitalSignal(start_time=0, end_time=5, time_step=0.001,
                               frequency=8, amplitude=1)

        self.digital_signals.append(Ds_1Hz)
        self.digital_signals.append(Ds_2Hz)
        self.digital_signals.append(Ds_4Hz)
        self.digital_signals.append(Ds_8Hz)

    def init_task2(self):
        ampl_Hs = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                 frequency=64, amplitude=2)
        ampl_Ds = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                                frequency=4, amplitude=1)

        ampl_modulation = AmplitudeModulation(main_signal=ampl_Hs, modeling_signal=ampl_Ds)

        freq_Hs = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                 frequency=64, amplitude=1)
        freq_Ds = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                                frequency=4, amplitude=1)

        freq_modulation = FrequencyModulation(main_signal=freq_Hs, modeling_signal=freq_Ds)

        phase_Hs = HarmonicSignal(start_time=0, end_time=1, time_step=0.001,
                                  frequency=64, amplitude=1)
        phase_Ds = DigitalSignal(start_time=0, end_time=1, time_step=0.001,
                                 frequency=4, amplitude=1)

        phase_modulation = PhaseModulation(main_signal=phase_Hs, modeling_signal=phase_Ds)

        self.modulations = {"amplitude_modulation": ampl_modulation,
                            "frequency_modulation": freq_modulation,
                            "phase_modulation": phase_modulation}
