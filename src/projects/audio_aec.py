import numpy as np
import pyaec

class EchoCanceller:
    def __init__(self, frame_length=160, filter_length=2000):
        """
        frame_length: samples per frame (e.g., 160 for 16kHz 10ms)
        filter_length: internal filter length for AEC
        """
        self.frame_length = frame_length
        self.aec = pyaec.Aec( frame_size=frame_length, filter_length=filter_length, sample_rate=16000)

    def process(self, mic_signal, playback_signal):
        """
        mic_signal: numpy array of int16 from microphone
        playback_signal: numpy array of int16 from speaker output
        returns: echo-cancelled mic signal
        """
        if mic_signal is None or playback_signal is None:
            return mic_signal
        # Ensure correct type
        mic_signal = np.array(mic_signal, dtype=np.int16)
        playback_signal = np.array(playback_signal, dtype=np.int16)
        return self.aec.filter(mic_signal, playback_signal)
