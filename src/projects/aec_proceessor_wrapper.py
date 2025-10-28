# aec_processor.py
from pipecat.processors.frame_processor import FrameProcessor
import numpy as np

class AECProcessorWrapper(FrameProcessor):
    """
    FrameProcessor that applies echo cancellation using EchoCanceller.
    """
    def __init__(self, aec, playback_buffer):
        super().__init__()
        self.aec = aec
        self.playback_buffer = playback_buffer

    async def process(self, frame):
        # Only process frames with audio data
        if hasattr(frame, "data"):
            mic_signal = np.array(frame.data, dtype=np.int16)
            playback_signal = np.array(self.playback_buffer, dtype=np.int16)

            # Apply echo cancellation
            processed = self.aec.process(mic_signal, playback_signal)

            # Update frame and buffer
            frame.data = processed
            self.playback_buffer.extend(processed)

        return frame
