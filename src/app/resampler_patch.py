from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
import numpy as np

async def fixed_resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
    """Resample audio data using soxr.ResampleStream resampler library.

    Args:
        audio: Input audio data as raw bytes (16-bit signed integers).
        in_rate: Original sample rate in Hz.
        out_rate: Target sample rate in Hz.

    Returns:
        Resampled audio data as raw bytes (16-bit signed integers).
    """
    if in_rate == out_rate:
        return audio

    self._maybe_initialize_sox_stream(in_rate, out_rate)
    # Ensure the input buffer length is a multiple of 2 bytes (size of int16)
    # If it's not, pad with a null byte to avoid ValueError from numpy.frombuffer.
    if not isinstance(audio, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(audio)
    else:
        audio_bytes = bytes(audio)
    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes + b'\x00'
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    resampled_audio = self._soxr_stream.resample_chunk(audio_data)
    result = resampled_audio.astype(np.int16).tobytes()
    return result

# apply patch
SOXRStreamAudioResampler.resample = fixed_resample