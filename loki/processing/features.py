"""Classes and functions for generating features"""
from librosa import power_to_db
import numpy as np

def compute_decibels(data, freq=44100):
    """Compute the total decibels from an audio waveform

    Compute the power by taking the square of the waveform. If the
    audio is binaural, then sum up the power of each audio channel.

    Arguments:
    ----------
    data -- loki.VideoClips
        Object containing the VideoClips

    Keyword Arguments:
    ------------------
    freq -- int -- default=44100
        Frequency at which to extract the audio.

    Return:
    -------
    decibels -- list[np.ndarray]:
        The loudness over time of each inputted clip.
    """
    decibels = []

    all_audio = data.compute_audio_waveform(freq=freq)
    
    for binaural in all_audio:
        power = binaural ** 2 # square for the power
        #sum up binaural audio channel
        if power.ndim == 2:
            power = np.sum(power, axis=1)

        decibel = power_to_db(power)
        decibels.append(decibel)

    return decibels
